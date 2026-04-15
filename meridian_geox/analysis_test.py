# Copyright 2026 The Meridian GeoX Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for GeoX analysis library."""

from absl.testing import absltest
from absl.testing import parameterized
import jax.numpy as jnp
from meridian_geox import analysis
from meridian_geox import api
import numpy as np
import pandas as pd


class AnalysisTest(parameterized.TestCase):

  def _create_sample_data(
      self, n_days=15, n_geos=10, include_spend=False
  ) -> pd.DataFrame:
    """Helper to create sample data for tests."""
    dates = pd.date_range('2024-01-01', periods=n_days)
    geos = [f'G{i}' for i in range(1, n_geos + 1)]
    data_rows = []
    for d in dates:
      for g in geos:
        row = {api.DATE: d, api.LOCATION: g, api.CONVERSIONS: 100.0}
        if include_spend:
          row[api.SPEND] = 50.0
        data_rows.append(row)
    return pd.DataFrame(data_rows)

  def test_prepare_data_filtering(self):
    data = self._create_sample_data(n_days=5, n_geos=3)
    # G3 is excluded, and 2024-01-05 is excluded.
    design_obj = api.Design(
        treatment_geos={'1': {'G1'}},
        control_geos={'G2'},
        excluded_geos={'G3'},
    )
    config = api.AnalysisConfig(
        methodology=api.Methodology.TBR,
        design=design_obj,
        analysis_start_date=pd.Timestamp('2024-01-01'),
        analysis_end_date=pd.Timestamp('2024-01-05'),
        excluded_dates={pd.Timestamp('2024-01-05')},
    )

    filtered_data = analysis._prepare_data(data, config)

    self.assertNotIn('G3', filtered_data[api.LOCATION].unique())
    self.assertNotIn(
        pd.Timestamp('2024-01-05'), filtered_data[api.DATE].unique()
    )
    self.assertLen(filtered_data[api.LOCATION].unique(), 2)
    self.assertLen(filtered_data[api.DATE].unique(), 4)

  def test_get_time_series(self):
    data = self._create_sample_data(n_days=10, n_geos=2)
    config = api.AnalysisConfig(
        methodology=api.Methodology.TBR,
        design=api.Design(
            treatment_geos={}, control_geos=set(), excluded_geos=set()
        ),
        analysis_start_date=pd.Timestamp('2024-01-06'),
        analysis_end_date=pd.Timestamp('2024-01-08'),
    )

    time_series = analysis._get_time_series(data, api.CONVERSIONS, config)

    # Pretest: 2024-01-01 to 2024-01-05 (5 days)
    # Test: 2024-01-06 to 2024-01-08 (3 days)
    self.assertEqual(time_series.pretest.shape, (5, 2))
    self.assertEqual(time_series.test.shape, (3, 2))
    self.assertLen(time_series.dates, 3)
    self.assertEqual(time_series.dates[0], pd.Timestamp('2024-01-06'))

  def test_prepare_design_config_overrides(self):
    design_config = api.DesignConfig(
        experiment_duration=2, alpha=0.05, n_aa_test_iterations=100
    )
    design_obj = api.Design(
        treatment_geos={},
        control_geos=set(),
        excluded_geos=set(),
        design_config=design_config,
    )
    # Analysis config has no alpha/test_type, should be filled from design.
    config = api.AnalysisConfig(
        methodology=api.Methodology.TBR,
        design=design_obj,
        analysis_start_date=pd.Timestamp('2024-01-01'),
        analysis_end_date=pd.Timestamp('2024-01-05'),
    )

    prepared_config = analysis._prepare_design_config(config)

    self.assertEqual(prepared_config.n_candidates, 100)
    self.assertEqual(config.alpha, 0.05)
    self.assertEqual(config.test_type, api.TestType.TWO_SIDED)

  def test_get_experiment_type(self):
    # Case 1: Single enum
    config = api.DesignConfig(
        experiment_duration=1, experiment_types=api.ExperimentType.GO_DARK
    )
    self.assertEqual(
        analysis._get_experiment_type(config), api.ExperimentType.GO_DARK
    )

    # Case 2: List with one element
    config = api.DesignConfig(
        experiment_duration=1, experiment_types=[api.ExperimentType.HOLDBACK]
    )
    self.assertEqual(
        analysis._get_experiment_type(config), api.ExperimentType.HOLDBACK
    )

    # Case 3: Invalid list
    config = api.DesignConfig(
        experiment_duration=1,
        experiment_types=[
            api.ExperimentType.GO_DARK,
            api.ExperimentType.HOLDBACK,
        ],
    )
    with self.assertRaisesRegex(ValueError, 'single experiment type'):
      analysis._get_experiment_type(config)

  def test_get_treatment_mask(self):
    data = self._create_sample_data(n_days=1, n_geos=4)
    # Geos are G1, G2, G3, G4.
    design_obj = api.Design(
        treatment_geos={'cell_1': {'G1', 'G3'}},
        control_geos={'G2', 'G4'},
        excluded_geos=set(),
    )

    treatment = analysis._get_treatment_mask(data, design_obj)

    # Sorted geos: G1, G2, G3, G4. Treatment: G1, G3.
    # Expected mask: [1, 0, 1, 0]
    np.testing.assert_array_equal(treatment.mask, jnp.array([1, 0, 1, 0]))
    self.assertEqual(treatment.geos, ['G1', 'G3'])

  def test_get_full_mask(self):
    treatment_mask = jnp.array([1, 0, 0, 1, 0])
    # control indices are [1, 2, 4]
    placebo_mask = jnp.array([1, 1, 0])
    full_mask = analysis._get_full_mask(treatment_mask, placebo_mask)
    # Expected: [0, 1, 1, 0, 0]
    # indices where treatment_mask == 0: 1, 2, 4
    # full_mask[1] = 1, full_mask[2] = 1, full_mask[4] = 0
    np.testing.assert_array_equal(full_mask, jnp.array([0, 1, 1, 0, 0]))

  def test_analyze_random_assignment(self):
    data = self._create_sample_data(n_days=15, n_geos=6)

    design_config = api.DesignConfig(
        experiment_duration=3,
        geo_assignment_rule=api.GeoAssignmentRule.RANDOM,
        seed=42,
        n_candidates=5,
        n_aa_test_iterations=10,
    )

    study_design = api.Design(
        treatment_geos={'1': {'G1', 'G2'}},
        control_geos={f'G{i}' for i in range(3, 7)},
        excluded_geos=set(),
        design_config=design_config,
        constraints=api.Constraints(max_conversions_percent=0.5),
    )

    config = api.AnalysisConfig(
        methodology=api.Methodology.TBR,
        design=study_design,
        analysis_start_date=pd.Timestamp('2024-01-11'),
        analysis_end_date=pd.Timestamp('2024-01-15'),
        alpha=0.1,
        test_type=api.TestType.TWO_SIDED,
    )

    result = analysis.analyze(data, config)

    self.assertIsInstance(result, api.AnalysisResult)
    self.assertIn('1', result.results)
    metrics = result.results['1']
    self.assertIsInstance(metrics, api.AnalysisMetrics)
    self.assertAlmostEqual(metrics.lift.point_estimate, 0.0, places=1)
    self.assertAlmostEqual(metrics.percent_lift.point_estimate, 0.0, places=1)
    self.assertEqual(
        list(metrics.cumulative_lift_estimates.columns),
        ['lift', 'lift_lower_bound', 'lift_upper_bound'],
    )
    self.assertLen(metrics.cumulative_lift_estimates, 5)
    self.assertAlmostEqual(
        metrics.cumulative_lift_estimates['lift'].iloc[-1],
        metrics.lift.point_estimate,
        places=5,
    )

  def test_analyze_stratified_sampling(self):
    data = self._create_sample_data(n_days=15, n_geos=10)

    design_config = api.DesignConfig(
        experiment_duration=3,
        geo_assignment_rule=api.GeoAssignmentRule.STRATIFIED_SAMPLING,
        seed=42,
        n_candidates=5,
        n_aa_test_iterations=10,
    )

    study_design = api.Design(
        treatment_geos={'1': {'G1'}},
        control_geos={f'G{i}' for i in range(2, 11)},
        excluded_geos=set(),
        design_config=design_config,
        constraints=api.Constraints(max_conversions_percent=0.5),
        geo_stratum_labels=jnp.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1]),
    )

    config = api.AnalysisConfig(
        methodology=api.Methodology.TBR,
        design=study_design,
        analysis_start_date=pd.Timestamp('2024-01-11'),
        analysis_end_date=pd.Timestamp('2024-01-15'),
        alpha=0.1,
        test_type=api.TestType.TWO_SIDED,
    )

    result = analysis.analyze(data, config)

    self.assertIsInstance(result, api.AnalysisResult)
    self.assertIn('1', result.results)
    metrics = result.results['1']
    self.assertAlmostEqual(metrics.lift.point_estimate, 0.0, places=1)
    self.assertAlmostEqual(metrics.percent_lift.point_estimate, 0.0, places=1)
    self.assertEqual(
        list(metrics.cumulative_lift_estimates.columns),
        ['lift', 'lift_lower_bound', 'lift_upper_bound'],
    )

  def test_analyze_unsupported_methodology(self):
    design_config = api.DesignConfig(experiment_duration=2)
    study_design = api.Design(
        treatment_geos={'1': {'G1'}},
        control_geos={'G2'},
        excluded_geos=set(),
        design_config=design_config,
        constraints=api.Constraints(),
    )
    # SDID is not supported yet.
    config = api.AnalysisConfig(
        methodology=api.Methodology.SDID,
        design=study_design,
        analysis_start_date=pd.Timestamp('2024-01-06'),
        analysis_end_date=pd.Timestamp('2024-01-10'),
        alpha=0.1,
        test_type=api.TestType.TWO_SIDED,
    )

    with self.assertRaisesRegex(ValueError, 'Unsupported methodology'):
      analysis.analyze(pd.DataFrame(), config)

  def test_analyze_excluded_geos(self):
    # Setup with more geos to avoid "Not enough geos" error in placebo
    # detection.
    data = self._create_sample_data(n_days=15, n_geos=10)

    design_config = api.DesignConfig(
        experiment_duration=2,
        geo_assignment_rule=api.GeoAssignmentRule.RANDOM,
        seed=42,
        n_candidates=3,
        n_aa_test_iterations=5,
    )

    # G10 is excluded
    study_design = api.Design(
        treatment_geos={'1': {'G1'}},
        control_geos={f'G{i}' for i in range(2, 10)},
        excluded_geos={'G10'},
        design_config=design_config,
        constraints=api.Constraints(
            excluded_geos={'G10'}, max_conversions_percent=0.5
        ),
    )

    config = api.AnalysisConfig(
        methodology=api.Methodology.TBR,
        design=study_design,
        analysis_start_date=pd.Timestamp('2024-01-11'),
        analysis_end_date=pd.Timestamp('2024-01-15'),
        alpha=0.1,
        test_type=api.TestType.TWO_SIDED,
    )

    result = analysis.analyze(data, config)

    self.assertIsInstance(result, api.AnalysisResult)
    self.assertIn('1', result.results)
    metrics = result.results['1']
    self.assertAlmostEqual(metrics.lift.point_estimate, 0.0, places=1)
    self.assertAlmostEqual(metrics.percent_lift.point_estimate, 0.0, places=1)


if __name__ == '__main__':
  absltest.main()
