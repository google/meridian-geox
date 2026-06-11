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
      for i, g in enumerate(geos):
        val = 100.0 + i * 10.0 + (d.day % 7) * 5.0
        row = {api.DATE: d, api.LOCATION: g, api.CONVERSIONS: val}
        if include_spend:
          row[api.SPEND] = val * 0.5
        data_rows.append(row)
    return pd.DataFrame(data_rows)

  def test_prepare_data_filtering(self):
    data = self._create_sample_data(n_days=5, n_geos=3)
    # G3 is excluded, and 2024-01-05 is excluded.
    design_obj = api.Design(
        designs={
            'cell_1': api.PerCellDesign(
                treatment_geos={'G1'},
                minimum_detectable_effect=0.0,
                design_implied_cpic=0.0,
                p_value=0.0,
                budget=0.0,
            )
        },
        control_geos={'G2'},
        excluded_geos={'G3'},
    )
    config = api.AnalysisConfig(
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
        design=api.Design(designs={}, control_geos=set(), excluded_geos=set()),
        analysis_start_date=pd.Timestamp('2024-01-06'),
        analysis_end_date=pd.Timestamp('2024-01-08'),
    )

    time_series = analysis._get_time_series(data, api.CONVERSIONS, config)

    # Pretest: 2024-01-01 to 2024-01-05 (5 days)
    # Test: 2024-01-06 to 2024-01-08 (3 days)
    self.assertEqual(time_series.pretest.shape, (5, 2))
    self.assertEqual(time_series.test.shape, (3, 2))
    self.assertLen(time_series.pretest_dates, 5)
    self.assertLen(time_series.test_dates, 3)
    self.assertEqual(time_series.test_dates[0], pd.Timestamp('2024-01-06'))

  def test_get_time_series_with_gap(self):
    data = self._create_sample_data(n_days=10, n_geos=2)
    config = api.AnalysisConfig(
        design=api.Design(designs={}, control_geos=set(), excluded_geos=set()),
        analysis_start_date=pd.Timestamp('2024-01-08'),
        analysis_end_date=pd.Timestamp('2024-01-10'),
        pretest_end_date=pd.Timestamp('2024-01-04'),
    )

    time_series = analysis._get_time_series(data, api.CONVERSIONS, config)

    # Pretest: 2024-01-01 to 2024-01-04 (4 days)
    # Test: 2024-01-08 to 2024-01-10 (3 days)
    # Gap: 2024-01-05 to 2024-01-07
    self.assertEqual(time_series.pretest.shape, (4, 2))
    self.assertEqual(time_series.test.shape, (3, 2))
    self.assertLen(time_series.pretest_dates, 4)
    self.assertLen(time_series.test_dates, 3)
    self.assertEqual(time_series.pretest_dates[-1], pd.Timestamp('2024-01-04'))
    self.assertEqual(time_series.test_dates[0], pd.Timestamp('2024-01-08'))

  def test_prepare_design_config_overrides(self):
    design_config = api.DesignConfig(
        experiment_duration=2,
        experiment_types=api.ExperimentType.HOLDBACK,
        alpha=0.05,
        n_aa_test_iterations=100,
    )
    design_obj = api.Design(
        designs={},
        control_geos=set(),
        excluded_geos=set(),
        design_config=design_config,
    )
    # Analysis config has no alpha/test_type, should be filled from design.
    config = api.AnalysisConfig(
        design=design_obj,
        analysis_start_date=pd.Timestamp('2024-01-01'),
        analysis_end_date=pd.Timestamp('2024-01-05'),
    )

    prepared_config = analysis._prepare_design_config(pd.DataFrame(), config)

    self.assertEqual(prepared_config.n_candidates, 100)
    self.assertEqual(config.alpha, 0.05)
    self.assertEqual(config.test_type, api.TestType.TWO_SIDED)

  def test_get_experiment_types(self):
    # Case 1: Single enum
    config = api.DesignConfig(
        experiment_duration=1, experiment_types=api.ExperimentType.GO_DARK
    )
    self.assertEqual(
        analysis._get_experiment_types(config),
        {api.CELL_1: api.ExperimentType.GO_DARK},
    )

    # Case 2: Singleton map
    config = api.DesignConfig(
        experiment_duration=1,
        experiment_types={'cell_1': api.ExperimentType.HOLDBACK},
    )
    self.assertEqual(
        analysis._get_experiment_types(config),
        {'cell_1': api.ExperimentType.HOLDBACK},
    )

  def test_get_treatment_mask(self):
    data = self._create_sample_data(n_days=1, n_geos=4)
    # Geos are G1, G2, G3, G4.
    design_obj = api.Design(
        designs={
            'cell_1': api.PerCellDesign(
                treatment_geos={'G1', 'G3'},
                minimum_detectable_effect=0.0,
                design_implied_cpic=0.0,
                p_value=0.0,
                budget=0.0,
            )
        },
        control_geos={'G2', 'G4'},
        excluded_geos=set(),
    )

    treatment = analysis._get_treatment_mask(data, design_obj)

    # Sorted geos: G1, G2, G3, G4. Treatment: G1, G3.
    # Expected mask: [1, 0, 1, 0]
    np.testing.assert_array_equal(treatment.mask, jnp.array([1, 0, 1, 0]))
    self.assertEqual(treatment.geos, ['G1', 'G3'])

  def test_get_treatment_mask_multicell(self):
    data = self._create_sample_data(n_days=1, n_geos=5)
    # Geos are G1, G2, G3, G4, G5.
    design_obj = api.Design(
        designs={
            'cell_1': api.PerCellDesign(
                treatment_geos={'G1', 'G3'},
                minimum_detectable_effect=0.0,
                design_implied_cpic=0.0,
                p_value=0.0,
                budget=0.0,
            ),
            'cell_2': api.PerCellDesign(
                treatment_geos={'G4'},
                minimum_detectable_effect=0.0,
                design_implied_cpic=0.0,
                p_value=0.0,
                budget=0.0,
            ),
        },
        control_geos={'G2', 'G5'},
        excluded_geos=set(),
    )

    treatment = analysis._get_treatment_mask(data, design_obj)

    # Sorted geos: G1, G2, G3, G4, G5.
    # Treatment cell_1: G1, G3 (expected mask value = 1.0).
    # Treatment cell_2: G4 (expected mask value = 2.0).
    # Control: G2, G5 (expected mask value = 0.0).
    # Expected mask: [1.0, 0.0, 1.0, 2.0, 0.0]
    np.testing.assert_array_equal(
        treatment.mask, jnp.array([1.0, 0.0, 1.0, 2.0, 0.0])
    )
    self.assertEqual(treatment.geos, ['G1', 'G3', 'G4'])

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
        experiment_types=api.ExperimentType.HOLDBACK,
        geo_assignment_rule=api.GeoAssignmentRule.RANDOM,
        seed=42,
        n_candidates=5,
        n_aa_test_iterations=10,
    )

    study_design = api.Design(
        designs={
            'cell_1': api.PerCellDesign(
                treatment_geos={'G1', 'G2'},
                minimum_detectable_effect=0.1,
                design_implied_cpic=0.0,
                p_value=0.5,
                budget=1000.0,
            )
        },
        control_geos={f'G{i}' for i in range(3, 7)},
        excluded_geos=set(),
        design_config=design_config,
        constraints=api.Constraints(max_conversions_percent=0.5),
        data=data,
    )

    config = api.AnalysisConfig(
        design=study_design,
        analysis_start_date=pd.Timestamp('2024-01-11'),
        analysis_end_date=pd.Timestamp('2024-01-15'),
        alpha=0.1,
        test_type=api.TestType.TWO_SIDED,
    )

    result = analysis.analyze(data, config)

    self.assertIsInstance(result, api.AnalysisResult)
    self.assertIs(result.analysis_config, config)
    self.assertEqual(
        result.analysis_config.analysis_start_date, pd.Timestamp('2024-01-11')
    )
    self.assertEqual(
        result.analysis_config.analysis_end_date, pd.Timestamp('2024-01-15')
    )
    self.assertEqual(result.analysis_config.alpha, 0.1)
    self.assertEqual(result.analysis_config.test_type, api.TestType.TWO_SIDED)

    self.assertIn('cell_1', result.results)

    metrics = result.results['cell_1']

    # Test reproducibility
    result2 = analysis.analyze(data, config)
    metrics2 = result2.results['cell_1']

    self.assertEqual(metrics.lift.point_estimate, metrics2.lift.point_estimate)
    self.assertEqual(metrics.lift.lower_bound, metrics2.lift.lower_bound)
    self.assertEqual(metrics.lift.upper_bound, metrics2.lift.upper_bound)
    self.assertEqual(
        metrics.lift.standard_deviation, metrics2.lift.standard_deviation
    )
    self.assertEqual(
        metrics.percent_lift.point_estimate,
        metrics2.percent_lift.point_estimate,
    )
    self.assertEqual(
        metrics.percent_lift.lower_bound, metrics2.percent_lift.lower_bound
    )
    self.assertEqual(
        metrics.percent_lift.upper_bound, metrics2.percent_lift.upper_bound
    )
    self.assertEqual(
        metrics.percent_lift.standard_deviation,
        metrics2.percent_lift.standard_deviation,
    )

    self.assertIsInstance(metrics, api.AnalysisMetrics)
    self.assertAlmostEqual(metrics.lift.point_estimate, 0.0, places=1)
    self.assertAlmostEqual(metrics.percent_lift.point_estimate, 0.0, places=1)
    self.assertEqual(
        list(metrics.cumulative_lift.columns),
        ['lift', 'lower_bound', 'upper_bound'],
    )
    self.assertLen(metrics.cumulative_lift, 5)
    self.assertAlmostEqual(
        metrics.cumulative_lift['lift'].iloc[-1],
        metrics.lift.point_estimate,
        places=5,
    )

    self.assertEqual(
        list(metrics.counterfactual_conversions.columns),
        ['observed', 'counterfactual', 'lower_bound', 'upper_bound'],
    )
    self.assertLen(metrics.counterfactual_conversions, 15)

    self.assertEqual(
        list(metrics.pointwise_difference.columns),
        ['difference', 'lower_bound', 'upper_bound'],
    )
    self.assertLen(metrics.pointwise_difference, 15)

  def test_analyze_stratified_sampling(self):
    data = self._create_sample_data(n_days=15, n_geos=10)

    design_config = api.DesignConfig(
        experiment_duration=3,
        experiment_types=api.ExperimentType.HOLDBACK,
        geo_assignment_rule=api.GeoAssignmentRule.STRATIFIED_SAMPLING,
        seed=42,
        n_candidates=5,
        n_aa_test_iterations=10,
    )

    study_design = api.Design(
        designs={
            'cell_1': api.PerCellDesign(
                treatment_geos={'G1'},
                minimum_detectable_effect=0.1,
                design_implied_cpic=0.0,
                p_value=0.5,
                budget=1000.0,
            )
        },
        control_geos={f'G{i}' for i in range(2, 11)},
        excluded_geos=set(),
        design_config=design_config,
        constraints=api.Constraints(max_conversions_percent=0.5),
        geo_stratum_labels=jnp.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1]),
        data=data,
    )

    config = api.AnalysisConfig(
        design=study_design,
        analysis_start_date=pd.Timestamp('2024-01-11'),
        analysis_end_date=pd.Timestamp('2024-01-15'),
        alpha=0.1,
        test_type=api.TestType.TWO_SIDED,
    )

    result = analysis.analyze(data, config)

    self.assertIsInstance(result, api.AnalysisResult)
    self.assertIs(result.analysis_config, config)
    self.assertIn('cell_1', result.results)
    metrics = result.results['cell_1']

    # Test reproducibility
    result2 = analysis.analyze(data, config)
    metrics2 = result2.results['cell_1']

    self.assertEqual(metrics.lift.point_estimate, metrics2.lift.point_estimate)
    self.assertEqual(metrics.lift.lower_bound, metrics2.lift.lower_bound)
    self.assertEqual(metrics.lift.upper_bound, metrics2.lift.upper_bound)
    self.assertEqual(
        metrics.lift.standard_deviation, metrics2.lift.standard_deviation
    )
    self.assertEqual(
        metrics.percent_lift.point_estimate,
        metrics2.percent_lift.point_estimate,
    )
    self.assertEqual(
        metrics.percent_lift.lower_bound, metrics2.percent_lift.lower_bound
    )
    self.assertEqual(
        metrics.percent_lift.upper_bound, metrics2.percent_lift.upper_bound
    )
    self.assertEqual(
        metrics.percent_lift.standard_deviation,
        metrics2.percent_lift.standard_deviation,
    )

    self.assertAlmostEqual(metrics.lift.point_estimate, 0.0, places=1)
    self.assertAlmostEqual(metrics.percent_lift.point_estimate, 0.0, places=1)
    self.assertEqual(
        list(metrics.cumulative_lift.columns),
        ['lift', 'lower_bound', 'upper_bound'],
    )

    self.assertEqual(
        list(metrics.counterfactual_conversions.columns),
        ['observed', 'counterfactual', 'lower_bound', 'upper_bound'],
    )
    self.assertLen(metrics.counterfactual_conversions, 15)

    self.assertEqual(
        list(metrics.pointwise_difference.columns),
        ['difference', 'lower_bound', 'upper_bound'],
    )
    self.assertLen(metrics.pointwise_difference, 15)

  def test_analyze_unsupported_methodology(self):
    # SDID is not supported yet.
    design_config = api.DesignConfig(
        experiment_duration=2,
        experiment_types=api.ExperimentType.HOLDBACK,
        methodology=api.Methodology.SDID,
    )
    study_design = api.Design(
        designs={
            'cell_1': api.PerCellDesign(
                treatment_geos={'G1'},
                minimum_detectable_effect=0.1,
                design_implied_cpic=0.0,
                p_value=0.5,
                budget=1000.0,
            )
        },
        control_geos={'G2'},
        excluded_geos=set(),
        design_config=design_config,
        constraints=api.Constraints(),
    )
    config = api.AnalysisConfig(
        design=study_design,
        analysis_start_date=pd.Timestamp('2024-01-06'),
        analysis_end_date=pd.Timestamp('2024-01-10'),
        alpha=0.1,
        test_type=api.TestType.TWO_SIDED,
    )

    data = self._create_sample_data(n_days=10, n_geos=2)
    with self.assertRaisesRegex(ValueError, 'Unsupported methodology'):
      analysis.analyze(data, config)

  def test_analyze_excluded_geos(self):
    # Setup with more geos to avoid "Not enough geos" error in placebo
    # detection.
    data = self._create_sample_data(n_days=15, n_geos=10)

    design_config = api.DesignConfig(
        experiment_duration=2,
        experiment_types=api.ExperimentType.HOLDBACK,
        geo_assignment_rule=api.GeoAssignmentRule.RANDOM,
        seed=42,
        n_candidates=3,
        n_aa_test_iterations=5,
    )

    # G10 is excluded
    study_design = api.Design(
        designs={
            'cell_1': api.PerCellDesign(
                treatment_geos={'G1'},
                minimum_detectable_effect=0.1,
                design_implied_cpic=0.0,
                p_value=0.5,
                budget=1000.0,
            )
        },
        control_geos={f'G{i}' for i in range(2, 10)},
        excluded_geos={'G10'},
        design_config=design_config,
        constraints=api.Constraints(
            excluded_geos={'G10'}, max_conversions_percent=0.5
        ),
        data=data,
    )

    config = api.AnalysisConfig(
        design=study_design,
        analysis_start_date=pd.Timestamp('2024-01-11'),
        analysis_end_date=pd.Timestamp('2024-01-15'),
        alpha=0.1,
        test_type=api.TestType.TWO_SIDED,
    )

    result = analysis.analyze(data, config)

    self.assertIsInstance(result, api.AnalysisResult)
    self.assertIs(result.analysis_config, config)
    self.assertIn('cell_1', result.results)
    metrics = result.results['cell_1']

    self.assertAlmostEqual(metrics.lift.point_estimate, 0.0, places=1)
    self.assertAlmostEqual(metrics.percent_lift.point_estimate, 0.0, places=1)

  def test_analyze_with_gap(self):
    data = self._create_sample_data(n_days=15, n_geos=10)

    design_config = api.DesignConfig(
        experiment_duration=2,
        experiment_types=api.ExperimentType.HOLDBACK,
        geo_assignment_rule=api.GeoAssignmentRule.RANDOM,
        seed=42,
        n_candidates=3,
        n_aa_test_iterations=5,
    )

    study_design = api.Design(
        designs={
            'cell_1': api.PerCellDesign(
                treatment_geos={'G1'},
                minimum_detectable_effect=0.1,
                design_implied_cpic=0.0,
                p_value=0.5,
                budget=1000.0,
            )
        },
        control_geos={f'G{i}' for i in range(2, 10)},
        excluded_geos={'G10'},
        design_config=design_config,
        constraints=api.Constraints(
            excluded_geos={'G10'}, max_conversions_percent=0.5
        ),
        data=data,
    )

    config = api.AnalysisConfig(
        design=study_design,
        analysis_start_date=pd.Timestamp('2024-01-11'),
        analysis_end_date=pd.Timestamp('2024-01-15'),
        pretest_end_date=pd.Timestamp('2024-01-08'),
        alpha=0.1,
        test_type=api.TestType.TWO_SIDED,
    )

    result = analysis.analyze(data, config)

    self.assertIsInstance(result, api.AnalysisResult)
    self.assertIs(result.analysis_config, config)
    self.assertIn('cell_1', result.results)
    metrics = result.results['cell_1']
    self.assertAlmostEqual(metrics.lift.point_estimate, 0.0, places=1)
    self.assertNotIn(
        pd.Timestamp('2024-01-09'), metrics.counterfactual_conversions.index
    )
    self.assertNotIn(
        pd.Timestamp('2024-01-10'), metrics.counterfactual_conversions.index
    )

  def test_analyze_locations_mismatch(self):
    data = self._create_sample_data(n_days=5, n_geos=3)
    # Original data locations: G1, G2, G3.

    # Data for analysis: remove G3, add G4.
    analysis_data = data[data[api.LOCATION] != 'G3'].copy()
    new_rows = []
    for d in data[api.DATE].unique():
      new_rows.append({api.DATE: d, api.LOCATION: 'G4', api.CONVERSIONS: 100.0})
    analysis_data = pd.concat(
        [analysis_data, pd.DataFrame(new_rows)], ignore_index=True
    )

    study_design = api.Design(
        designs={
            'cell_1': api.PerCellDesign(
                treatment_geos={'G1'},
                minimum_detectable_effect=0.1,
                design_implied_cpic=0.0,
                p_value=0.5,
                budget=1000.0,
            )
        },
        control_geos={'G2'},
        excluded_geos={'G3'},
        data=data,
        design_config=api.DesignConfig(
            experiment_duration=1, experiment_types=api.ExperimentType.HOLDBACK
        ),
    )

    config = api.AnalysisConfig(
        design=study_design,
        analysis_start_date=pd.Timestamp('2024-01-01'),
        analysis_end_date=pd.Timestamp('2024-01-05'),
    )

    with self.assertRaisesRegex(
        ValueError, 'locations in the analysis data do not match'
    ):
      analysis.analyze(analysis_data, config)

  def test_plot_analysis_with_gap(self):
    data = self._create_sample_data(n_days=15, n_geos=10)

    design_config = api.DesignConfig(
        experiment_duration=2,
        experiment_types=api.ExperimentType.HOLDBACK,
        geo_assignment_rule=api.GeoAssignmentRule.RANDOM,
        seed=42,
        n_candidates=3,
        n_aa_test_iterations=5,
    )

    study_design = api.Design(
        designs={
            'cell_1': api.PerCellDesign(
                treatment_geos={'G1'},
                minimum_detectable_effect=0.1,
                design_implied_cpic=0.0,
                p_value=0.5,
                budget=1000.0,
            )
        },
        control_geos={f'G{i}' for i in range(2, 10)},
        excluded_geos={'G10'},
        design_config=design_config,
        constraints=api.Constraints(
            excluded_geos={'G10'}, max_conversions_percent=0.5
        ),
        data=data,
    )

    config = api.AnalysisConfig(
        design=study_design,
        analysis_start_date=pd.Timestamp('2024-01-11'),
        analysis_end_date=pd.Timestamp('2024-01-15'),
        pretest_end_date=pd.Timestamp('2024-01-08'),
        alpha=0.1,
        test_type=api.TestType.TWO_SIDED,
    )

    result = analysis.analyze(data, config)

    with absltest.mock.patch.object(analysis.plt, 'show') as mock_show:
      analysis.plot_analysis(result)
      mock_show.assert_called_once()

  def _create_multicell_sample_data(
      self, n_days=15, n_geos=10, cell_names=None
  ) -> pd.DataFrame:
    """Helper to create sample data with multicell spend."""
    if cell_names is None:
      cell_names = ['cell_1', 'cell_2']
    dates = pd.date_range('2024-01-01', periods=n_days)
    geos = [f'G{i}' for i in range(1, n_geos + 1)]
    data_rows = []
    for d in dates:
      for i, g in enumerate(geos):
        val = 100.0 + i * 10.0 + (d.day % 7) * 5.0
        row = {api.DATE: d, api.LOCATION: g, api.CONVERSIONS: val}
        for cell in cell_names:
          row[f'spend_{cell}'] = val * 0.5
        data_rows.append(row)
    return pd.DataFrame(data_rows)

  def test_analyze_multicell(self):
    data = self._create_multicell_sample_data(n_days=15, n_geos=10)
    cell_names = ['cell_1', 'cell_2']

    design_config = api.DesignConfig(
        experiment_duration=3,
        experiment_types={
            'cell_1': api.ExperimentType.HOLDBACK,
            'cell_2': api.ExperimentType.HOLDBACK,
        },
        geo_assignment_rule=api.GeoAssignmentRule.RANDOM,
        seed=42,
        cell_count=2,
        n_candidates=5,
        n_aa_test_iterations=10,
    )

    study_design = api.Design(
        designs={
            'cell_1': api.PerCellDesign(
                treatment_geos={'G1', 'G2'},
                minimum_detectable_effect=0.1,
                design_implied_cpic=0.0,
                p_value=0.5,
                budget=1000.0,
            ),
            'cell_2': api.PerCellDesign(
                treatment_geos={'G3', 'G4'},
                minimum_detectable_effect=0.1,
                design_implied_cpic=0.0,
                p_value=0.5,
                budget=1000.0,
            ),
        },
        control_geos={f'G{i}' for i in range(5, 11)},
        excluded_geos=set(),
        design_config=design_config,
        constraints=api.Constraints(max_conversions_percent=0.5),
        data=data,
    )

    config = api.AnalysisConfig(
        design=study_design,
        analysis_start_date=pd.Timestamp('2024-01-11'),
        analysis_end_date=pd.Timestamp('2024-01-15'),
        alpha=0.1,
        test_type=api.TestType.TWO_SIDED,
    )

    result = analysis.analyze(data, config)

    self.assertIsInstance(result, api.AnalysisResult)
    self.assertIs(result.analysis_config, config)
    self.assertEqual(sorted(list(result.results.keys())), ['cell_1', 'cell_2'])

    for cell in cell_names:
      metrics = result.results[cell]
      self.assertIsInstance(metrics, api.AnalysisMetrics)
      self.assertLen(metrics.counterfactual_conversions, 15)
      self.assertLen(metrics.pointwise_difference, 15)
      self.assertLen(metrics.cumulative_lift, 5)
      self.assertIsNotNone(metrics.icpd)


if __name__ == '__main__':
  absltest.main()
