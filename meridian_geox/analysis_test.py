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

import datetime
import typing
from absl.testing import absltest
from absl.testing import parameterized
import jax
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
    # G3 is excluded, and 2024-01-04 (by design) and 2024-01-05 (by config) are
    # excluded.
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
        excluded_dates={pd.Timestamp('2024-01-04')},
    )
    config = api.AnalysisConfig(
        n_placebo_candidates=10,
        design=design_obj,
        analysis_start_date=pd.Timestamp('2024-01-01'),
        analysis_end_date=pd.Timestamp('2024-01-05'),
        excluded_dates={pd.Timestamp('2024-01-05')},
    )

    filtered_data = analysis._prepare_data(data, config)

    self.assertNotIn('G3', filtered_data[api.LOCATION].unique())
    self.assertIn(pd.Timestamp('2024-01-04'), filtered_data[api.DATE].unique())
    self.assertNotIn(
        pd.Timestamp('2024-01-05'), filtered_data[api.DATE].unique()
    )
    self.assertLen(filtered_data[api.LOCATION].unique(), 2)
    self.assertLen(filtered_data[api.DATE].unique(), 4)

  def test_get_time_series(self):
    data = self._create_sample_data(n_days=10, n_geos=2)
    config = api.AnalysisConfig(
        n_placebo_candidates=10,
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
        n_placebo_candidates=10,
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
        n_candidates=500,
        experiment_duration=datetime.timedelta(days=2),
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
        n_placebo_candidates=10,
        design=design_obj,
        analysis_start_date=pd.Timestamp('2024-01-01'),
        analysis_end_date=pd.Timestamp('2024-01-05'),
    )

    prepared_config = analysis._prepare_design_config(config)

    self.assertEqual(prepared_config.n_candidates, 10)
    self.assertEqual(config.alpha, 0.05)
    self.assertEqual(config.test_type, api.TestType.TWO_SIDED)

  def test_get_experiment_types(self):
    # Case 1: Single enum
    config = api.DesignConfig(
        n_candidates=500,
        experiment_duration=datetime.timedelta(days=1),
        experiment_types=api.ExperimentType.GO_DARK,
    )
    self.assertEqual(
        analysis._get_experiment_types(config),
        {api.CELL_1: api.ExperimentType.GO_DARK},
    )

    # Case 2: Singleton map
    config = api.DesignConfig(
        n_candidates=500,
        experiment_duration=datetime.timedelta(days=1),
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

  def test_get_effective_constraints(self):
    constraints = api.Constraints(
        excluded_geos={'G1'},
        excluded_dates={pd.Timestamp('2024-01-01')},
    )
    design_obj = api.Design(
        designs={},
        control_geos={'G2'},
        excluded_geos={'G1', 'G_OUTLIER'},
        excluded_dates={pd.Timestamp('2024-01-01'), pd.Timestamp('2024-01-02')},
        constraints=constraints,
    )
    effective_constraints = analysis._get_effective_constraints(design_obj)

    # Validates that effective constraints are properly updated
    self.assertIn('G_OUTLIER', effective_constraints.excluded_geos)
    self.assertIn('G1', effective_constraints.excluded_geos)
    self.assertIn(
        pd.Timestamp('2024-01-02'), effective_constraints.excluded_dates
    )
    self.assertIn(
        pd.Timestamp('2024-01-01'), effective_constraints.excluded_dates
    )

    # Validates that we didn't mutate the original constraints
    self.assertNotIn('G_OUTLIER', constraints.excluded_geos)
    self.assertNotIn(pd.Timestamp('2024-01-02'), constraints.excluded_dates)

  def test_get_effective_constraints_none_constraints(self):
    design_obj = api.Design(
        designs={},
        control_geos={'G2'},
        excluded_geos={'G_OUTLIER'},
        excluded_dates={pd.Timestamp('2024-01-02')},
        constraints=None,
    )
    effective_constraints = analysis._get_effective_constraints(design_obj)

    # Validates that it initializes a new Constraints object
    self.assertIn('G_OUTLIER', effective_constraints.excluded_geos)
    self.assertIn(
        pd.Timestamp('2024-01-02'), effective_constraints.excluded_dates
    )

  def test_analyze_random_assignment(self):
    data = self._create_sample_data(n_days=15, n_geos=6)

    design_config = api.DesignConfig(
        n_candidates=500,
        experiment_duration=datetime.timedelta(days=3),
        experiment_types=api.ExperimentType.HOLDBACK,
        geo_assignment_rule=api.GeoAssignmentRule.RANDOM,
        seed=42,
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
        n_placebo_candidates=10,
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
        n_candidates=500,
        experiment_duration=datetime.timedelta(days=3),
        experiment_types=api.ExperimentType.HOLDBACK,
        geo_assignment_rule=api.GeoAssignmentRule.STRATIFIED_SAMPLING,
        seed=42,
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
        n_placebo_candidates=10,
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

  @absltest.mock.patch(
      'meridian_geox.data_quality.data_quality.check_analysis_data_quality'
  )
  def test_analyze_excludes_outlier_dates_from_data_quality_check(
      self, mock_check_analysis_data_quality
  ):
    data = self._create_sample_data(n_days=15, n_geos=10)
    design_config = api.DesignConfig(
        experiment_duration=datetime.timedelta(days=3),
        experiment_types=api.ExperimentType.HOLDBACK,
        geo_assignment_rule=api.GeoAssignmentRule.RANDOM,
        seed=42,
        design_output_count=1,
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
        control_geos={f'G{i}' for i in range(3, 11)},
        excluded_geos=set(),
        excluded_dates=set(),
        design_config=design_config,
        constraints=api.Constraints(max_conversions_percent=0.5),
        data=data,
    )
    analysis_config = api.AnalysisConfig(
        design=study_design,
        analysis_start_date=pd.Timestamp('2024-01-07'),
        analysis_end_date=pd.Timestamp('2024-01-15'),
    )

    # Mock data quality check returning outlier dates.
    mock_check_analysis_data_quality.return_value = api.QualityCheckResult(
        quality_check_config=api.QualityCheckConfig(exclude_outlier_dates=True),
        quality_metrics=pd.DataFrame(),
        outlier_dates={pd.Timestamp('2024-01-08')},
    )

    result = analysis.analyze(
        data,
        analysis_config,
        data_quality_check_config=api.QualityCheckConfig(
            exclude_outlier_dates=True
        ),
    )

    self.assertIn(pd.Timestamp('2024-01-08'), result.excluded_dates)
    self.assertEqual(result.excluded_geos, set())

  def test_analyze_unsupported_methodology(self):
    # SDID is not supported yet.
    design_config = api.DesignConfig(
        n_candidates=500,
        experiment_duration=datetime.timedelta(days=2),
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
        control_geos={'G2', 'G3', 'G4'},
        excluded_geos=set(),
        design_config=design_config,
        constraints=api.Constraints(),
    )
    config = api.AnalysisConfig(
        n_placebo_candidates=10,
        design=study_design,
        analysis_start_date=pd.Timestamp('2024-01-06'),
        analysis_end_date=pd.Timestamp('2024-01-10'),
        alpha=0.1,
        test_type=api.TestType.TWO_SIDED,
    )

    data = self._create_sample_data(n_days=10, n_geos=4)
    with self.assertRaisesRegex(ValueError, 'Unsupported methodology'):
      analysis.analyze(data, config)

  def test_analyze_excluded_geos(self):
    # Setup with more geos to avoid "Not enough geos" error in placebo
    # detection.
    data = self._create_sample_data(n_days=15, n_geos=10)

    design_config = api.DesignConfig(
        n_candidates=500,
        experiment_duration=datetime.timedelta(days=2),
        experiment_types=api.ExperimentType.HOLDBACK,
        geo_assignment_rule=api.GeoAssignmentRule.RANDOM,
        seed=42,
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
        n_placebo_candidates=10,
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
        n_candidates=500,
        experiment_duration=datetime.timedelta(days=2),
        experiment_types=api.ExperimentType.HOLDBACK,
        geo_assignment_rule=api.GeoAssignmentRule.RANDOM,
        seed=42,
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
        n_placebo_candidates=10,
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
            n_candidates=500,
            experiment_duration=datetime.timedelta(days=1),
            experiment_types=api.ExperimentType.HOLDBACK,
        ),
    )

    config = api.AnalysisConfig(
        n_placebo_candidates=10,
        design=study_design,
        analysis_start_date=pd.Timestamp('2024-01-01'),
        analysis_end_date=pd.Timestamp('2024-01-05'),
    )

    with self.assertRaisesRegex(
        ValueError, 'locations in the analysis data do not match'
    ):
      analysis.analyze(analysis_data, config)  # pyrefly: ignore[bad-argument-type]

  def test_plot_analysis_with_gap(self):
    data = self._create_sample_data(n_days=15, n_geos=10)

    design_config = api.DesignConfig(
        n_candidates=500,
        experiment_duration=datetime.timedelta(days=2),
        experiment_types=api.ExperimentType.HOLDBACK,
        geo_assignment_rule=api.GeoAssignmentRule.RANDOM,
        seed=42,
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
        n_placebo_candidates=10,
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
    data = self._create_multicell_sample_data(n_days=15, n_geos=14)
    cell_names = ['cell_1', 'cell_2']

    design_config = api.DesignConfig(
        experiment_duration=datetime.timedelta(days=3),
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
        control_geos={f'G{i}' for i in range(5, 15)},
        excluded_geos=set(),
        design_config=design_config,
        constraints=api.Constraints(max_conversions_percent=0.5),
        data=data,
    )

    config = api.AnalysisConfig(
        n_placebo_candidates=10,
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

  def test_analyze_estimated_bau_spend_holdback(self):
    data = self._create_sample_data(n_days=15, n_geos=10, include_spend=True)

    design_config = api.DesignConfig(
        experiment_duration=datetime.timedelta(days=5),
        experiment_types=api.ExperimentType.HOLDBACK,
        geo_assignment_rule=api.GeoAssignmentRule.RANDOM,
        seed=42,
        n_candidates=3,
        n_aa_test_iterations=5,
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
        control_geos={f'G{i}' for i in range(3, 11)},
        excluded_geos=set(),
        design_config=design_config,
        constraints=api.Constraints(),
        data=data,
    )

    config = api.AnalysisConfig(
        n_placebo_candidates=10,
        design=study_design,
        analysis_start_date=pd.Timestamp('2024-01-11'),
        analysis_end_date=pd.Timestamp('2024-01-15'),
        alpha=0.1,
        test_type=api.TestType.TWO_SIDED,
    )

    result = analysis.analyze(data, config)
    metrics = result.results['cell_1']
    self.assertIsNotNone(metrics.descriptive_metrics)
    assert metrics.descriptive_metrics is not None

    pretest_data = data[data[api.DATE] < pd.Timestamp('2024-01-11')]
    test_data = data[
        (data[api.DATE] >= pd.Timestamp('2024-01-11'))
        & (data[api.DATE] <= pd.Timestamp('2024-01-15'))
    ]

    treatment_geos = {'G1', 'G2'}
    treatment_pretest_conv = pretest_data[
        pretest_data[api.LOCATION].isin(treatment_geos)
    ][api.CONVERSIONS].sum()

    control_pretest_conv = pretest_data[
        ~pretest_data[api.LOCATION].isin(treatment_geos)
    ][api.CONVERSIONS].sum()

    treatment_test_spend = test_data[
        test_data[api.LOCATION].isin(treatment_geos)
    ][api.SPEND].sum()

    control_predicted_spend = (
        control_pretest_conv / treatment_pretest_conv
    ) * treatment_test_spend

    expected_bau_spend = treatment_test_spend + control_predicted_spend
    self.assertAlmostEqual(
        metrics.descriptive_metrics.estimated_bau_spend,
        expected_bau_spend,
        places=2,
    )

  @parameterized.named_parameters(
      ('go_dark', api.ExperimentType.GO_DARK),
      ('heavy_up', api.ExperimentType.HEAVY_UP),
  )
  def test_analyze_estimated_bau_spend_go_dark_or_heavy_up(
      self, experiment_type
  ):
    data = self._create_sample_data(n_days=15, n_geos=10, include_spend=True)

    design_config = api.DesignConfig(
        experiment_duration=datetime.timedelta(days=5),
        experiment_types=experiment_type,
        geo_assignment_rule=api.GeoAssignmentRule.RANDOM,
        seed=42,
        n_candidates=3,
        n_aa_test_iterations=5,
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
        control_geos={f'G{i}' for i in range(3, 11)},
        excluded_geos=set(),
        design_config=design_config,
        constraints=api.Constraints(),
        data=data,
    )

    config = api.AnalysisConfig(
        n_placebo_candidates=10,
        design=study_design,
        analysis_start_date=pd.Timestamp('2024-01-11'),
        analysis_end_date=pd.Timestamp('2024-01-15'),
        alpha=0.1,
        test_type=api.TestType.TWO_SIDED,
    )

    result = analysis.analyze(data, config)
    metrics = result.results['cell_1']
    self.assertIsNotNone(metrics.descriptive_metrics)
    assert metrics.descriptive_metrics is not None

    pretest_data = data[data[api.DATE] < pd.Timestamp('2024-01-11')]
    test_data = data[
        (data[api.DATE] >= pd.Timestamp('2024-01-11'))
        & (data[api.DATE] <= pd.Timestamp('2024-01-15'))
    ]

    treatment_geos = {'G1', 'G2'}
    pretest_by_geo = pretest_data.pivot_table(
        index=api.DATE, columns=api.LOCATION, values=api.SPEND, aggfunc='sum'
    )
    test_by_geo = test_data.pivot_table(
        index=api.DATE, columns=api.LOCATION, values=api.SPEND, aggfunc='sum'
    )

    y_spend_pre = pretest_by_geo[list(treatment_geos)].mean(axis=1)
    x_spend_pre = pretest_by_geo[
        [c for c in pretest_by_geo.columns if c not in treatment_geos]
    ].mean(axis=1)

    x_mean = x_spend_pre.mean()
    y_mean = y_spend_pre.mean()
    ss_xx = ((x_spend_pre - x_mean) ** 2).sum()
    ss_xy = ((x_spend_pre - x_mean) * (y_spend_pre - y_mean)).sum()
    slope = ss_xy / ss_xx if ss_xx > 1e-10 else 0.0
    intercept = y_mean - slope * x_mean

    x_spend_test = test_by_geo[
        [c for c in test_by_geo.columns if c not in treatment_geos]
    ].mean(axis=1)
    y_spend_pred = intercept + slope * x_spend_test
    treatment_predicted_spend = y_spend_pred.sum() * len(treatment_geos)

    control_test_spend = test_by_geo[
        [c for c in test_by_geo.columns if c not in treatment_geos]
    ].values.sum()

    expected_bau_spend = control_test_spend + treatment_predicted_spend

    self.assertAlmostEqual(
        metrics.descriptive_metrics.estimated_bau_spend,
        expected_bau_spend,
        places=2,
    )

  def test_analyze_estimated_bau_spend_multicell(self):
    data = self._create_sample_data(n_days=15, n_geos=50, include_spend=True)

    design_config = api.DesignConfig(
        experiment_duration=datetime.timedelta(days=5),
        experiment_types=api.ExperimentType.HOLDBACK,
        geo_assignment_rule=api.GeoAssignmentRule.RANDOM,
        seed=42,
        n_candidates=3,
        n_aa_test_iterations=5,
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
        control_geos={f'G{i}' for i in range(5, 51)},
        excluded_geos=set(),
        design_config=design_config,
        constraints=api.Constraints(),
        data=data,
    )

    config = api.AnalysisConfig(
        n_placebo_candidates=10,
        design=study_design,
        analysis_start_date=pd.Timestamp('2024-01-11'),
        analysis_end_date=pd.Timestamp('2024-01-15'),
        alpha=0.1,
        test_type=api.TestType.TWO_SIDED,
    )

    result = analysis.analyze(data, config)
    metrics_cell1 = result.results['cell_1']
    assert metrics_cell1.descriptive_metrics is not None

    pretest_data = data[data[api.DATE] < pd.Timestamp('2024-01-11')]
    test_data = data[
        (data[api.DATE] >= pd.Timestamp('2024-01-11'))
        & (data[api.DATE] <= pd.Timestamp('2024-01-15'))
    ]

    treatment_geos = {'G1', 'G2'}
    control_geos = {f'G{i}' for i in range(5, 51)}

    treatment_pretest_conv = pretest_data[
        pretest_data[api.LOCATION].isin(treatment_geos)
    ][api.CONVERSIONS].sum()

    control_pretest_conv = pretest_data[
        pretest_data[api.LOCATION].isin(control_geos)
    ][api.CONVERSIONS].sum()

    treatment_test_spend = test_data[
        test_data[api.LOCATION].isin(treatment_geos)
    ][api.SPEND].sum()

    control_predicted_spend = (
        control_pretest_conv / treatment_pretest_conv
    ) * treatment_test_spend

    expected_bau_spend_cell1 = treatment_test_spend + control_predicted_spend
    self.assertAlmostEqual(
        metrics_cell1.descriptive_metrics.estimated_bau_spend,
        expected_bau_spend_cell1,
        places=2,
    )

  def test_analyze_estimated_bau_spend_unsupported_type(self):
    data = self._create_sample_data(n_days=15, n_geos=10, include_spend=True)

    design_config = api.DesignConfig(
        experiment_duration=datetime.timedelta(days=5),
        experiment_types=api.ExperimentType.HOLDBACK,
        geo_assignment_rule=api.GeoAssignmentRule.RANDOM,
        seed=42,
        n_candidates=3,
        n_aa_test_iterations=5,
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
        control_geos={f'G{i}' for i in range(3, 11)},
        excluded_geos=set(),
        design_config=design_config,
        constraints=api.Constraints(),
        data=data,
    )

    config = api.AnalysisConfig(
        n_placebo_candidates=10,
        design=study_design,
        analysis_start_date=pd.Timestamp('2024-01-11'),
        analysis_end_date=pd.Timestamp('2024-01-15'),
        alpha=0.1,
        test_type=api.TestType.TWO_SIDED,
    )

    with absltest.mock.patch.object(
        analysis,
        '_get_experiment_types',
        return_value={'cell_1': typing.cast(api.ExperimentType, None)},
    ):
      result = analysis.analyze(data, config)

    metrics = result.results['cell_1']
    self.assertIsNotNone(metrics.descriptive_metrics)
    assert metrics.descriptive_metrics is not None
    self.assertIsNone(metrics.descriptive_metrics.estimated_bau_spend)

  @parameterized.named_parameters(
      (
          'zero_conversions',
          [[0.0, 0.0], [0.0, 0.0]],
      ),
      (
          'zero_treatment_conversions',
          [[0.0, 10.0]],
      ),
  )
  def test_get_estimated_bau_spend_holdback_nil_returns(
      self, pretest_conversions
  ):
    treatment_mask = np.array([True, False])
    test_spend = jnp.array([[10.0, 20.0]])

    result = analysis._get_estimated_bau_spend_holdback(
        treatment_mask=treatment_mask,
        control_mask=~treatment_mask,
        pretest_conversions=jnp.array(pretest_conversions),
        test_spend=test_spend,
    )
    self.assertIsNone(result)

  def test_get_estimated_bau_spend_parent_missing_cell(self):
    design_config = api.DesignConfig(
        experiment_duration=datetime.timedelta(days=5),
        experiment_types=api.ExperimentType.HOLDBACK,
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
    )
    config = api.AnalysisConfig(
        n_placebo_candidates=10,
        design=study_design,
        analysis_start_date=pd.Timestamp('2024-01-11'),
        analysis_end_date=pd.Timestamp('2024-01-15'),
    )
    result = analysis._get_estimated_bau_spend(
        analysis_config=config,
        cell='cell_1',
        geos=['G1', 'G2'],
        spend={},
        conversions=analysis.TimeSeries(
            pretest=jnp.array([[10.0, 10.0]]),
            test=jnp.array([[10.0, 10.0]]),
            pretest_dates=pd.date_range('2024-01-01', periods=1),
            test_dates=pd.date_range('2024-01-11', periods=1),
        ),
        experiment_type=api.ExperimentType.HOLDBACK,
    )
    self.assertIsNone(result)

  def test_get_estimated_bau_spend_parent_missing_design(self):
    design_config = api.DesignConfig(
        experiment_duration=datetime.timedelta(days=5),
        experiment_types=api.ExperimentType.HOLDBACK,
    )
    study_design = api.Design(
        designs={},
        control_geos=set(),
        excluded_geos=set(),
        design_config=design_config,
    )
    config = api.AnalysisConfig(
        n_placebo_candidates=10,
        design=study_design,
        analysis_start_date=pd.Timestamp('2024-01-11'),
        analysis_end_date=pd.Timestamp('2024-01-15'),
    )

    cell_spend = analysis.TimeSeries(
        pretest=jnp.array([[10.0, 10.0]]),
        test=jnp.array([[10.0, 10.0]]),
        pretest_dates=pd.date_range('2024-01-01', periods=1),
        test_dates=pd.date_range('2024-01-11', periods=1),
    )

    result = analysis._get_estimated_bau_spend(
        analysis_config=config,
        cell='cell_1',
        geos=['G1', 'G2'],
        spend={'cell_1': cell_spend},
        conversions=cell_spend,
        experiment_type=api.ExperimentType.HOLDBACK,
    )
    self.assertIsNone(result)

  def test_get_estimated_bau_spend_parent_empty_treatment_geos(self):
    design_config = api.DesignConfig(
        experiment_duration=datetime.timedelta(days=5),
        experiment_types=api.ExperimentType.HOLDBACK,
    )
    study_design = api.Design(
        designs={
            'cell_1': api.PerCellDesign(
                treatment_geos=set(),
                minimum_detectable_effect=0.1,
                design_implied_cpic=0.0,
                p_value=0.5,
                budget=1000.0,
            )
        },
        control_geos={'G2'},
        excluded_geos=set(),
        design_config=design_config,
    )
    config = api.AnalysisConfig(
        n_placebo_candidates=10,
        design=study_design,
        analysis_start_date=pd.Timestamp('2024-01-11'),
        analysis_end_date=pd.Timestamp('2024-01-15'),
    )

    cell_spend = analysis.TimeSeries(
        pretest=jnp.array([[10.0, 10.0]]),
        test=jnp.array([[10.0, 10.0]]),
        pretest_dates=pd.date_range('2024-01-01', periods=1),
        test_dates=pd.date_range('2024-01-11', periods=1),
    )

    with absltest.mock.patch.object(
        analysis, '_get_estimated_bau_spend_holdback'
    ) as mock_holdback:
      result = analysis._get_estimated_bau_spend(
          analysis_config=config,
          cell='cell_1',
          geos=['G1', 'G2'],
          spend={'cell_1': cell_spend},
          conversions=cell_spend,
          experiment_type=api.ExperimentType.HOLDBACK,
      )
      self.assertIsNone(result)
      mock_holdback.assert_not_called()

  def test_get_estimated_bau_spend_parent_zero_pretest_days(self):
    design_config = api.DesignConfig(
        experiment_duration=datetime.timedelta(days=5),
        experiment_types=api.ExperimentType.HOLDBACK,
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
    )
    config = api.AnalysisConfig(
        n_placebo_candidates=10,
        design=study_design,
        analysis_start_date=pd.Timestamp('2024-01-11'),
        analysis_end_date=pd.Timestamp('2024-01-15'),
    )

    cell_spend = analysis.TimeSeries(
        pretest=jnp.ones((0, 2)),
        test=jnp.ones((1, 2)),
        pretest_dates=pd.date_range('2024-01-01', periods=0),
        test_dates=pd.date_range('2024-01-11', periods=1),
    )

    result = analysis._get_estimated_bau_spend(
        analysis_config=config,
        cell='cell_1',
        geos=['G1', 'G2'],
        spend={'cell_1': cell_spend},
        conversions=cell_spend,
        experiment_type=api.ExperimentType.HOLDBACK,
    )
    self.assertIsNone(result)

  def test_get_estimated_bau_spend_parent_zero_test_days(self):
    design_config = api.DesignConfig(
        experiment_duration=datetime.timedelta(days=5),
        experiment_types=api.ExperimentType.HOLDBACK,
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
    )
    config = api.AnalysisConfig(
        n_placebo_candidates=10,
        design=study_design,
        analysis_start_date=pd.Timestamp('2024-01-11'),
        analysis_end_date=pd.Timestamp('2024-01-15'),
    )

    cell_spend = analysis.TimeSeries(
        pretest=jnp.ones((1, 2)),
        test=jnp.ones((0, 2)),
        pretest_dates=pd.date_range('2024-01-01', periods=1),
        test_dates=pd.date_range('2024-01-11', periods=0),
    )

    result = analysis._get_estimated_bau_spend(
        analysis_config=config,
        cell='cell_1',
        geos=['G1', 'G2'],
        spend={'cell_1': cell_spend},
        conversions=cell_spend,
        experiment_type=api.ExperimentType.HOLDBACK,
    )
    self.assertIsNone(result)

  def test_get_estimated_bau_spend_parent_unsupported_experiment_type(self):
    design_config = api.DesignConfig(
        experiment_duration=datetime.timedelta(days=5),
        experiment_types=api.ExperimentType.HOLDBACK,
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
    )
    config = api.AnalysisConfig(
        n_placebo_candidates=10,
        design=study_design,
        analysis_start_date=pd.Timestamp('2024-01-11'),
        analysis_end_date=pd.Timestamp('2024-01-15'),
    )

    cell_spend = analysis.TimeSeries(
        pretest=jnp.ones((1, 2)),
        test=jnp.ones((1, 2)),
        pretest_dates=pd.date_range('2024-01-01', periods=1),
        test_dates=pd.date_range('2024-01-11', periods=1),
    )

    result = analysis._get_estimated_bau_spend(
        analysis_config=config,
        cell='cell_1',
        geos=['G1', 'G2'],
        spend={'cell_1': cell_spend},
        conversions=cell_spend,
        experiment_type=typing.cast(api.ExperimentType, None),
    )
    self.assertIsNone(result)

  @absltest.mock.patch.object(analysis.tbr, 'get_r2')
  @absltest.mock.patch.object(
      analysis.generate_candidates, 'get_random_candidates'
  )
  @absltest.mock.patch.object(analysis.design, 'prepare_data')
  def test_get_placebo_masks_multicell(
      self, mock_prepare_data, mock_get_random_candidates, mock_get_r2
  ):
    # Verify that multicell placebo selection correctly optimizes min(R2).
    # This ensures that we aggregate R-squared correctly across all metrics.
    design_config = api.DesignConfig(
        n_candidates=5,
        n_aa_test_iterations=2,
        experiment_duration=datetime.timedelta(days=1),
        experiment_types={
            'cell_1': api.ExperimentType.HOLDBACK,
            'cell_2': api.ExperimentType.HOLDBACK,
        },
        geo_assignment_rule=api.GeoAssignmentRule.RANDOM,
    )

    design_obj = api.Design(
        designs={},
        control_geos=set(),
        excluded_geos=set(),
        data=pd.DataFrame(
            {api.LOCATION: [], api.DATE: [], api.CONVERSIONS: []}
        ),
    )

    treatment = analysis.TreatmentMask(
        geos=['G1', 'G2'],
        mask=jnp.array([1.0, 2.0, 0.0, 0.0]),
    )

    mock_prepare_data.return_value = analysis.design.ProcessedData(
        selection_train=jnp.zeros((10, 4)),
        selection_eval=jnp.zeros((5, 4)),
        estimation_train=jnp.zeros((10, 4)),
        estimation_eval=jnp.zeros((5, 4)),
        training_period=[pd.Timestamp('2024-01-01')],
        filtered_data=pd.DataFrame(),
        selection_train_spend={},
    )

    mock_get_random_candidates.return_value = jnp.array([
        [0.0, 0.0],
        [1.0, 2.0],
        [2.0, 1.0],
        [1.0, 1.0],
        [2.0, 2.0],
    ])

    # For shape (5 candidates, 2 metrics), mock R2 scores.
    # The minimum across the two cells are:
    # 0.1, 0.8, 0.2, 0.85, 0.4.
    # We want to select the top 2 candidates, so those with minimum R2 0.85
    # and 0.8.
    # This corresponds to indices 3 and 1.
    mock_get_r2.return_value = jnp.array([
        [0.9, 0.1],
        [0.8, 0.8],
        [0.2, 0.9],
        [0.85, 0.85],
        [0.4, 0.6],
    ])

    analysis_config = api.AnalysisConfig(
        n_placebo_candidates=5,
        design=design_obj,
        analysis_start_date=pd.Timestamp('2024-01-11'),
        analysis_end_date=pd.Timestamp('2024-01-15'),
        min_placebo_r2=0.0,
        min_placebo_count_warning=0,
        min_placebo_count_error=0,
    )

    result_masks = analysis._get_placebo_masks(
        design_obj=design_obj,
        treatment=treatment,
        design_config=design_config,
        analysis_config=analysis_config,
        key=jax.random.PRNGKey(0),
    )

    # _get_full_mask populates the treatment mask with 0 at treatment geos,
    # and the placebo masks at the control geos.
    expected_masks = jnp.array([
        [0.0, 0.0, 1.0, 1.0],  # From candidate 3
        [0.0, 0.0, 1.0, 2.0],  # From candidate 1
    ])

    np.testing.assert_array_equal(result_masks, expected_masks)

  @absltest.mock.patch.object(analysis.tbr, 'get_r2')
  @absltest.mock.patch.object(
      analysis.generate_candidates, 'get_random_candidates'
  )
  @absltest.mock.patch.object(analysis.design, 'prepare_data')
  def test_get_placebo_masks_r2_filtering_and_thresholds(
      self, mock_prepare_data, mock_get_random_candidates, mock_get_r2
  ):
    # Test R-squared thresholding (min_placebo_r2) and warning/error thresholds
    design_config = api.DesignConfig(
        n_candidates=1000,
        n_aa_test_iterations=500,
        experiment_duration=datetime.timedelta(days=1),
        geo_assignment_rule=api.GeoAssignmentRule.RANDOM,
    )

    design_obj = api.Design(
        designs={},
        control_geos=set(),
        excluded_geos=set(),
        data=pd.DataFrame(
            {api.LOCATION: [], api.DATE: [], api.CONVERSIONS: []}
        ),
    )

    treatment = analysis.TreatmentMask(
        geos=['G1', 'G2'],
        mask=jnp.array([1.0, 2.0, 0.0, 0.0]),
    )

    mock_prepare_data.return_value = analysis.design.ProcessedData(
        selection_train=jnp.zeros((10, 4)),
        selection_eval=jnp.zeros((5, 4)),
        estimation_train=jnp.zeros((10, 4)),
        estimation_eval=jnp.zeros((5, 4)),
        training_period=[pd.Timestamp('2024-01-01')],
        filtered_data=pd.DataFrame(),
        selection_train_spend={},
    )

    analysis_config = api.AnalysisConfig(
        n_placebo_candidates=1000,
        design=design_obj,
        analysis_start_date=pd.Timestamp('2024-01-11'),
        analysis_end_date=pd.Timestamp('2024-01-15'),
        min_placebo_r2=0.6,
        min_placebo_count_warning=100,
        min_placebo_count_error=10,
    )

    def run_get_placebo_masks(r2_scores):
      n_cands = r2_scores.shape[0]
      # mock returned candidates as arange to track which were selected
      mock_get_random_candidates.return_value = jnp.arange(
          n_cands * 2, dtype=jnp.float32
      ).reshape((n_cands, 2))
      mock_get_r2.return_value = r2_scores

      return analysis._get_placebo_masks(
          design_obj=design_obj,
          treatment=treatment,
          design_config=design_config,
          analysis_config=analysis_config,
          key=jax.random.PRNGKey(0),
      )

    # 1. Abundance state: 1000 candidates, all pass R2 filter (>= 0.6)
    # Expected: filter leaves 1000, then sorts and subsets to top 500
    # (n_aa_test_iterations)
    r2_abundant = jnp.linspace(0.61, 0.99, 1000).reshape((1000, 1))
    result = run_get_placebo_masks(r2_abundant)
    self.assertEqual(result.shape[0], 500)

    # 2. Acceptable state: 1000 candidates, only 200 pass R2 filter (>= 0.6)
    # Expected: filter leaves 200, which is > 100 warning threshold,
    # so exactly 200 returned
    r2_acceptable = jnp.concatenate(
        [jnp.ones(200) * 0.8, jnp.ones(800) * 0.1]
    ).reshape((1000, 1))
    with absltest.mock.patch('absl.logging.warning') as mock_warning:
      result = run_get_placebo_masks(r2_acceptable)
      self.assertEqual(result.shape[0], 200)
      mock_warning.assert_not_called()

    # 3. Warning state: 1000 candidates, only 50 pass R2 filter
    # Expected: logs a warning, returns 50
    r2_warning = jnp.concatenate(
        [jnp.ones(50) * 0.8, jnp.ones(950) * 0.1]
    ).reshape((1000, 1))
    with absltest.mock.patch('absl.logging.warning') as mock_warning:
      result = run_get_placebo_masks(r2_warning)
      self.assertEqual(result.shape[0], 50)
      mock_warning.assert_called_once()
      warning_msg = mock_warning.call_args[0][0]
      self.assertIn('Low number of valid placebo candidates', warning_msg)

    # 4. Error state: 1000 candidates, only 5 pass R2 filter
    # Expected: ValueError raised
    r2_error = jnp.concatenate(
        [jnp.ones(5) * 0.8, jnp.ones(995) * 0.1]
    ).reshape((1000, 1))
    with self.assertRaisesRegex(
        ValueError, 'Insufficient valid placebo candidates'
    ):
      run_get_placebo_masks(r2_error)


if __name__ == '__main__':
  absltest.main()
