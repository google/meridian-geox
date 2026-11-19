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

import datetime
from typing import Any
from absl.testing import absltest
from absl.testing import parameterized
from meridian_geox import api


class ApiTest(parameterized.TestCase):

  def test_design_config_normalization_single_cell(self):
    config = api.DesignConfig(
        experiment_duration=datetime.timedelta(days=10),
        experiment_types=api.ExperimentType.HOLDBACK,
        cell_count=1,
        cost_per_incremental_conversion=1.5,
    )
    self.assertEqual(
        config.experiment_types, {'cell_1': api.ExperimentType.HOLDBACK}
    )
    self.assertEqual(config.cost_per_incremental_conversion, {'cell_1': 1.5})
    self.assertEqual(config.cell_count, 1)

  def test_design_config_normalization_multi_cell_scalar(self):
    config = api.DesignConfig(
        experiment_duration=datetime.timedelta(days=10),
        experiment_types=api.ExperimentType.HOLDBACK,
        cell_count=3,
        cost_per_incremental_conversion=2.0,
    )
    expected_types = {
        'cell_1': api.ExperimentType.HOLDBACK,
        'cell_2': api.ExperimentType.HOLDBACK,
        'cell_3': api.ExperimentType.HOLDBACK,
    }
    expected_cpic = {
        'cell_1': 2.0,
        'cell_2': 2.0,
        'cell_3': 2.0,
    }
    self.assertEqual(config.experiment_types, expected_types)
    self.assertEqual(config.cost_per_incremental_conversion, expected_cpic)
    self.assertEqual(config.cell_count, 3)

  def test_design_config_normalization_multi_cell_mixed(self):
    # cell_1 is HOLDBACK, cell_2 is GO_DARK.
    # CPIC should only be populated for cell_1.
    config = api.DesignConfig(
        experiment_duration=datetime.timedelta(days=10),
        experiment_types={
            'cell_1': api.ExperimentType.HOLDBACK,
            'cell_2': api.ExperimentType.GO_DARK,
        },
        cell_count=2,
        cost_per_incremental_conversion=3.0,
    )
    self.assertEqual(config.cell_count, 2)
    self.assertEqual(config.cost_per_incremental_conversion, {'cell_1': 3.0})

  def test_constraints_normalize_budget_holdback(self):
    constraints = api.Constraints(budget_constraint=api.Budget(budget=1000.0))
    experiment_types = {
        'cell_1': api.ExperimentType.HOLDBACK,
        'cell_2': api.ExperimentType.GO_DARK,
    }
    constraints.normalize(experiment_types)
    # After normalization, budget_constraint is a dict.
    self.assertIsInstance(constraints.budget_constraint, dict)
    # Both cells get the same Budget object.
    self.assertEqual(constraints.budget_constraint['cell_1'].budget, 1000.0)
    self.assertEqual(constraints.budget_constraint['cell_2'].budget, 1000.0)

  def test_constraints_normalize_budget_percent_godark(self):
    constraints = api.Constraints(budget_constraint=api.Budget(budget_pct=0.5))
    experiment_types = {
        'cell_1': api.ExperimentType.HOLDBACK,
        'cell_2': api.ExperimentType.GO_DARK,
        'cell_3': api.ExperimentType.HEAVY_UP,
    }
    constraints.normalize(experiment_types)
    # After normalization, budget_constraint is a dict.
    self.assertIsInstance(constraints.budget_constraint, dict)
    # All cells get the same Budget object.
    self.assertEqual(constraints.budget_constraint['cell_1'].budget_pct, 0.5)
    self.assertEqual(constraints.budget_constraint['cell_2'].budget_pct, 0.5)
    self.assertEqual(constraints.budget_constraint['cell_3'].budget_pct, 0.5)

  def test_budget_post_init_validation(self):
    with self.assertRaisesRegex(
        ValueError, "Exactly one of 'budget' or 'budget_pct' must be provided."
    ):
      api.Budget(budget=100, budget_pct=0.5)

    with self.assertRaisesRegex(
        ValueError, "Exactly one of 'budget' or 'budget_pct' must be provided."
    ):
      api.Budget()

  def test_design_config_default_cpic(self):
    config = api.DesignConfig(
        experiment_duration=datetime.timedelta(days=5),
        experiment_types=api.ExperimentType.HOLDBACK,
    )
    # Default cpic is 1.0.
    self.assertEqual(config.cost_per_incremental_conversion, {'cell_1': 1.0})

  def test_design_config_invalid_duration_type(self):
    with self.assertRaises(Exception):
      api.DesignConfig(
          experiment_duration=10,  # pytype: disable=wrong-arg-types
          experiment_types=api.ExperimentType.HOLDBACK,
      )

  def test_design_config_duration_sub_daily_rejected(self):
    with self.assertRaisesRegex(
        (ValueError, Exception),
        'experiment_duration must be in full days or weeks',
    ):
      api.DesignConfig(
          experiment_duration=datetime.timedelta(hours=6),
          experiment_types=api.ExperimentType.HOLDBACK,
      )
    with self.assertRaisesRegex(
        (ValueError, Exception),
        'experiment_duration must be in full days or weeks',
    ):
      api.DesignConfig(
          experiment_duration=datetime.timedelta(seconds=30),
          experiment_types=api.ExperimentType.HOLDBACK,
      )

  def test_analysis_config_date_order_validation(self):
    dummy_design = api.Design(
        designs={
            'cell_1': api.PerCellDesign(
                treatment_geos={'G0', 'G1'},
                minimum_detectable_effect=0.05,
                design_implied_cpic=1.0,
                p_value=0.05,
                budget=1000.0,
            )
        },
        control_geos={'G2'},
        excluded_geos=set(),
    )
    # Pydantic dataclass validation error wraps the ValueError or raises
    # ValidationError. Depending on Pydantic configuration, it might raise
    # a ValidationError. We check for ValueError here.
    with self.assertRaisesRegex(
        (ValueError, Exception),
        'analysis_start_date must be less than or equal to analysis_end_date.',
    ):
      api.AnalysisConfig(
          design=dummy_design,
          analysis_start_date='2024-01-25',
          analysis_end_date='2024-01-20',
      )

  def test_analysis_config_pretest_overlap_validation(self):
    dummy_design = api.Design(
        designs={
            'cell_1': api.PerCellDesign(
                treatment_geos={'G0', 'G1'},
                minimum_detectable_effect=0.05,
                design_implied_cpic=1.0,
                p_value=0.05,
                budget=1000.0,
            )
        },
        control_geos={'G2'},
        excluded_geos=set(),
    )
    with self.assertRaisesRegex(
        (ValueError, Exception),
        'pretest_end_date must be strictly before analysis_start_date.',
    ):
      api.AnalysisConfig(
          design=dummy_design,
          analysis_start_date='2024-01-20',
          analysis_end_date='2024-01-30',
          pretest_end_date='2024-01-20',
      )

  @parameterized.parameters(-0.1, 0.0, 1.0, 1.5)
  def test_analysis_config_alpha_validation(self, alpha):
    dummy_design = api.Design(
        designs={
            'cell_1': api.PerCellDesign(
                treatment_geos={'G0', 'G1'},
                minimum_detectable_effect=0.05,
                design_implied_cpic=1.0,
                p_value=0.05,
                budget=1000.0,
            )
        },
        control_geos={'G2'},
        excluded_geos=set(),
    )
    with self.assertRaisesRegex(ValueError, 'alpha'):
      api.AnalysisConfig(
          design=dummy_design,
          analysis_start_date='2024-01-20',
          analysis_end_date='2024-01-30',
          alpha=alpha,
      )

  def test_granularity_specs_cover_all_granularities(self):
    self.assertEqual(
        set(api.GRANULARITY_SPECS.keys()), set(api.TimeGranularity)
    )
    for granularity, spec in api.GRANULARITY_SPECS.items():
      self.assertEqual(spec.granularity, granularity)

  @parameterized.named_parameters(
      dict(
          testcase_name='daily_10_days',
          granularity=api.TimeGranularity.DAILY,
          duration=datetime.timedelta(days=10),
          expected_periods=10,
          expected_pretest_periods=30,
      ),
      dict(
          testcase_name='weekly_4_weeks',
          granularity=api.TimeGranularity.WEEKLY,
          duration=datetime.timedelta(weeks=4),
          expected_periods=4,
          expected_pretest_periods=12,
      ),
  )
  def test_granularity_spec_to_periods_and_pretest_periods(
      self,
      granularity,
      duration,
      expected_periods,
      expected_pretest_periods,
  ):
    spec = api.GRANULARITY_SPECS[granularity]
    self.assertEqual(spec.to_periods(duration), expected_periods)
    self.assertEqual(
        spec.required_pretest_periods(duration), expected_pretest_periods
    )
    self.assertEqual(spec.min_experiment_periods, 2)

  @parameterized.named_parameters(
      dict(
          testcase_name='weekly_non_multiple_days',
          granularity=api.TimeGranularity.WEEKLY,
          duration=datetime.timedelta(days=10),
          error_regex='must be a positive multiple of',
      ),
      dict(
          testcase_name='weekly_zero_days',
          granularity=api.TimeGranularity.WEEKLY,
          duration=datetime.timedelta(days=0),
          error_regex='must be a positive multiple of',
      ),
  )
  def test_granularity_spec_to_periods_invalid_duration_raises(
      self, granularity, duration, error_regex
  ):
    spec = api.GRANULARITY_SPECS[granularity]
    with self.assertRaisesRegex(ValueError, error_regex):
      spec.to_periods(duration)

  def test_design_config_time_granularity(self):
    default_config = api.DesignConfig(
        datetime.timedelta(days=14),
        api.ExperimentType.GO_DARK,
    )
    self.assertEqual(default_config.time_granularity, api.TimeGranularity.DAILY)
    self.assertEqual(
        default_config.experiment_types,
        {'cell_1': api.ExperimentType.GO_DARK},
    )

    weekly_config = api.DesignConfig(
        experiment_duration=datetime.timedelta(weeks=4),
        time_granularity=api.TimeGranularity.WEEKLY,
        experiment_types=api.ExperimentType.HOLDBACK,
    )
    self.assertEqual(weekly_config.time_granularity, api.TimeGranularity.WEEKLY)

    positional_args: list[Any] = [
        datetime.timedelta(weeks=4),
        api.TimeGranularity.WEEKLY,
    ]
    with self.assertRaisesRegex(ValueError, 'ExperimentType'):
      api.DesignConfig(*positional_args)


if __name__ == '__main__':
  absltest.main()
