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

"""Tests for GeoX validation functions."""

import logging
from absl.testing import absltest
from absl.testing import parameterized
from meridian_geox import api
from meridian_geox import validation
import pandas as pd


class ValidationTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.dates = pd.date_range('2024-01-01', periods=30)
    self.geos = [f'G{i}' for i in range(10)]
    data_rows = []
    for d in self.dates:
      for g in self.geos:
        data_rows.append({
            'date': d,
            'location': g,
            'conversions': 100.0,
            'spend': 10.0,
        })
    self.data = pd.DataFrame(data_rows)

    self.design_config = api.DesignConfig(
        experiment_duration=10,
        experiment_types=api.ExperimentType.HOLDBACK,
        alpha=0.1,
        power=0.8,
    )
    self.constraints = api.Constraints(
        budget_constraint={'cell_1': api.Budget(budget=1000.0)}
    )

    self.design = api.Design(
        designs={
            'cell_1': api.PerCellDesign(
                treatment_geos={'G0', 'G1', 'G2'},
                minimum_detectable_effect=0.05,
                design_implied_cpic=1.0,
                p_value=0.05,
                budget=1000.0,
            )
        },
        control_geos={'G3', 'G4', 'G5', 'G6', 'G7', 'G8', 'G9'},
        excluded_geos=set(),
        design_config=self.design_config,
        constraints=self.constraints,
    )

    self.analysis_config = api.AnalysisConfig(
        design=self.design,
        analysis_start_date=pd.Timestamp('2024-01-21'),
        analysis_end_date=pd.Timestamp('2024-01-30'),
        pretest_end_date=pd.Timestamp('2024-01-20'),
    )

  def test_validate_general_checks_success(self):
    errors = validation.validate_general_checks(self.data)
    self.assertEmpty(errors)

  def test_validate_general_checks_missing_cols(self):
    invalid_data = self.data.drop(columns=['location'])
    errors = validation.validate_general_checks(invalid_data)
    self.assertNotEmpty(errors)
    self.assertIn('location', errors[0])

  def test_validate_general_checks_null_conversions(self):
    invalid_data = self.data.copy()
    invalid_data.loc[0, 'conversions'] = None
    errors = validation.validate_general_checks(invalid_data)
    self.assertNotEmpty(errors)
    self.assertTrue(
        any('conversions' in err and 'null values' in err for err in errors)
    )

  def test_validate_general_checks_null_spend(self):
    invalid_data = self.data.copy()
    invalid_data.loc[0, 'spend'] = None
    errors = validation.validate_general_checks(invalid_data)
    self.assertNotEmpty(errors)
    self.assertTrue(
        any('spend' in err and 'null values' in err for err in errors)
    )

  def test_validate_general_checks_null_date(self):
    invalid_data = self.data.copy()
    invalid_data.loc[0, 'date'] = None
    errors = validation.validate_general_checks(invalid_data)
    self.assertNotEmpty(errors)
    self.assertTrue(
        any('date' in err and 'null values' in err for err in errors)
    )

  def test_validate_general_checks_null_location(self):
    invalid_data = self.data.copy()
    invalid_data.loc[0, 'location'] = None
    errors = validation.validate_general_checks(invalid_data)
    self.assertNotEmpty(errors)
    self.assertTrue(
        any('location' in err and 'null values' in err for err in errors)
    )

  def test_validate_general_checks_total_response_non_positive(self):
    invalid_data = self.data.copy()
    invalid_data['conversions'] = 0.0
    errors = validation.validate_general_checks(invalid_data)
    self.assertNotEmpty(errors)
    self.assertIn('Total conversions must be greater than 0.', errors)

  def test_validate_general_checks_weekly_pattern(self):
    weekly_dates = pd.date_range('2024-01-01', periods=4, freq='7D')
    data_rows = []
    for d in weekly_dates:
      for g in self.geos:
        data_rows.append({
            'date': d,
            'location': g,
            'conversions': 100.0,
            'spend': 10.0,
        })
    invalid_data = pd.DataFrame(data_rows)
    errors = validation.validate_general_checks(invalid_data)
    self.assertNotEmpty(errors)
    self.assertIn('Weekly patterns are not supported.', errors)

  def test_validate_design_input_success(self):
    errors = validation.validate_design_input(
        self.data, self.design_config, self.constraints
    )
    self.assertEmpty(errors)

  def test_validate_design_input_multiple_spend_cells_success(self):
    data = self.data.copy()
    data['spend_cell_1'] = 10.0
    data['spend_cell_2'] = 20.0
    data['spend_cell_3'] = 30.0
    errors = validation.validate_design_input(
        data, self.design_config, self.constraints
    )
    self.assertEmpty(errors)

  def test_validate_design_input_spend_cell_negative_error(self):
    data = self.data.copy()
    data['spend_cell_1'] = -10.0
    errors = validation.validate_design_input(
        data, self.design_config, self.constraints
    )
    self.assertNotEmpty(errors)

  def test_validate_design_input_schema_error(self):
    invalid_data = self.data.drop(columns=['location'])
    errors = validation.validate_design_input(
        invalid_data, self.design_config, self.constraints
    )
    self.assertNotEmpty(errors)
    self.assertIn('location', errors[0])

  def test_validate_design_input_not_enough_dates(self):
    design_config = api.DesignConfig(
        experiment_duration=11, experiment_types=api.ExperimentType.HOLDBACK
    )
    errors = validation.validate_design_input(
        self.data, design_config, self.constraints
    )
    self.assertLen(errors, 1)
    self.assertIn(
        'Need at least 3 * experiment duration dates during design phase.',
        errors[0],
    )

  def test_validate_design_input_not_enough_geos_after_exclusion(self):
    constraints = api.Constraints(
        excluded_geos={f'G{i}' for i in range(7)},
        budget_constraint={'cell_1': api.Budget(budget=1000.0)},
    )
    errors = validation.validate_design_input(
        self.data, self.design_config, constraints
    )
    self.assertLen(errors, 1)
    self.assertIn(
        'Not enough geos left after excluding excluded geos', errors[0]
    )

  def test_validate_design_input_overlap_excluded_included_control(self):
    constraints = api.Constraints(
        excluded_geos={'G1'},
        included_control_geos={'G1'},
        budget_constraint={'cell_1': api.Budget(budget=1000.0)},
    )
    errors = validation.validate_design_input(
        self.data, self.design_config, constraints
    )
    self.assertLen(errors, 1)
    self.assertIn(
        'Excluded geos must not overlap with included control geos', errors[0]
    )

  @parameterized.named_parameters(
      ('alpha_zero', 0.0, 0.8, 'Alpha must be between 0 and 1'),
      ('alpha_one', 1.0, 0.8, 'Alpha must be between 0 and 1'),
      ('power_zero', 0.1, 0.0, 'Power must be between 0 and 1'),
      ('power_one', 0.1, 1.0, 'Power must be between 0 and 1'),
  )
  def test_validate_design_input_invalid_alpha_power(
      self, alpha, power, expected_error
  ):
    design_config = api.DesignConfig(
        experiment_duration=5,
        experiment_types=api.ExperimentType.HOLDBACK,
        alpha=alpha,
        power=power,
    )
    errors = validation.validate_design_input(
        self.data, design_config, self.constraints
    )
    self.assertLen(errors, 1)
    self.assertIn(expected_error, errors[0])

  def test_validate_design_input_holdback_missing_cpic(self):
    design_config = api.DesignConfig(
        experiment_duration=5,
        experiment_types=api.ExperimentType.HOLDBACK,
        cost_per_incremental_conversion=0.0,
    )
    errors = validation.validate_design_input(
        self.data, design_config, self.constraints
    )
    self.assertLen(errors, 1)
    self.assertIn(
        'Cost per incremental conversion must be set and larger than 0',
        errors[0],
    )

  def test_validate_design_input_go_dark_missing_cpic_no_error(self):
    design_config = api.DesignConfig(
        experiment_duration=5,
        experiment_types=api.ExperimentType.GO_DARK,
        cost_per_incremental_conversion=0.0,
    )
    constraints = api.Constraints(
        budget_constraint={'cell_1': api.Budget(budget_pct=0.5)}
    )
    errors = validation.validate_design_input(
        self.data, design_config, constraints
    )
    self.assertEmpty(errors)

  def test_validate_design_input_holdback_no_budget_warning(self):
    design_config = api.DesignConfig(
        experiment_duration=5,
        experiment_types={'cell_1': api.ExperimentType.HOLDBACK},
        cost_per_incremental_conversion={'cell_1': 1.0},
    )
    constraints = api.Constraints(budget_constraint={})
    with absltest.mock.patch.object(logging, 'warning') as mock_warning:
      validation.validate_design_input(self.data, design_config, constraints)
      mock_warning.assert_any_call(
          '%s is HOLDBACK but no budget is provided.', 'cell_1'
      )

  def test_validate_design_input_holdback_budget_pct_warning(self):
    design_config = api.DesignConfig(
        experiment_duration=5,
        experiment_types={'cell_1': api.ExperimentType.HOLDBACK},
        cost_per_incremental_conversion={'cell_1': 1.0},
    )
    constraints = api.Constraints(
        budget_constraint={'cell_1': api.Budget(budget_pct=0.1)}
    )
    with absltest.mock.patch.object(logging, 'warning') as mock_warning:
      validation.validate_design_input(self.data, design_config, constraints)
      mock_warning.assert_any_call(
          '%s is HOLDBACK but budget_pct is provided.', 'cell_1'
      )

  def test_validate_design_input_go_dark_no_budget_pct_warning(self):
    design_config = api.DesignConfig(
        experiment_duration=5,
        experiment_types={'cell_1': api.ExperimentType.GO_DARK},
    )
    with absltest.mock.patch.object(logging, 'warning') as mock_warning:
      validation.validate_design_input(
          self.data, design_config, self.constraints
      )
      mock_warning.assert_any_call(
          '%s is %s but budget_pct is not provided. Using 100%% as default.',
          'cell_1',
          'GO_DARK',
      )

  def test_validate_design_input_go_dark_absolute_budget_warning(self):
    design_config = api.DesignConfig(
        experiment_duration=5,
        experiment_types={'cell_1': api.ExperimentType.GO_DARK},
    )
    constraints = api.Constraints(
        budget_constraint={'cell_1': api.Budget(budget=1000.0)}
    )
    with absltest.mock.patch.object(logging, 'warning') as mock_warning:
      validation.validate_design_input(self.data, design_config, constraints)
      mock_warning.assert_any_call(
          '%s is %s but budget (absolute) is provided instead of budget_pct.',
          'cell_1',
          'GO_DARK',
      )

  def test_validate_design_input_cell_count_mismatch(self):
    design_config = api.DesignConfig(
        experiment_duration=5,
        experiment_types={
            'cell_1': api.ExperimentType.HOLDBACK,
            'cell_2': api.ExperimentType.GO_DARK,
        },
        cell_count=3,
    )
    errors = validation.validate_design_input(
        self.data, design_config, self.constraints
    )
    self.assertLen(errors, 1)
    self.assertIn('Experiment types must be set for 3 cells', errors[0])

  def test_validate_design_input_single_cell_godark_missing_spend_error(self):
    design_config = api.DesignConfig(
        experiment_duration=5,
        experiment_types=api.ExperimentType.GO_DARK,
    )
    constraints = api.Constraints(
        budget_constraint={'cell_1': api.Budget(budget_pct=0.5)}
    )
    data = self.data.drop(columns=['spend'])
    errors = validation.validate_design_input(data, design_config, constraints)
    self.assertLen(errors, 1)
    self.assertIn(
        'Spend column must be set for GO_DARK experiments in cell cell_1. '
        'Expected spend or spend_cell_1.',
        errors[0],
    )

  def test_validate_design_input_multicell_godark_missing_cell_2_spend_error(
      self,
  ):
    design_config = api.DesignConfig(
        experiment_duration=5,
        cell_count=2,
        experiment_types={
            'cell_1': api.ExperimentType.GO_DARK,
            'cell_2': api.ExperimentType.GO_DARK,
        },
    )
    constraints = api.Constraints(
        budget_constraint={
            'cell_1': api.Budget(budget_pct=0.5),
            'cell_2': api.Budget(budget_pct=0.5),
        }
    )
    data = self.data.copy()
    data['spend_cell_1'] = 10.0
    errors = validation.validate_design_input(data, design_config, constraints)
    self.assertLen(errors, 1)
    self.assertIn(
        'Spend column must be set for GO_DARK experiments in cell cell_2. '
        'Expected spend_cell_2.',
        errors[0],
    )

  def test_validate_design_input_max_conversions_percent_error(self):
    design_config = api.DesignConfig(
        experiment_duration=5,
        experiment_types={'cell_1': api.ExperimentType.HOLDBACK},
        cost_per_incremental_conversion={'cell_1': 1.0},
    )
    constraints = api.Constraints(
        budget_constraint={'cell_1': api.Budget(budget=1000.0)},
        max_conversions_percent=0.5,
    )
    errors = validation.validate_design_input(
        self.data, design_config, constraints
    )
    self.assertLen(errors, 1)
    self.assertIn('Max conversions percent must be less than 0.5.', errors[0])

  def test_validate_design_input_max_conversions_percent_warning(self):
    design_config = api.DesignConfig(
        experiment_duration=5,
        experiment_types={'cell_1': api.ExperimentType.HOLDBACK},
        cost_per_incremental_conversion={'cell_1': 1.0},
    )
    constraints = api.Constraints(
        budget_constraint={'cell_1': api.Budget(budget=1000.0)},
        max_conversions_percent=0.08,
    )
    with absltest.mock.patch.object(logging, 'warning') as mock_warning:
      errors = validation.validate_design_input(
          self.data, design_config, constraints
      )
      self.assertEmpty(errors)
      mock_warning.assert_any_call(
          'Max conversions percent is less than 0.1 times the number of cells.'
      )

  def test_validate_analysis_input_success(self):
    errors = validation.validate_analysis_input(self.data, self.analysis_config)
    self.assertEmpty(errors)

  def test_validate_analysis_input_geo_set_mismatch(self):
    data = self.data.copy()
    data.loc[data['location'] == 'G9', 'location'] = 'G10'
    errors = validation.validate_analysis_input(data, self.analysis_config)
    self.assertNotEmpty(errors)
    self.assertIn(
        'The locations in the analysis data do not match',
        errors[0],
    )

  def test_validate_analysis_input_pretest_duration_insufficient(self):
    design_config = api.DesignConfig(
        experiment_duration=16,
        experiment_types=api.ExperimentType.HOLDBACK,
    )
    design = api.Design(
        designs={
            'cell_1': api.PerCellDesign(
                treatment_geos={'G0', 'G1', 'G2'},
                minimum_detectable_effect=0.05,
                design_implied_cpic=1.0,
                p_value=0.05,
                budget=1000.0,
            )
        },
        control_geos={'G3', 'G4', 'G5', 'G6', 'G7', 'G8', 'G9'},
        excluded_geos=set(),
        design_config=design_config,
        constraints=self.constraints,
    )
    config = api.AnalysisConfig(
        design=design,
        analysis_start_date=pd.Timestamp('2024-01-20'),
        analysis_end_date=pd.Timestamp('2024-01-30'),
        pretest_end_date=pd.Timestamp('2024-01-19'),
    )
    errors = validation.validate_analysis_input(self.data, config)
    self.assertNotEmpty(errors)
    self.assertIn(
        'Need at least 2 * experiment duration pretest dates during analysis'
        ' phase.',
        errors[0],
    )


if __name__ == '__main__':
  absltest.main()
