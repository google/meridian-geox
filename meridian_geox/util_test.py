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

"""Tests for GeoX utility functions."""

import logging
from absl.testing import absltest
from absl.testing import parameterized
from meridian_geox import api
from meridian_geox import util
import pandas as pd


class UtilTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.dates = pd.date_range('2024-01-01', periods=30)
    self.geos = [f'G{i}' for i in range(10)]
    data_rows = []
    for d in self.dates:
      for g in self.geos:
        data_rows.append({'date': d, 'location': g, 'conversions': 100.0})
    self.data = pd.DataFrame(data_rows)

    self.design_config = api.DesignConfig(
        experiment_duration=10,
        experiment_types=api.ExperimentType.HOLDBACK,
        alpha=0.1,
        power=0.8,
    )
    self.constraints = api.Constraints()

  def test_validate_design_input_success(self):
    errors = util.validate_design_input(
        self.data, self.design_config, self.constraints
    )
    self.assertEmpty(errors)

  def test_validate_design_input_multiple_spend_cells_success(self):
    data = self.data.copy()
    data['spend_cell_1'] = 10.0
    data['spend_cell_2'] = 20.0
    data['spend_cell_3'] = 30.0
    errors = util.validate_design_input(
        data, self.design_config, self.constraints
    )
    self.assertEmpty(errors)

  def test_validate_design_input_spend_cell_negative_error(self):
    data = self.data.copy()
    data['spend_cell_1'] = -10.0
    errors = util.validate_design_input(
        data, self.design_config, self.constraints
    )
    self.assertNotEmpty(errors)

  def test_validate_design_input_schema_error(self):
    # Missing 'location' column.
    invalid_data = self.data.drop(columns=['location'])
    errors = util.validate_design_input(
        invalid_data, self.design_config, self.constraints
    )
    self.assertNotEmpty(errors)
    self.assertIn('location', errors[0])

  def test_validate_design_input_not_enough_dates(self):
    # experiment_duration is 10, need at least 30 dates.
    # We have exactly 30 dates in setUp. Let's set duration to 11.
    design_config = api.DesignConfig(
        experiment_duration=11, experiment_types=api.ExperimentType.HOLDBACK
    )
    errors = util.validate_design_input(
        self.data, design_config, self.constraints
    )
    self.assertLen(errors, 1)
    self.assertIn('Need at least 3 * experiment duration dates', errors[0])

  def test_validate_design_input_not_enough_geos_after_exclusion(self):
    # Exclude 7 out of 10 geos, leaving only 3. Need at least 4.
    constraints = api.Constraints(excluded_geos={f'G{i}' for i in range(7)})
    errors = util.validate_design_input(
        self.data, self.design_config, constraints
    )
    self.assertLen(errors, 1)
    self.assertIn(
        'Not enough geos left after excluding excluded geos', errors[0]
    )

  def test_validate_design_input_overlap_excluded_included_control(self):
    constraints = api.Constraints(
        excluded_geos={'G1'}, included_control_geos={'G1'}
    )
    errors = util.validate_design_input(
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
    errors = util.validate_design_input(
        self.data, design_config, self.constraints
    )
    self.assertLen(errors, 1)
    self.assertIn(expected_error, errors[0])

  def test_validate_design_input_holdback_missing_cpic(self):
    # It must be > 0.0 if methodology is HOLDBACK.
    design_config = api.DesignConfig(
        experiment_duration=5,
        experiment_types=api.ExperimentType.HOLDBACK,
        cost_per_incremental_conversion=0.0,
    )
    errors = util.validate_design_input(
        self.data, design_config, self.constraints
    )
    self.assertLen(errors, 1)
    self.assertIn(
        'Cost per incremental conversion must be set and larger than 0',
        errors[0],
    )

  def test_validate_design_input_go_dark_missing_cpic_no_error(self):
    # Cost per incremental conversion is NOT required for GO_DARK.
    design_config = api.DesignConfig(
        experiment_duration=5,
        experiment_types=api.ExperimentType.GO_DARK,
        cost_per_incremental_conversion=0.0,
    )
    errors = util.validate_design_input(
        self.data, design_config, self.constraints
    )
    self.assertEmpty(errors)

  def test_validate_design_input_holdback_no_budget_warning(self):
    # Cell 1 is HOLDBACK but no budget provided.
    design_config = api.DesignConfig(
        experiment_duration=5,
        experiment_types={'cell_1': api.ExperimentType.HOLDBACK},
        cost_per_incremental_conversion={'cell_1': 1.0},
    )
    with absltest.mock.patch.object(logging, 'warning') as mock_warning:
      util.validate_design_input(self.data, design_config, self.constraints)
      mock_warning.assert_any_call(
          'Cell %s is HOLDBACK but no budget is provided.', 'cell_1'
      )

  def test_validate_design_input_holdback_budget_pct_warning(self):
    # Cell 1 is HOLDBACK but budget_pct provided instead of budget.
    design_config = api.DesignConfig(
        experiment_duration=5,
        experiment_types={'cell_1': api.ExperimentType.HOLDBACK},
        cost_per_incremental_conversion={'cell_1': 1.0},
    )
    constraints = api.Constraints(
        budget_constraint={'cell_1': api.Budget(budget_pct=0.1)}
    )
    with absltest.mock.patch.object(logging, 'warning') as mock_warning:
      util.validate_design_input(self.data, design_config, constraints)
      mock_warning.assert_any_call(
          'Cell %s is HOLDBACK but budget_pct is provided.', 'cell_1'
      )

  def test_validate_design_input_go_dark_no_budget_pct_warning(self):
    # Cell 1 is GO_DARK but no budget_pct provided.
    design_config = api.DesignConfig(
        experiment_duration=5,
        experiment_types={'cell_1': api.ExperimentType.GO_DARK},
    )
    with absltest.mock.patch.object(logging, 'warning') as mock_warning:
      util.validate_design_input(self.data, design_config, self.constraints)
      mock_warning.assert_any_call(
          'Cell %s is %s but budget_pct is not provided. Using 100%% as'
          ' default.',
          'cell_1',
          'GO_DARK',
      )

  def test_validate_design_input_go_dark_absolute_budget_warning(self):
    # Cell 1 is GO_DARK but absolute budget provided instead of budget_pct.
    design_config = api.DesignConfig(
        experiment_duration=5,
        experiment_types={'cell_1': api.ExperimentType.GO_DARK},
    )
    constraints = api.Constraints(
        budget_constraint={'cell_1': api.Budget(budget=1000.0)}
    )
    with absltest.mock.patch.object(logging, 'warning') as mock_warning:
      util.validate_design_input(self.data, design_config, constraints)
      mock_warning.assert_any_call(
          'Cell %s is %s but budget (absolute) is provided instead of'
          ' budget_pct.',
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
        cell_count=3,  # Mismatch.
    )
    errors = util.validate_design_input(
        self.data, design_config, self.constraints
    )
    self.assertLen(errors, 1)
    self.assertIn('cell_count (3) must match', errors[0])

  def test_cell_id_from_cell_name_success(self):
    self.assertEqual(util.cell_id_from_cell_name('cell_3'), 3)
    self.assertEqual(util.cell_id_from_cell_name('cell_123'), 123)

  @parameterized.named_parameters(
      ('zero', 'cell_0'),
      ('negative', 'cell_-5'),
      ('non_integer', 'cell_abc'),
      ('wrong_prefix', 'cell1'),
      ('other_string', 'group_1'),
  )
  def test_cell_id_from_cell_name_value_error(self, cell_name):
    with self.assertRaises(ValueError):
      util.cell_id_from_cell_name(cell_name)


if __name__ == '__main__':
  absltest.main()
