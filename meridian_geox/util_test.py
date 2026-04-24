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
        alpha=0.1,
        power=0.8,
    )
    self.constraints = api.Constraints()

  def test_validate_design_input_success(self):
    errors = util.validate_design_input(
        self.data, self.design_config, self.constraints
    )
    self.assertEmpty(errors)

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
    design_config = api.DesignConfig(experiment_duration=11)
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
        alpha=alpha,
        power=power,
    )
    errors = util.validate_design_input(
        self.data, design_config, self.constraints
    )
    self.assertLen(errors, 1)
    self.assertIn(expected_error, errors[0])

  def test_validate_design_input_holdback_missing_cpic(self):
    design_config = api.DesignConfig(
        experiment_duration=5,
        experiment_types=api.ExperimentType.HOLDBACK,
        cost_per_incremental_conversion=None,
    )
    errors = util.validate_design_input(
        self.data, design_config, self.constraints
    )
    self.assertLen(errors, 1)
    self.assertIn(
        'Cost per incremental conversion must be set for HOLDBACK experiments',
        errors[0],
    )

  def test_validate_design_input_go_dark_missing_cpic_no_error(self):
    # Cost per incremental conversion is NOT required for GO_DARK.
    design_config = api.DesignConfig(
        experiment_duration=5,
        experiment_types=api.ExperimentType.GO_DARK,
        cost_per_incremental_conversion=None,
    )
    errors = util.validate_design_input(
        self.data, design_config, self.constraints
    )
    self.assertEmpty(errors)


if __name__ == '__main__':
  absltest.main()
