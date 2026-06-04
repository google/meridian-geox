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

from absl.testing import absltest
from absl.testing import parameterized
from meridian_geox import api


class ApiTest(parameterized.TestCase):

  def test_design_config_normalization_single_cell(self):
    config = api.DesignConfig(
        experiment_duration=10,
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
        experiment_duration=10,
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
        experiment_duration=10,
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
    constraints = api.Constraints(budget=1000.0)
    experiment_types = {
        'cell_1': api.ExperimentType.HOLDBACK,
        'cell_2': api.ExperimentType.GO_DARK,
    }
    constraints.normalize(experiment_types)
    # Budget only for HOLDBACK cell.
    self.assertEqual(constraints.budget, {'cell_1': 1000.0})

  def test_constraints_normalize_budget_percent_godark(self):
    constraints = api.Constraints(budget_percent=0.5)
    experiment_types = {
        'cell_1': api.ExperimentType.HOLDBACK,
        'cell_2': api.ExperimentType.GO_DARK,
        'cell_3': api.ExperimentType.HEAVY_UP,
    }
    constraints.normalize(experiment_types)
    # Budget percent for GO_DARK and HEAVY_UP cells.
    expected_bp = {
        'cell_2': 0.5,
        'cell_3': 0.5,
    }
    self.assertEqual(constraints.budget_percent, expected_bp)

  def test_design_config_default_cpic(self):
    config = api.DesignConfig(
        experiment_duration=5,
        experiment_types=api.ExperimentType.HOLDBACK,
    )
    # Default cpic is 1.0.
    self.assertEqual(config.cost_per_incremental_conversion, {'cell_1': 1.0})


if __name__ == '__main__':
  absltest.main()
