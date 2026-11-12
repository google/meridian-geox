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

"""Tests for GeoX design charts."""

from typing import Any

from absl.testing import absltest
from meridian_geox import charts
from meridian_geox import design_charts
from meridian_geox import report_data
from meridian_geox import report_fixtures


def _spec(chart: Any) -> dict[str, Any]:
  return charts.configure(chart).to_dict()


class DesignChartsTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.design_set = report_fixtures.make_design_set(n_candidates=3)
    self.goals = report_data.DesignGoals(
        target_mde=0.03, target_budget=12_000, target_cpic=1.0
    )
    self.summaries = report_data.design_summaries(self.design_set, self.goals)
    self.design = self.design_set.designs[self.summaries[0].design_id]

  def test_sensitivity_curve(self):
    spec = _spec(
        design_charts.sensitivity_curve(
            self.summaries[0], self.design, target_mde=0.03
        )
    )
    # Goal rule, zone, zone label, power rule, MDE rule, curve, MDE point,
    # MDE label and goal label.
    self.assertLen(spec['layer'], 9)


if __name__ == '__main__':
  absltest.main()
