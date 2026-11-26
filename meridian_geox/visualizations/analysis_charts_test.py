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

"""Tests for GeoX analysis charts."""

from typing import Any

from absl.testing import absltest
from meridian_geox.visualizations import analysis_charts
from meridian_geox.visualizations import charts
from meridian_geox.visualizations import report_data
from meridian_geox.visualizations import report_fixtures


def _spec(chart: Any) -> dict[str, Any]:
  return charts.configure(chart).to_dict()


class AnalysisChartsTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    geos = report_fixtures.locations(8)
    self.design = report_fixtures.make_design(
        geos, {'cell_1': geos[:2]}, target_budget=None
    )
    self.result = report_fixtures.make_analysis_result(self.design)
    self.metrics = self.result.results['cell_1']
    self.window = report_data.analysis_window(self.result)
    self.data = report_fixtures.make_analysis_data(self.design)

  def test_effect_totals(self):
    spec = _spec(
        analysis_charts.effect_totals_chart(
            self.result, {'cell_1': 'Treatment'}
        )
    )
    self.assertLen(spec['hconcat'], 2)
    self.assertEqual(spec['hconcat'][0]['layer'][0]['mark']['type'], 'bar')
    self.assertEqual(
        spec['hconcat'][1]['title'], 'Incremental = observed − counterfactual'
    )


if __name__ == '__main__':
  absltest.main()
