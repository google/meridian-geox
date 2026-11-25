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

import json
from typing import Any

from absl.testing import absltest
from meridian_geox import api
from meridian_geox import charts
from meridian_geox import design_charts
from meridian_geox import report_data
from meridian_geox import report_fixtures


def _spec(chart: Any) -> dict[str, Any]:
  return charts.configure(chart).to_dict()


def _params(spec: dict[str, Any]) -> dict[str, Any]:
  return {p['name']: p for p in spec.get('params', [])}


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

  def test_pretest_fit_chart(self):
    spec = _spec(design_charts.pretest_fit_chart(self.design))
    # Excluded date rule and lines.
    self.assertLen(spec['layer'], 2)

  def test_largest_geos_chart(self):
    table = report_data.geo_table(self.design, report_data.GeoNames())
    spec = _spec(
        design_charts.largest_geos_chart(
            table, charts.group_scale(['Treatment']), top_n=5
        )
    )
    self.assertEqual(spec['mark']['type'], 'bar')

  def test_power_simulator(self):
    spec = _spec(
        design_charts.power_simulator(
            self.summaries[0], self.design, self.goals, '$', 'Campaign spend',
            reference_cpic=1.0,
        )
    )
    params = _params(spec)
    self.assertEqual(
        set(params), {'sim_spend', 'sim_icpa', 'sim_tail', 'sim_conf'}
    )
    self.assertEqual(params['sim_spend']['value'], 12_000)
    self.assertEqual(params['sim_icpa']['value'], 1.0)
    self.assertEqual(params['sim_conf']['value'], 0.9)

  def test_power_simulator_multicell(self):
    types = {
        'cell_1': api.ExperimentType.GO_DARK,
        'cell_2': api.ExperimentType.HEAVY_UP,
    }
    design_set = report_fixtures.make_design_set(
        n_candidates=1, cells=2, experiment_types=types, target_budget=None
    )
    summaries = report_data.design_summaries(design_set)
    design = design_set.designs[summaries[0].design_id]
    spec = _spec(
        design_charts.power_simulator(
            summaries[0], design, report_data.DesignGoals(target_cpic=6.5)
        )
    )
    self.assertIn('sim_cell', _params(spec))

  def test_largest_geos_has_spend_share(self):
    table = report_data.geo_table(self.design, report_data.GeoNames())
    spec = _spec(
        design_charts.largest_geos_chart(
            table, charts.group_scale(['Treatment'])
        )
    )
    folds = [t for t in spec['transform'] if 'fold' in t]
    self.assertEqual(folds[0]['fold'], ['share', 'spend_share'])

  def test_tradeoff_toggle_chart(self):
    spec = _spec(
        design_charts.tradeoff_toggle_chart(
            self.summaries, self.goals, '$', 'Campaign spend'
        )
    )
    params = _params(spec)
    self.assertIn('tradeoff_axis', params)
    self.assertIn(charts.DESIGN_SELECTION, params)
    # Ideal zone + label, points, short labels and the selected label.
    self.assertLen(spec['layer'], 5)
    self.assertIn('Ideal zone', json.dumps(spec))

  def test_tradeoff_toggle_chart_quotes_currency(self):
    currency = "C$'"
    spec = _spec(
        design_charts.tradeoff_toggle_chart(
            self.summaries, self.goals, currency
        )
    )
    self.assertIn(json.dumps(json.dumps(currency))[1:-1], json.dumps(spec))

  def test_ideal_zone_is_clipped(self):
    zone = design_charts._ideal_zone(
        (10.0, 20.0), (0.01, 0.03), 15.0, 0.02, True
    )
    self.assertEqual(zone, {'x0': 10.0, 'x1': 15.0, 'y0': 0.01, 'y1': 0.02})
    # Goals outside the plotted range: no (inverted) zone.
    self.assertIsNone(
        design_charts._ideal_zone((10.0, 20.0), (0.01, 0.03), 5.0, 0.02, True)
    )
    self.assertIsNone(
        design_charts._ideal_zone((10.0, 20.0), (0.01, 0.03), 15.0, 0.005, True)
    )

  def test_aa_drift_chart(self):
    drift = report_data.aa_drift(self.design, 'cell_1')
    assert drift is not None
    spec = _spec(design_charts.aa_drift_chart(drift, 500.0))
    self.assertLen(spec['layer'], 3)

  def test_power_simulator_has_icpa_panel(self):
    spec = _spec(
        design_charts.power_simulator(
            self.summaries[0], self.design, self.goals, '$'
        )
    )
    # Numbers; noise vs. lift and iCPA panels; spend panel.
    self.assertLen(spec['vconcat'], 3)
    self.assertLen(spec['vconcat'][1]['hconcat'], 2)
    self.assertIn(
        'Would the expected lift stand out from noise?', json.dumps(spec)
    )
    stats = json.dumps(spec['vconcat'][0])
    self.assertIn('Conclusive-result chance', stats)
    self.assertIn('Max detectable iCPA', stats)


if __name__ == '__main__':
  absltest.main()
