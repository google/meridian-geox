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
from meridian_geox import analysis_charts
from meridian_geox import api
from meridian_geox import charts
from meridian_geox import report_data
from meridian_geox import report_fixtures


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

  def test_observed_counterfactual(self):
    spec = _spec(
        analysis_charts.observed_counterfactual_chart(
            self.metrics, self.window
        )
    )
    # Test + cooldown shading, CI band and lines.
    self.assertLen(spec['layer'], 3)
    phase_data = spec['layer'][0]['data']
    values = phase_data.get('values') or spec['datasets'][phase_data['name']]
    self.assertEqual([v['phase'] for v in values], ['Test', 'Cooldown'])

  def test_pointwise_and_cumulative(self):
    self.assertLen(
        _spec(analysis_charts.pointwise_chart(self.metrics, self.window))[
            'layer'
        ],
        4,
    )
    self.assertLen(
        _spec(analysis_charts.cumulative_chart(self.metrics, self.window))[
            'layer'
        ],
        4,
    )
    icpd = analysis_charts.cumulative_icpd_chart(self.metrics, self.window)
    assert icpd is not None
    self.assertLen(_spec(icpd)['layer'], 4)

  def test_cumulative_icpd_without_spend(self):
    result = report_fixtures.make_analysis_result(self.design, icpd=None)
    self.assertIsNone(
        analysis_charts.cumulative_icpd_chart(
            result.results['cell_1'], self.window
        )
    )

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

  def test_confidence_curve(self):
    chart = analysis_charts.confidence_curve(
        self.metrics.lift, 'Extra conversions', ',.0f'
    )
    assert chart is not None
    self.assertLen(_spec(chart)['layer'], 2)
    icpd = self.metrics.icpd
    assert icpd is not None
    cpa = analysis_charts.confidence_curve(
        icpd, 'iCPA', ',.2f', transform=lambda x: 1 / x if x > 0 else
        float('inf')
    )
    self.assertIsNotNone(cpa)

  def test_confidence_curve_without_sd(self):
    estimate = api.Estimate(1.0, 1.0, 1.0, 0.0, 0.5)
    self.assertIsNone(analysis_charts.confidence_curve(estimate, 'x', ',.0f'))

  def test_confidence_curve_uses_interval(self):
    estimate = api.Estimate(10.0, 0.0, 20.0, 999.0, 0.5)
    chart = analysis_charts.confidence_curve(estimate, 'x', ',.0f', level=0.9)
    assert chart is not None
    spec = _spec(chart)
    data = spec['layer'][1]['data']
    values = data.get('values') or spec['datasets'][data['name']]
    xs = [v['value'] for v in values]
    # sd = 20 / (2 * 1.645) ≈ 6.08, so the curve spans about ±24.
    self.assertLess(max(xs), 40)

  def test_spend_pacing(self):
    spend = report_data.daily_spend_by_group(self.result, self.data, 'cell_1')
    chart = analysis_charts.spend_pacing_chart(
        spend, self.window, ['Treatment']
    )
    assert chart is not None
    self.assertLen(_spec(chart)['layer'], 2)

  def test_anomaly_heatmap(self):
    z = report_data.robust_z_scores(self.data)
    chart = analysis_charts.anomaly_heatmap(
        z, report_data.GeoNames(), self.window
    )
    assert chart is not None
    self.assertLen(_spec(chart)['layer'], 2)


if __name__ == '__main__':
  absltest.main()
