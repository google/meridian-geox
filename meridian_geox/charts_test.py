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

"""Tests for the shared GeoX chart styling."""

import json

from absl.testing import absltest
import altair as alt
from meridian_geox import charts
import pandas as pd


def _chart() -> alt.Chart:
  return (
      alt.Chart(pd.DataFrame({'x': [1, 2], 'y': [3, 4]}))
      .mark_line()
      .encode(x='x:Q', y='y:Q')
  )


class ChartsTest(absltest.TestCase):

  def test_treatment_color_wraps_around(self):
    n = len(charts.TREATMENT_COLORS)
    self.assertEqual(charts.treatment_color(0), charts.TREATMENT_COLORS[0])
    self.assertEqual(charts.treatment_color(n), charts.TREATMENT_COLORS[0])
    self.assertEqual(charts.treatment_color(n + 1), charts.TREATMENT_COLORS[1])

  def test_treatment_color_rejects_negative_index(self):
    with self.assertRaises(ValueError):
      charts.treatment_color(-1)

  def test_treatment_colors_avoid_control_and_excluded(self):
    self.assertNotIn(charts.CONTROL_COLOR, charts.TREATMENT_COLORS)
    self.assertNotIn(charts.EXCLUDED_COLOR, charts.TREATMENT_COLORS)

  def test_group_scale_aligns_domain_and_range(self):
    scale = charts.group_scale(['Treatment 1', 'Treatment 2']).to_dict()
    self.assertEqual(
        scale['domain'],
        [
            'Treatment 1',
            'Treatment 2',
            charts.CONTROL_LABEL,
            charts.EXCLUDED_LABEL,
            charts.NOT_IN_TEST_LABEL,
        ],
    )
    self.assertEqual(
        scale['range'],
        [
            charts.treatment_color(0),
            charts.treatment_color(1),
            charts.CONTROL_COLOR,
            charts.EXCLUDED_COLOR,
            charts.GREY_100,
        ],
    )

  def test_group_colors(self):
    self.assertEqual(
        charts.group_colors(['T1']),
        {'T1': charts.treatment_color(0),
         charts.CONTROL_LABEL: charts.CONTROL_COLOR,
         charts.EXCLUDED_LABEL: charts.EXCLUDED_COLOR},
    )

  def test_framed_sets_title_and_subtitle(self):
    spec = charts.framed(
        _chart(), charts.Framing('Where would the test run?', subtitle='Sub')
    ).to_dict()
    self.assertEqual(spec['title']['text'], 'Where would the test run?')
    self.assertEqual(spec['title']['subtitle'], 'Sub')
    self.assertEqual(spec['title']['anchor'], 'start')

  def test_framed_without_subtitle(self):
    spec = charts.framed(_chart(), charts.Framing('Title')).to_dict()
    self.assertNotIn('subtitle', spec['title'])

  def test_configure_applies_shared_look(self):
    spec = charts.configure(_chart()).to_dict()
    self.assertEqual(spec['config']['font'], charts.FONT)
    self.assertEqual(spec['config']['legend']['orient'], 'bottom')

  def test_to_json_allows_more_than_5000_rows(self):
    chart = alt.Chart(pd.DataFrame({'x': range(6000), 'y': range(6000)}))
    spec = json.loads(charts.to_json(chart.mark_line().encode(x='x', y='y')))
    self.assertLen(next(iter(spec['datasets'].values())), 6000)

  def test_to_json_applies_shared_look(self):
    spec = json.loads(charts.to_json(_chart()))
    self.assertEqual(spec['config']['font'], charts.FONT)
    self.assertEqual(spec['config']['background'], 'transparent')
    self.assertEqual(spec['mark']['type'], 'line')


if __name__ == '__main__':
  absltest.main()
