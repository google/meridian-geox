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

  def test_framed_sets_title_and_subtitle(self):
    spec = charts.framed(
        _chart(), charts.Framing('Where would the test run?', subtitle='Sub')
    ).to_dict()
    self.assertEqual(spec['title']['text'], 'Where would the test run?')
    self.assertEqual(spec['title']['subtitle'], 'Sub')
    self.assertEqual(spec['title']['anchor'], 'start')

  def test_framed_without_subtitle(self):
    spec = charts.framed(_chart(), charts.Framing('Title')).to_dict()
    self.assertEqual(spec['title']['subtitle'], '')

  def test_configure_applies_shared_look(self):
    spec = charts.configure(_chart()).to_dict()
    self.assertEqual(spec['config']['font'], charts.FONT)
    self.assertEqual(spec['config']['legend']['orient'], 'bottom')

  def test_selection_param_name(self):
    spec = _chart().add_params(charts.selection_param('abc')).to_dict()
    self.assertEqual(
        spec['params'], [{'name': charts.DESIGN_SELECTION, 'value': 'abc'}]
    )


if __name__ == '__main__':
  absltest.main()
