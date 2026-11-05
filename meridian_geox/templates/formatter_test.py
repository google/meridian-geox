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

"""Tests for the GeoX report formatter."""

import base64
import json
import os
import re
import tempfile
from typing import Any

from absl.testing import absltest
import altair as alt
from meridian_geox import charts
from meridian_geox.templates import formatter
import pandas as pd


def _simple_chart() -> alt.Chart:
  return (
      alt.Chart(pd.DataFrame({'x': [1, 2], 'y': [3, 4]}))
      .mark_line()
      .encode(x='x:Q', y='y:Q')
  )


def _extract_specs(html: str) -> dict[str, Any]:
  match = re.search(
      r'<script type="application/json" id="geox-specs">(.*?)</script>',
      html,
      re.S,
  )
  assert match is not None
  return json.loads(match.group(1))


class EscapeTest(absltest.TestCase):

  def test_escape_html(self):
    self.assertEqual(
        formatter.escape('<b>"x" & y</b>'),
        '&lt;b&gt;&#34;x&#34; &amp; y&lt;/b&gt;',
    )

  def test_escape_none(self):
    self.assertEqual(formatter.escape(None), '')


class CsvTest(absltest.TestCase):

  def test_without_formulas(self):
    self.assertEqual(formatter.without_formulas('=SUM(A1)'), "'=SUM(A1)")
    self.assertEqual(formatter.without_formulas('+1'), "'+1")
    # Numbers stay numeric, including negative ones.
    self.assertEqual(formatter.without_formulas('-1'), '-1')
    self.assertEqual(formatter.without_formulas(-0.05), '-0.05')
    self.assertEqual(formatter.without_formulas('-1+A1'), "'-1+A1")
    self.assertEqual(formatter.without_formulas('@cmd'), "'@cmd")
    self.assertEqual(formatter.without_formulas('\tx'), "'\tx")
    self.assertEqual(formatter.without_formulas('Berlin'), 'Berlin')
    self.assertEqual(formatter.without_formulas(None), '')

  def test_to_csv(self):
    csv_text = formatter.to_csv(
        [{'location': '501', 'geo_name': '=HYPERLINK("x")'}],
        ['location', 'geo_name'],
    )
    self.assertEqual(
        csv_text, 'location,geo_name\n501,"\'=HYPERLINK(""x"")"\n'
    )


class JsonForScriptTest(absltest.TestCase):

  def test_escapes_script_breakers(self):
    text = formatter.json_for_script({'a': '</script><b>&'})
    self.assertNotIn('<', text)
    self.assertNotIn('>', text)
    self.assertEqual(json.loads(text), {'a': '</script><b>&'})

  def test_strips_bom(self):
    self.assertEqual(formatter.strip_bom('\ufeffabc'), 'abc')
    self.assertEqual(formatter.strip_bom('abc'), 'abc')


class SafetyTest(absltest.TestCase):

  def test_json_replaces_nan_and_inf(self):
    text = formatter.json_for_script({'a': float('nan'), 'b': [float('inf')]})
    self.assertEqual(json.loads(text), {'a': None, 'b': [None]})

  def test_callout_rejects_non_http_links(self):
    with self.assertRaises(ValueError):
      formatter.Callout('x', links=[('Bad', 'javascript:alert(1)')])

  def test_download_button_has_filename(self):
    self.assertEqual(
        formatter.Button('CSV', 'download', 'a,b').filename, 'download.csv'
    )

  def test_only_url_data_uris_are_hoisted(self):
    store = formatter.ResourceStore()
    uri = 'data:image/png;base64,' + 'A' * formatter.HOIST_MIN_CHARS
    spec = store.hoist({'url': uri, 'values': [{'img': uri}]})
    self.assertTrue(spec['url'].startswith(formatter.RESOURCE_PREFIX))
    self.assertEqual(spec['values'][0]['img'], uri)


class ResourceStoreTest(absltest.TestCase):

  def test_hoists_large_data_uris_once(self):
    payload = base64.b64encode(b'x' * 2000).decode()
    uri = 'data:application/json;base64,' + payload
    store = formatter.ResourceStore()
    hoisted = store.hoist({'a': {'url': uri}, 'b': [{'url': uri}]})
    self.assertLen(store.resources, 1)
    pointer = hoisted['a']['url']
    self.assertTrue(pointer.startswith(formatter.RESOURCE_PREFIX))
    self.assertEqual(hoisted['b'][0]['url'], pointer)

  def test_keeps_small_values(self):
    store = formatter.ResourceStore()
    self.assertEqual(store.hoist({'url': 'data:,{}'}), {'url': 'data:,{}'})
    self.assertEmpty(store.resources)


class BlockTest(absltest.TestCase):

  def test_kinds_and_defaults(self):
    self.assertEqual(formatter.Text('x').kind, 'text')
    callout = formatter.Callout('x', links=[('Docs', 'https://example.com')])
    self.assertEqual(callout.kind, 'callout')
    self.assertEqual(callout.tone, 'info')
    self.assertEqual(callout.audience, 'all')
    self.assertEqual(list(callout.links), [('Docs', 'https://example.com')])

  def test_tiles_and_table_kinds(self):
    tiles = formatter.Tiles([formatter.Tile('MDE', '1.4%')])
    self.assertEqual(tiles.kind, 'tiles')
    table = formatter.Table(columns=['a'], rows=[['1']])
    self.assertEqual(table.kind, 'table')
    self.assertFalse(table.selectable)

  def test_button_href_round_trips_payload(self):
    button = formatter.Button('CSV', 'download', 'a,b\n1,2', 'x.csv')
    prefix = 'data:text/csv;charset=utf-8;base64,'
    self.assertTrue(button.href.startswith(prefix))
    decoded = base64.b64decode(button.href[len(prefix):]).decode('utf-8')
    self.assertEqual(decoded, 'a,b\n1,2')


class ChartToDictTest(absltest.TestCase):

  def test_dict_passes_through(self):
    spec = {'mark': 'point'}
    self.assertIs(formatter.chart_to_dict(spec), spec)

  def test_applies_shared_style(self):
    chart = (
        alt.Chart(pd.DataFrame({'x': [1, 2], 'y': [3, 4]}))
        .mark_line()
        .encode(x='x:Q', y='y:Q')
    )
    spec = formatter.chart_to_dict(chart)
    self.assertEqual(spec['config']['font'], charts.FONT)


class RenderReportTest(absltest.TestCase):

  def _report(self, **kwargs) -> formatter.Report:
    cards = [
        formatter.Card(
            card_id='summary',
            title='Summary <title>',
            cue='Select a candidate',
            blocks=[
                formatter.Text('<script>alert(1)</script>'),
                formatter.Tiles(
                    [formatter.Tile('Lift', '+5%', ci='90% CI: 1% to 9%')]
                ),
                formatter.Callout('Careful', title='Note', tone='warning'),
                formatter.Table(
                    columns=['Option', 'MDE'],
                    rows=[['Option 1', '2%'], ['Option 2', '3%']],
                    row_ids=['a', 'b'],
                    selectable=True,
                ),
                formatter.Chart(
                    _simple_chart(),
                    framing=charts.Framing('Title', reading_guide='Guide'),
                    candidate='a',
                ),
                formatter.Buttons([
                    formatter.Button(
                        'Download', 'download', 'a,b\n1,2\n', 'geos.csv'
                    ),
                    formatter.Button('Copy ID', 'copy', 'abc-123'),
                ]),
                formatter.Text('Analyst only', audience='technical'),
                formatter.Row(
                    [formatter.Text('left'), formatter.Text('right')]
                ),
                formatter.Html('<em>trusted</em>'),
            ],
        )
    ]
    return formatter.Report(
        title='Design <report>',
        kind='design',
        cards=cards,
        chips=[formatter.Chip('Cells', '1')],
        default_candidate='a',
        **kwargs,
    )

  def test_render_escapes_user_text(self):
    html = formatter.render_report(self._report())
    self.assertIn('&lt;script&gt;alert(1)&lt;/script&gt;', html)
    self.assertNotIn('<script>alert(1)</script>', html)
    self.assertIn('Design &lt;report&gt;', html)
    self.assertIn('<em>trusted</em>', html)

  def test_render_structure(self):
    html = formatter.render_report(self._report())
    self.assertIn('data-geox-view="executive"', html)
    self.assertIn('data-audience="technical"', html)
    self.assertIn('data-design-id="a"', html)
    self.assertIn('class="geox-cand" data-cand="a"', html)
    self.assertIn('download="geos.csv"', html)
    self.assertIn('data-copy="abc-123"', html)
    self.assertIn('Save Executive PDF', html)
    self.assertIn('Select a candidate', html)
    self.assertIn('How to read', html)
    specs = _extract_specs(html)
    self.assertEqual(list(specs), ['geox-chart-1'])
    self.assertIn('$schema', specs['geox-chart-1'])

  def test_render_hoists_resources(self):
    uri = 'data:application/json;base64,' + base64.b64encode(
        json.dumps({'type': 'FeatureCollection', 'pad': 'x' * 2000}).encode()
    ).decode()
    chart = (
        alt.Chart(alt.Data(url=uri, format=alt.DataFormat(type='json')))
        .mark_geoshape()
    )
    report = formatter.Report(
        title='Maps',
        kind='analysis',
        cards=[
            formatter.Card(
                'maps',
                'Maps',
                [formatter.Chart(chart), formatter.Chart(chart)],
            )
        ],
    )
    html = formatter.render_report(report)
    self.assertEqual(html.count('id="geox-resource-'), 1)
    self.assertEqual(html.count(uri), 1)
    specs = _extract_specs(html)
    self.assertTrue(
        specs['geox-chart-1']['data']['url'].startswith(
            formatter.RESOURCE_PREFIX
        )
    )

  def test_write_html(self):
    tmp = self.enter_context(tempfile.TemporaryDirectory())
    path = formatter.write_html('<html></html>', 'out.html', tmp)
    self.assertEqual(path, os.path.join(tmp, 'out.html'))
    with open(path) as f:
      self.assertEqual(f.read(), '<html></html>')

  def test_write_html_requires_filename(self):
    with self.assertRaises(ValueError):
      formatter.write_html('x', '', self.enter_context(tempfile.TemporaryDirectory()))

  def test_iframe_escapes_page(self):
    frame = formatter.iframe('<html><body data-x="1">"hi"</body></html>', 600)
    self.assertTrue(frame.startswith('<iframe srcdoc="&lt;html&gt;'))
    self.assertIn('&#34;hi&#34;', frame)
    self.assertIn('height:600px', frame)
    self.assertIn('allow-scripts', frame)
    self.assertNotIn('allow-same-origin', frame)

  def test_iframe_rejects_bad_height(self):
    with self.assertRaises(ValueError):
      formatter.iframe('<html></html>', 0)

  def test_takeaway_rendered(self):
    report = formatter.Report(
        title='T',
        kind='design',
        cards=[
            formatter.Card(
                'c',
                'C',
                [
                    formatter.Chart(
                        _simple_chart(),
                        framing=charts.Framing('A', takeaway='Big <win>'),
                    )
                ],
            )
        ],
    )
    self.assertIn('→ Big &lt;win&gt;', formatter.render_report(report))


if __name__ == '__main__':
  absltest.main()
