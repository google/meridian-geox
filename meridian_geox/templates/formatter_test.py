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

from absl.testing import absltest
import altair as alt
from meridian_geox import charts
from meridian_geox.templates import formatter
import pandas as pd


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


if __name__ == '__main__':
  absltest.main()
