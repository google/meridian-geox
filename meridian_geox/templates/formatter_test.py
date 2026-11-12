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
import jinja2
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


class JsonForScriptTest(absltest.TestCase):

  def test_escapes_script_breakers(self):
    text = formatter.json_for_script({'a': '</script><b>&'})
    self.assertNotIn('<', text)
    self.assertNotIn('>', text)
    self.assertEqual(json.loads(text), {'a': '</script><b>&'})


class SafetyTest(absltest.TestCase):

  def test_json_replaces_nan_and_inf(self):
    text = formatter.json_for_script({'a': float('nan'), 'b': [float('inf')]})
    self.assertEqual(json.loads(text), {'a': None, 'b': [None]})

  def test_json_unwraps_scalars_and_tuples(self):

    class Scalar:  # Stands in for a NumPy scalar.

      def __init__(self, value):
        self.value = value

      def item(self):
        return self.value

    text = formatter.json_for_script({
        'n': Scalar(2),
        'nan': Scalar(float('nan')),
        't': (1.5, float('inf')),
    })
    self.assertEqual(json.loads(text), {'n': 2, 'nan': None, 't': [1.5, None]})

  def test_callout_rejects_non_http_links(self):
    with self.assertRaises(ValueError):
      formatter.CalloutSpec('x', links=[('Bad', 'javascript:alert(1)')])


class PageTest(absltest.TestCase):

  def _page(self):
    return formatter._page(formatter.create_template_env())

  def test_only_url_data_uris_are_hoisted(self):
    uri = 'data:image/png;base64,' + 'A' * formatter.HOIST_MIN_CHARS
    spec = self._page().hoist({'url': uri, 'values': [{'img': uri}]})
    self.assertTrue(spec['url'].startswith(formatter.RESOURCE_PREFIX))
    self.assertEqual(spec['values'][0]['img'], uri)

  def test_hoists_large_data_uris_once(self):
    payload = base64.b64encode(b'x' * 2000).decode()
    uri = 'data:application/json;base64,' + payload
    page = self._page()
    hoisted = page.hoist({'a': {'url': uri}, 'b': [{'url': uri}]})
    self.assertLen(page.resources, 1)
    pointer = hoisted['a']['url']
    self.assertTrue(pointer.startswith(formatter.RESOURCE_PREFIX))
    self.assertEqual(hoisted['b'][0]['url'], pointer)

  def test_keeps_small_values(self):
    page = self._page()
    self.assertEqual(page.hoist({'url': 'data:,{}'}), {'url': 'data:,{}'})
    self.assertEmpty(page.resources)

  def test_requires_created_environment(self):
    with self.assertRaises(ValueError):
      formatter.create_card_html(
          jinja2.Environment(), formatter.CardSpec('c', 'C'), []
      )

  def test_renders_card_blocks_into_the_page(self):
    env = formatter.create_template_env()
    banner = formatter.TemplateSpec(
        'option_banner.html.jinja',
        {'warn': False, 'name': 'Option 1', 'narrative': 'Detects 1.3%.',
         'learn_more_url': 'https://example.com/doc'},
        candidate='d1',
    )
    tiles = formatter.StatsSectionSpec([
        formatter.StatsSpec('Feasibility', 'High chance', detail='99% chance',
                            tone='success'),
        formatter.StatsSpec('MDE', '1.3%'),
    ], candidate='d1')
    note = formatter.CalloutSpec('Inspecting Option 2', candidate='d2')
    card = formatter.create_card_html(
        env, formatter.CardSpec('recommended', 'Recommended design'),
        [banner, None, tiles, note],
    )
    for text in ('Option 1', 'Detects 1.3%.', 'href="https://example.com/doc"',
                 '<stats class="geox-tone-success">', 'Feasibility',
                 'High chance', '99% chance', 'Inspecting Option 2',
                 'data-cand="d1"'):
      self.assertIn(text, card)
    page = formatter.create_summary_html(env, 'GeoX experiment design', [card])
    self.assertIn('<card id="recommended"', page.split('<cards>', 1)[1])

  def test_doc_link_drops_non_http_urls(self):
    env = formatter.create_template_env()
    banner = formatter.TemplateSpec(
        'option_banner.html.jinja',
        {'warn': False, 'name': 'Option 1', 'narrative': 'x',
         'learn_more_url': 'javascript:alert(1)'},
    )
    card = formatter.create_card_html(
        env, formatter.CardSpec('c', 'C'), [banner]
    )
    self.assertNotIn('javascript:', card)
    self.assertIn('Learn more', card)


class SpecTest(absltest.TestCase):

  def test_defaults(self):
    callout = formatter.CalloutSpec(
        'x', links=[('Docs', 'https://example.com')]
    )
    self.assertEqual(callout.tone, 'info')
    self.assertEqual(callout.audience, 'all')
    self.assertIsNone(callout.candidate)
    self.assertEqual(list(callout.links), [('Docs', 'https://example.com')])
    table = formatter.TableSpec('t', 'T', ['a'], [['1']])
    self.assertFalse(table.selectable)


def _chart_spec(chart=None, **kwargs) -> formatter.ChartSpec:
  return formatter.ChartSpec(
      '', charts.to_json(chart if chart is not None else _simple_chart()),
      **kwargs,
  )


def _render(specs) -> str:
  """Renders specs in one card on a page."""
  env = formatter.create_template_env()
  card = formatter.create_card_html(env, formatter.CardSpec('c', 'C'), specs)
  return formatter.create_summary_html(env, 'C', [card], kind='analysis')


class CreateSummaryHtmlTest(absltest.TestCase):

  def test_hoists_resources(self):
    uri = 'data:application/json;base64,' + base64.b64encode(
        json.dumps({'type': 'FeatureCollection', 'pad': 'x' * 2000}).encode()
    ).decode()
    chart = alt.Chart(
        alt.Data(url=uri, format=alt.DataFormat(type='json'))
    ).mark_geoshape()
    html = _render([_chart_spec(chart), _chart_spec(chart)])
    self.assertEqual(html.count('id="geox-resource-'), 1)
    self.assertEqual(html.count(uri), 1)
    specs = _extract_specs(html)
    self.assertTrue(
        specs['geox-chart-1']['data']['url'].startswith(
            formatter.RESOURCE_PREFIX
        )
    )

  def test_chart_ids(self):
    explicit = formatter.ChartSpec(
        'geox-chart-2', charts.to_json(_simple_chart())
    )
    html = _render([explicit, _chart_spec()])
    self.assertIn('id="geox-chart-2"', html)
    # Auto IDs skip the explicit one.
    self.assertEqual(
        list(_extract_specs(html)), ['geox-chart-2', 'geox-chart-3']
    )

  def test_duplicate_chart_id_raises(self):
    env = formatter.create_template_env()
    chart = formatter.ChartSpec('same', charts.to_json(_simple_chart()))
    with self.assertRaisesRegex(ValueError, 'Duplicate chart id'):
      formatter.create_card_html(
          env, formatter.CardSpec('c', 'C'), [chart, chart]
      )

  def test_collapsed_table(self):
    html = _render([formatter.TableSpec(
        '', 'Geos', ['a'], [['1']], collapsed_summary='Show all geos'
    )])
    self.assertIn('<summary>Show all geos</summary>', html)

  def test_findings_and_description(self):
    html = _render(
        [_chart_spec(description='Big <win>', warnings=['Check <this>'])]
    )
    self.assertIn('→ Big &lt;win&gt;', html)
    self.assertIn('<warnings>', html)
    self.assertIn('Check &lt;this&gt;', html)

  def test_write_html(self):
    tmp = self.enter_context(tempfile.TemporaryDirectory())
    path = formatter.write_html('<html></html>', 'out.html', tmp)
    self.assertEqual(path, os.path.join(tmp, 'out.html'))
    with open(path) as f:
      self.assertEqual(f.read(), '<html></html>')
    # An empty filepath writes to the current directory.
    self.addCleanup(os.chdir, os.getcwd())
    os.chdir(tmp)
    self.assertEqual(
        formatter.write_html('x', 'cwd.html', ''), os.path.join('.', 'cwd.html')
    )
    self.assertTrue(os.path.exists(os.path.join(tmp, 'cwd.html')))
    with self.assertRaises(ValueError):
      formatter.write_html('x', '', tmp)

  def test_template_env_trims_block_whitespace(self):
    env = formatter.create_template_env()
    self.assertTrue(env.trim_blocks)
    self.assertTrue(env.lstrip_blocks)
    template = env.from_string(
        '<ul>\n  {% for x in xs %}\n<li>{{ x }}</li>\n  {% endfor %}\n</ul>'
    )
    self.assertEqual(
        template.render(xs=[1, 2]), '<ul>\n<li>1</li>\n<li>2</li>\n</ul>'
    )

  def test_iframe_escapes_page(self):
    frame = formatter.iframe('<html><body data-x="1">"hi"</body></html>', 600)
    self.assertTrue(frame.startswith('<iframe srcdoc="&lt;html&gt;'))
    self.assertIn('&#34;hi&#34;', frame)
    self.assertIn('height:600px', frame)
    self.assertIn('allow-scripts', frame)
    self.assertNotIn('allow-same-origin', frame)
    with self.assertRaises(ValueError):
      formatter.iframe('<html></html>', 0)


class SpecValidationTest(absltest.TestCase):

  def test_callout_links_are_stored_as_tuple(self):
    links = [('Docs', 'https://example.com')]
    spec = formatter.CalloutSpec('x', links=links)
    links.append(('More', 'https://example.org'))
    self.assertEqual(spec.links, (('Docs', 'https://example.com'),))

  def test_template_params_reject_reserved_keys(self):
    for key in ('audience', 'candidate'):
      with self.assertRaisesRegex(ValueError, key):
        formatter.TemplateSpec('x.html.jinja', {key: 'y'})
    formatter.TemplateSpec('x.html.jinja', {'name': 'y'})


class TableSpecTest(absltest.TestCase):

  def _table(self, **kwargs):
    return formatter.TableSpec('t', 'T', ['a'], [['1'], ['2']], **kwargs)

  def test_valid(self):
    self._table(row_ids=['x', 'y'], selectable=True, row_tones=['info'] * 2)

  def test_row_lengths_must_match(self):
    with self.assertRaisesRegex(ValueError, 'row_ids'):
      self._table(row_ids=['x'])
    with self.assertRaisesRegex(ValueError, 'row_tones'):
      self._table(row_tones=['info'])

  def test_selectable_needs_row_ids(self):
    with self.assertRaisesRegex(ValueError, 'row_ids'):
      self._table(selectable=True)


class DefaultCandidateTest(absltest.TestCase):

  def test_default_candidate_is_selected_on_load(self):
    env = formatter.create_template_env()
    self.assertIn(
        'var selected = "";', formatter.create_summary_html(env, 'T', [])
    )
    html = formatter.create_summary_html(
        env, 'T', [], default_candidate='cand1'
    )
    self.assertIn('var selected = "cand1";', html)


if __name__ == '__main__':
  absltest.main()
