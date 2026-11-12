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
import markupsafe
from meridian_geox.visualizations import charts
from meridian_geox.visualizations.templates import formatter
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
    # Numbers the reports format, with or without an interval.
    for text in ('-1,000 [-1,658, -342]', '-5.00%', '-$1,200', '-€2.50'):
      self.assertEqual(formatter.without_formulas(text), text)
    for text in ('-1+1', '-cmd', '-1 [=A1, 2]'):
      self.assertEqual(formatter.without_formulas(text), "'" + text)

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

  def test_download_button_has_filename(self):
    self.assertEqual(
        formatter.Button('CSV', 'a,b').filename, 'download.csv'
    )


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
    banner = formatter.CustomBlockSpec(
        'option_banner.html.jinja',
        {'warn': True, 'name': 'Option 1', 'narrative': 'Detects 1.3%.',
         'learn_more_url': 'https://example.com/doc'},
        candidate='d1',
    )
    tiles = formatter.StatsSectionSpec([
        formatter.StatsSpec('Feasibility', 'High chance', detail='99% chance',
                            tone='success'),
        formatter.StatsSpec('MDE', '1.3%', ci='90% CI 1.1% to 1.5%'),
    ], candidate='d1')
    analyst = formatter.StatsSectionSpec(
        [formatter.StatsSpec('Analyst only', 'x')], report_view='full_only'
    )
    note = formatter.CalloutSpec('Inspecting Option 2', candidate='d2')
    card = formatter.create_card_html(
        env, formatter.CardSpec('recommended', 'Recommended design'),
        [banner, None, tiles, note, analyst],
    )
    for text in ('Option 1', 'Detects 1.3%.', 'href="https://example.com/doc"',
                 '<stats class="geox-tone-success">', 'Feasibility',
                 'High chance', '99% chance', 'Inspecting Option 2',
                 'data-cand="d1"', 'class="geox-banner geox-banner-warn"',
                 '<stats-ci>90% CI 1.1% to 1.5%</stats-ci>',
                 'data-report-view="full_only"'):
      self.assertIn(text, card)
    page = formatter.create_summary_html(
        env, 'GeoX experiment design', [card],
        chips=[formatter.ChipSpec('Region', 'US')],
    )
    self.assertIn('<card id="recommended"', page.split('<cards>', 1)[1])
    self.assertIn('<span>Region:</span> US', page)

  def test_escapes_user_text(self):
    env = formatter.create_template_env()
    tiles = formatter.StatsSectionSpec([
        formatter.StatsSpec('<script>alert(1)</script>', 'A & B',
                            detail='"quoted"'),
    ])
    card = formatter.create_card_html(
        env, formatter.CardSpec('c', 'Title <b>'), [tiles]
    )
    page = formatter.create_summary_html(
        env, 'Design <report>', [card], subtitle='"sub" & more'
    )
    self.assertIn('&lt;script&gt;alert(1)&lt;/script&gt;', page)
    self.assertNotIn('<script>alert(1)</script>', page)
    self.assertIn('A &amp; B', page)
    self.assertIn('&#34;quoted&#34;', page)
    self.assertIn('Title &lt;b&gt;', page)
    self.assertIn('Design &lt;report&gt;', page)
    self.assertIn('&#34;sub&#34; &amp; more', page)

  def test_doc_link_drops_non_http_urls(self):
    env = formatter.create_template_env()
    banner = formatter.CustomBlockSpec(
        'option_banner.html.jinja',
        {'warn': False, 'name': 'Option 1', 'narrative': 'x',
         'learn_more_url': 'javascript:alert(1)'},
    )
    card = formatter.create_card_html(
        env, formatter.CardSpec('c', 'C'), [banner]
    )
    self.assertNotIn('javascript:', card)
    self.assertNotIn('Learn more', card)


class SpecTest(absltest.TestCase):

  def test_defaults(self):
    callout = formatter.CalloutSpec(
        'x', links=[('Docs', 'https://example.com')]
    )
    self.assertEqual(callout.tone, 'info')
    self.assertEqual(callout.report_view, 'all_views')
    self.assertIsNone(callout.candidate)
    self.assertEqual(list(callout.links), [('Docs', 'https://example.com')])
    table = formatter.TableSpec('t', 'T', ['a'], [['1']])
    self.assertFalse(table.selectable)

  def test_button_href_round_trips_payload(self):
    button = formatter.Button('CSV', 'a,b\n1,2', 'x.csv')
    prefix = 'data:text/csv;charset=utf-8;base64,'
    self.assertTrue(button.href.startswith(prefix))
    decoded = base64.b64decode(button.href[len(prefix):]).decode('utf-8')
    self.assertEqual(decoded, 'a,b\n1,2')


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

  def _html(self) -> str:
    env = formatter.create_template_env()
    card = formatter.create_card_html(
        env,
        formatter.CardSpec('summary', 'Summary <title>', cue='Select one'),
        [
            formatter.TextSpec('<script>alert(1)</script>'),
            formatter.StatsSectionSpec(
                [formatter.StatsSpec('Lift', '+5%', ci='90% CI: 1% to 9%')]
            ),
            formatter.CalloutSpec('Careful', title='Note', tone='warning'),
            formatter.TableSpec(
                '',
                '',
                ['Option', 'MDE'],
                [['Option 1', '2%'], ['Option 2', '3%']],
                row_ids=['a', 'b'],
                selectable=True,
            ),
            _chart_spec(reading_guide='Guide', candidate='a'),
            None,
            formatter.ButtonsSpec([
                formatter.Button('Download', 'a,b\n1,2\n', 'g.csv'),
            ]),
            formatter.TextSpec('Analyst only', report_view='full_only'),
            formatter.RowSpec(
                [formatter.TextSpec('left'), formatter.TextSpec('right')]
            ),
        ],
    )
    return formatter.create_summary_html(
        env,
        'Design <report>',
        [card],
        kind='design',
        chips=[formatter.ChipSpec('Cells', '1')],
        default_candidate='a',
    )

  def test_structure(self):
    html = self._html()
    self.assertIn('<card id="summary" data-report-view="all_views">', html)
    self.assertIn('data-geox-view="executive"', html)
    self.assertIn('data-report-view="full_only"', html)
    self.assertIn('data-design-id="a"', html)
    self.assertIn('data-cand="a" class="geox-chart-frame geox-cand"', html)
    self.assertIn('<stats-section', html)
    self.assertIn('<chips>', html)
    self.assertIn('download="g.csv"', html)
    self.assertIn('Save Executive PDF', html)
    self.assertIn('Select one', html)
    self.assertIn('How to read', html)
    specs = _extract_specs(html)
    self.assertEqual(list(specs), ['geox-chart-1'])
    self.assertIn('$schema', specs['geox-chart-1'])
    self.assertEqual(specs['geox-chart-1']['config']['font'], charts.FONT)

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

  def test_iframe_escapes_markup_input(self):
    frame = formatter.iframe(markupsafe.Markup('<div class="x">'), 600)
    self.assertIn('srcdoc="&lt;div class=&#34;x&#34;&gt;"', frame)

  def test_write_html_creates_nested_directories(self):
    root = self.create_tempdir().full_path
    path = formatter.write_html('x', 'r.html', os.path.join(root, 'a', 'b'))
    self.assertTrue(os.path.exists(path))

  def test_undefined_template_variables_raise(self):
    env = formatter.create_template_env()
    with self.assertRaises(jinja2.UndefinedError):
      env.from_string('{{ missing }}').render()


class SpecValidationTest(absltest.TestCase):

  def test_callout_links_are_stored_as_tuple(self):
    links = [('Docs', 'https://example.com')]
    spec = formatter.CalloutSpec('x', links=links)
    links.append(('More', 'https://example.org'))
    self.assertEqual(spec.links, (('Docs', 'https://example.com'),))

  def test_template_params_reject_reserved_keys(self):
    for key in ('report_view', 'candidate'):
      with self.assertRaisesRegex(ValueError, key):
        formatter.CustomBlockSpec('x.html.jinja', {key: 'y'})
    formatter.CustomBlockSpec('x.html.jinja', {'name': 'y'})


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


class NumberTest(absltest.TestCase):

  def test_numbers_must_match_whole_cell(self):
    self.assertEqual(formatter.without_formulas('-1.5e3'), '-1.5e3')
    self.assertEqual(formatter.without_formulas('-1\n'), "'-1\n")


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


class TextSpecTest(absltest.TestCase):

  def test_text_and_detail_note(self):
    env = formatter.create_template_env()
    card = formatter.create_card_html(env, formatter.CardSpec('c', 'C'), [
        formatter.TextSpec('Plain.'),
        formatter.TextSpec(
            'Note.', detail=True, links=(('Docs', 'https://example.com/d'),)
        ),
    ])
    self.assertIn('<p data-report-view="all_views">Plain.</p>', card)
    self.assertIn(
        '<p class="geox-detail">Note. <a href="https://example.com/d"', card
    )


if __name__ == '__main__':
  absltest.main()
