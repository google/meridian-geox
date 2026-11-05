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

import json
import os
import tempfile

from absl.testing import absltest
import jinja2
from meridian_geox.templates import formatter


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


class CreateSummaryHtmlTest(absltest.TestCase):

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


if __name__ == '__main__':
  absltest.main()
