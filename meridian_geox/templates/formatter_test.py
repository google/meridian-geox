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
from meridian_geox.templates import formatter


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

  def test_button_href_round_trips_payload(self):
    button = formatter.Button('CSV', 'a,b\n1,2', 'x.csv')
    prefix = 'data:text/csv;charset=utf-8;base64,'
    self.assertTrue(button.href.startswith(prefix))
    decoded = base64.b64decode(button.href[len(prefix):]).decode('utf-8')
    self.assertEqual(decoded, 'a,b\n1,2')


if __name__ == '__main__':
  absltest.main()
