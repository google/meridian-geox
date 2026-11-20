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

"""Tests for GeoX report wording."""

import datetime

from absl.testing import absltest
from absl.testing import parameterized
from meridian_geox import api
from meridian_geox import report_text


class NamingTest(parameterized.TestCase):

  def test_option_name(self):
    self.assertEqual(report_text.option_name(1), 'Option 1 (Recommended)')
    self.assertEqual(report_text.option_name(3), 'Option 3')

  def test_option_name_rejects_zero(self):
    with self.assertRaises(ValueError):
      report_text.option_name(0)

  @parameterized.parameters(('cell_1', 1), ('cell_12', 12), ('weird', 1))
  def test_cell_index(self, cell, expected):
    self.assertEqual(report_text.cell_index(cell), expected)

  def test_treatment_label(self):
    self.assertEqual(
        report_text.treatment_label(
            'cell_1', api.ExperimentType.GO_DARK, multi_cell=True
        ),
        'Treatment 1 · ads paused',
    )
    self.assertEqual(
        report_text.treatment_label(
            'cell_2', api.ExperimentType.HEAVY_UP, multi_cell=True
        ),
        'Treatment 2 · spend increased',
    )
    self.assertEqual(
        report_text.treatment_label(
            'cell_1', api.ExperimentType.HOLDBACK, multi_cell=False
        ),
        'Treatment',
    )


class HelperTest(absltest.TestCase):

  def test_short_money(self):
    self.assertEqual(report_text.short_money(8683, '$'), '$8.7k')
    self.assertEqual(report_text.short_money(1_200_000, '€'), '€1.2M')
    self.assertEqual(report_text.short_money(-950), '-950')
    self.assertEqual(report_text.short_money(None), 'n/a')


class SpendTermsTest(parameterized.TestCase):

  @parameterized.parameters(
      (api.ExperimentType.HOLDBACK, 'Campaign spend'),
      (api.ExperimentType.HEAVY_UP, 'Extra spend'),
      (api.ExperimentType.GO_DARK, 'Paused spend'),
      (None, 'Incremental spend'),
  )
  def test_labels(self, experiment_type, label):
    self.assertEqual(report_text.spend_terms(experiment_type).label, label)


class FormatTest(absltest.TestCase):

  def test_format_number(self):
    self.assertEqual(report_text.format_number(1234.5), '1,234')
    self.assertEqual(report_text.format_number(1234.5, 1), '1,234.5')
    self.assertEqual(report_text.format_number(None), 'n/a')
    self.assertEqual(report_text.format_number(float('nan')), 'n/a')
    self.assertEqual(report_text.format_number(float('inf')), '∞')

  def test_format_money(self):
    self.assertEqual(report_text.format_money(1234, '$'), '$1,234')
    self.assertEqual(report_text.format_money(-2.5, '€', 2), '-€2.50')
    self.assertEqual(report_text.format_money(float('-inf'), '$'), '-∞')

  def test_format_percent(self):
    self.assertEqual(report_text.format_percent(0.051), '5.1%')
    self.assertEqual(report_text.format_percent(0.05, 0, signed=True), '+5%')
    self.assertEqual(report_text.format_percent(None), 'n/a')
    self.assertEqual(report_text.format_percent(float('nan')), 'n/a')

  def test_date_range(self):
    self.assertEqual(
        report_text.date_range(datetime.date(2020, 3, 2),
                               datetime.date(2020, 3, 31)),
        'Mar 2 – Mar 31, 2020',
    )
    self.assertEqual(
        report_text.date_range(datetime.date(2019, 12, 2),
                               datetime.date(2020, 1, 31)),
        'Dec 2, 2019 – Jan 31, 2020',
    )


if __name__ == '__main__':
  absltest.main()
