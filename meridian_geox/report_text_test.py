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
import types

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

  def test_plural_and_names(self):
    self.assertEqual(report_text.plural(1, 'geo'), '1 geo')
    self.assertEqual(report_text.plural(3, 'geo'), '3 geos')
    self.assertEqual(
        report_text.name_list(['a', 'b', 'c', 'd', 'e']), 'a, b, c and 2 more'
    )
    self.assertEqual(report_text.name_list(['a']), 'a')
    self.assertEqual(report_text.name_list([]), '')

  def test_short_money(self):
    self.assertEqual(report_text.short_money(8683, '$'), '$8.7k')
    self.assertEqual(report_text.short_money(1_200_000, '€'), '€1.2M')
    self.assertEqual(report_text.short_money(-950), '-950')
    self.assertEqual(report_text.short_money(None), 'n/a')
    self.assertEqual(report_text.short_money(-8700, '$'), '-$8.7k')
    self.assertEqual(report_text.short_money(float('inf'), '$'), '∞')
    self.assertEqual(report_text.short_money(float('-inf'), '$'), '-∞')

  def test_dates(self):
    self.assertEqual(
        report_text.short_date(datetime.date(2020, 4, 1)), 'Wed, Apr 1, 2020'
    )
    self.assertEqual(
        report_text.date_range(
            datetime.date(2020, 1, 1), datetime.date(2020, 3, 31)
        ),
        'Jan 1 – Mar 31, 2020',
    )

  def test_every_type_has_copy(self):
    for et in api.ExperimentType:
      self.assertTrue(report_text.spend_terms(et).explanation)
      self.assertIn(et, report_text.SPEND_DEFINITIONS)
      self.assertIn(et, report_text.GROUP_DESCRIPTIONS)


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

  def test_format_signed(self):
    self.assertEqual(report_text.format_signed(5), '+5')
    self.assertEqual(report_text.format_signed(-5), '-5')

  def test_format_money(self):
    self.assertEqual(report_text.format_money(1234, '$'), '$1,234')
    self.assertEqual(report_text.format_money(-2.5, '€', 2), '-€2.50')
    self.assertEqual(report_text.format_money(float('-inf'), '$'), '-∞')
    self.assertEqual(report_text.format_money(float('inf'), '$'), '∞')
    self.assertEqual(report_text.format_money(None, '$'), 'n/a')
    self.assertEqual(report_text.format_money(float('nan'), '$'), 'n/a')

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


class SpendTileTest(absltest.TestCase):

  def test_spend_tile(self):
    text = report_text.DesignText('$')
    go_dark = [('Treatment', api.ExperimentType.GO_DARK, -5000.0, -0.5)]
    self.assertEqual(
        text.spend_tile(go_dark, 5000.0, None, 10_000.0, 30, None),
        ('$5.0k', '50% of treatment BAU spend',
         '= $10.0k BAU spend in treatment geos over 30 days × 50%'),
    )
    heavy_up = [('Treatment', api.ExperimentType.HEAVY_UP, 2000.0, 0.25)]
    self.assertEqual(
        text.spend_tile(heavy_up, 2000.0, None, None, 30, None),
        ('+$2.0k', '25% of treatment BAU spend', None),
    )
    holdback = [('Treatment', api.ExperimentType.HOLDBACK, 8000.0, None)]
    self.assertEqual(
        text.spend_tile(holdback, 8000.0, 6000.0, None, 30, (100.0, 2.0)),
        ('$8.0k', '$2.0k over your $6.0k goal',
         '= 100 incremental conversions (MDE) × $2.00 iCPA'),
    )
    self.assertEqual(report_text.change_percent(None), '')
    self.assertEqual(report_text.change_percent(-0.5, signed=True), '-50%')


def _schedule(dated, ramp_up_days=0, cooldown_days=7):
  day = datetime.date
  return types.SimpleNamespace(
      dated=dated, test_start=day(2020, 4, 1), test_end=day(2020, 4, 28),
      cooldown_start=day(2020, 4, 29), cooldown_end=day(2020, 5, 5),
      test_days=28, ramp_up_days=ramp_up_days, cooldown_days=cooldown_days,
  )


class ScheduleTextTest(parameterized.TestCase):

  @parameterized.parameters(
      (False, 0, 0, True, 'for 28 days'),
      (False, 0, 7, True, 'for 28 days, then a 7-day cooldown'),
      (True, 3, 0, False, 'from Wed, Apr 1 to Tue, Apr 28, 2020 (28 days)'),
      (
          True, 3, 7, True,
          (
              'from Wed, Apr 1 to Tue, Apr 28, 2020 (28 days, the first 3 a'
              ' ramp-up), then a 7-day cooldown to Tue, May 5, 2020'
          ),
      ),
  )
  def test_schedule_span(self, dated, ramp, cooldown, ramp_up, expected):
    schedule = _schedule(dated, ramp, cooldown)
    self.assertEqual(report_text.schedule_span(schedule, ramp_up), expected)

  def test_calendar_takeaway(self):
    pretest = (datetime.date(2020, 1, 1), datetime.date(2020, 3, 31))
    self.assertEqual(
        report_text.calendar_takeaway(_schedule(True), pretest),
        'Test: Wed, Apr 1, 2020 – Tue, Apr 28, 2020 (28 days); cooldown: Wed,'
        ' Apr 29, 2020 – Tue, May 5, 2020 (7 days). Pretest data: Jan 1 – Mar'
        ' 31, 2020.',
    )
    self.assertIsNone(report_text.calendar_takeaway(_schedule(False), None))


if __name__ == '__main__':
  absltest.main()
