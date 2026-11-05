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

  def test_short_id(self):
    self.assertEqual(report_text.short_id('1234567890abcdef'), '12345678')

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
      self.assertIn(et, report_text.EXPERIMENT_TYPE_EXPLANATIONS)
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

  def test_format_percent(self):
    self.assertEqual(report_text.format_percent(0.051), '5.1%')
    self.assertEqual(report_text.format_percent(0.05, 0, signed=True), '+5%')
    self.assertEqual(report_text.format_percent(None), 'n/a')

  def test_format_interval(self):
    self.assertEqual(
        report_text.format_interval('1', '2', 0.9), '90% CI: 1 to 2'
    )


class AnalysisWordingTest(parameterized.TestCase):

  @parameterized.parameters(list(report_text.Verdict))
  def test_every_verdict_has_wording(self, verdict):
    wording = report_text.RESULT_WORDING[verdict]
    self.assertNotEmpty(wording.meaning)
    headline = report_text.headline(verdict, '+1,000', 'iCPA', '$5.00')
    self.assertIn('+1,000', headline)
    self.assertIn('$5.00', headline)
    self.assertNotEmpty(report_text.next_step(verdict, 'iCPA'))

  def test_yes_next_step_mentions_budget_and_mmm(self):
    step = report_text.next_step(report_text.Verdict.YES, 'iCPA')
    self.assertIn('budget planning', step)
    self.assertIn('MMM', step)

  def test_value_headline(self):
    headline = report_text.headline(
        report_text.Verdict.YES, '$9,000', 'iROAS', '2.1',
        response_is_value=True,
    )
    self.assertIn('extra conversion value', headline)

  @parameterized.parameters(
      (api.ExperimentType.HOLDBACK, 'running the campaign'),
      (api.ExperimentType.HEAVY_UP, 'increasing spend'),
      (api.ExperimentType.GO_DARK, 'pausing ads'),
      (None, 'different spend changes'),
  )
  def test_test_objective(self, experiment_type, phrase):
    text = report_text.test_objective(experiment_type, 10, 40, 30)
    self.assertIn(phrase, text)
    self.assertIn('10 treatment geos', text)
    self.assertIn('30-day test', text)


class ExecutionWordingTest(absltest.TestCase):

  def test_no_spend(self):
    wording = report_text.execution_wording(None, None, None)
    self.assertEqual(wording.tone, 'neutral')
    self.assertIn('not checked', wording.title)

  def test_observed_only_is_neutral(self):
    wording = report_text.execution_wording(0.3, None, None)
    self.assertEqual(wording.title, 'Observed spend change')
    self.assertIn('not been verified', wording.sentence)
    self.assertEqual(wording.tone, 'neutral')

  def test_verified(self):
    wording = report_text.execution_wording(-0.98, -1.0, True)
    self.assertEqual(wording.title, 'Spend changed as planned')
    self.assertEqual(wording.tone, 'success')

  def test_differs(self):
    wording = report_text.execution_wording(-0.2, -1.0, False)
    self.assertEqual(wording.tone, 'warning')

  def test_issue_geos(self):
    wording = report_text.execution_wording(
        -0.8, -1.0, False, tuple(f'g{i}' for i in range(7))
    )
    self.assertEqual(wording.tone, 'error')
    self.assertIn('and 2 more', wording.sentence)

  def test_effect_split_sentence(self):
    self.assertIn(
        '7-day cooldown',
        report_text.effect_split_sentence('+100', '+20', 7, 'conversions'),
    )


if __name__ == '__main__':
  absltest.main()
