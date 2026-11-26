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
import functools
import types

from absl.testing import absltest
from absl.testing import parameterized
from meridian_geox import api
from meridian_geox.visualizations import report_text


class NamingTest(parameterized.TestCase):

  def test_option_name(self):
    self.assertEqual(report_text.option_name(1), 'Option 1 (Recommended)')
    self.assertEqual(report_text.option_name(3), 'Option 3')

  def test_option_name_rejects_zero(self):
    with self.assertRaises(ValueError):
      report_text.option_name(0)

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


class HelperTest(parameterized.TestCase):

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
    self.assertEqual(report_text.short_money(999_950, '$'), '$1.0M')
    self.assertEqual(report_text.short_money(999.6), '1.0k')
    self.assertEqual(report_text.short_money(2.5e9, '$'), '$2.5B')
    self.assertEqual(report_text.short_money(951, '$'), '$951')
    self.assertEqual(report_text.short_money(951_000, '$'), '$951.0k')
    self.assertEqual(report_text.short_money(999.5, '$'), '$1.0k')
    self.assertEqual(report_text.short_money(951_000_000, '$'), '$951.0M')
    self.assertEqual(report_text.short_money(-951, '$'), '-$951')
    self.assertEqual(report_text.short_money(-951_000, '$'), '-$951.0k')
    self.assertEqual(report_text.short_money(-999.5, '$'), '-$1.0k')
    self.assertEqual(report_text.short_money(float('nan'), '$'), 'n/a')

  @parameterized.named_parameters(
      ('normal', 0.934, 0.8, 'R² 0.93; minimum 0.8'),
      ('just_below', 0.796, 0.8, 'R² 0.796; minimum 0.8'),
      ('closer', 0.7996, 0.8, 'R² 0.7996; minimum 0.8'),
      ('max_digits', 0.79999, 0.8, 'R² 0.8000; minimum 0.8'),
      ('nan', float('nan'), 0.8, 'R² n/a; minimum 0.8'),
  )
  def test_fit_detail(self, r2, minimum, expected):
    self.assertEqual(report_text.fit_detail(r2, minimum), expected)

  def test_aa_detail(self):
    self.assertEqual(
        report_text.aa_detail(0.0996, 0.1), 'AA p-value 0.0996; passes at ≥ 0.1'
    )
    self.assertEqual(
        report_text.aa_detail(0.42, 0.1), 'AA p-value 0.42; passes at ≥ 0.1'
    )

  @parameterized.named_parameters(
      ('below_high', 0.796, '79.6%'),
      ('below_medium', 0.496, '49.6%'),
      ('nan', float('nan'), 'n/a'),
      ('high', 0.8, '80%'),
      ('medium', 0.5, '50%'),
      ('normal', 0.64, '64%'),
  )
  def test_chance_text(self, chance, expected):
    self.assertEqual(report_text.chance_text(chance), expected)

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

  def test_ci_line(self):
    self.assertEqual(
        report_text.CI_LINE.format(level=0.9, lower='1', upper='2'),
        '90% CI: 1 to 2',
    )


class AnalysisWordingTest(parameterized.TestCase):

  @parameterized.parameters(list(report_text.Verdict))
  def test_every_verdict_has_wording(self, verdict):
    self.assertNotEmpty(report_text.RESULT_TONE[verdict])
    headline = report_text.AnalysisText(currency_symbol='$').headline(
        verdict, 1000, 5.0, False
    )
    self.assertIn('+1,000', headline)
    self.assertIn('$5.00', headline)
    self.assertNotEmpty(report_text.NEXT_STEP[verdict].format('iCPA'))

  def test_yes_next_step_and_value_headline(self):
    step = report_text.NEXT_STEP[report_text.Verdict.YES].format('iCPA')
    self.assertIn('budget planning', step)
    self.assertIn('MMM', step)
    text = report_text.AnalysisText(True, '$')
    headline = text.headline(report_text.Verdict.YES, 9000, 2.1, False)
    self.assertIn('extra conversion value', headline)
    self.assertIn('2.10 iROAS', headline)
    self.assertEqual(text.efficiency_metric[:5], 'iROAS')

  @parameterized.parameters(
      (api.ExperimentType.HOLDBACK, 'running the campaign'),
      (api.ExperimentType.HEAVY_UP, 'increasing spend'),
      (api.ExperimentType.GO_DARK, 'pausing ads'),
  )
  def test_test_objective(self, experiment_type, phrase):
    text = report_text.test_objective([('T', experiment_type, 10)], 40, 30)
    self.assertIn(phrase, text)
    self.assertIn('10 treatment geos', text)
    self.assertIn('30-day test', text)
    multi = report_text.test_objective(
        [('T1', experiment_type, 5), ('T2', experiment_type, 5)], 40, 30
    )
    self.assertIn(f'{phrase} in T1 (5 geos) and {phrase} in T2', multi)
    self.assertIn('40 shared control geos', multi)


class ExecutionSentenceTest(parameterized.TestCase):

  @parameterized.parameters(
      (False, None, None, None, (), 'No spend data was passed'),
      (True, 0.3, None, None, (), 'not been verified'),
      (True, -0.98, -1.0, True, (), 'in line with the planned change (-100%)'),
      (True, -0.2, -1.0, False, (), 'differs from the planned change'),
      (True, -0.8, -1.0, False, tuple(f'g{i}' for i in range(7)),
       'and 2 more'),
  )
  def test_execution_sentence(
      self, has_spend, observed, planned, verified, issues, phrase
  ):
    sentence = report_text.AnalysisText().execution_sentence(
        api.ExperimentType.HOLDBACK, has_spend, observed, planned, verified,
        issues, None,
    )
    self.assertIn(phrase, sentence)


_HOLDBACK, _GO_DARK, _HEAVY_UP = (
    api.ExperimentType.HOLDBACK,
    api.ExperimentType.GO_DARK,
    api.ExperimentType.HEAVY_UP,
)


class DesignTextTest(parameterized.TestCase):

  def test_spend_tile(self):
    text = report_text.DesignText('$')
    tile = functools.partial(
        text.spend_tile, budget=None, meets_budget=None, bau=None, days=30,
        mde_spend=None,
    )
    self.assertEqual(
        tile(changes=[('Treatment', _GO_DARK, -5000.0, -0.5)],
             total_spend=5000.0, bau=10_000.0),
        ('$5.0k', '50% of treatment BAU spend',
         '= $10.0k BAU spend in treatment geos over 30 days × 50%'),
    )
    self.assertEqual(
        tile(changes=[('Treatment', _HEAVY_UP, 2000.0, 0.25)],
             total_spend=2000.0),
        ('+$2.0k', '25% of treatment BAU spend', None),
    )
    holdback = [('Treatment', _HOLDBACK, 8000.0, None)]
    over = tile(changes=holdback, total_spend=8000.0, budget=6000.0,
                meets_budget=False, mde_spend=(100.0, 2.0))
    self.assertEqual(
        over,
        ('$8.0k', '$2.0k over your $6.0k goal',
         '= 100 incremental conversions (MDE) × $2.00 iCPA'),
    )
    self.assertEqual(over.detail, '$2.0k over your $6.0k goal')
    within = tile(changes=holdback, total_spend=8000.0, budget=9000.0,
                  meets_budget=True)
    self.assertEqual(within.detail, 'within your $9.0k goal')
    # Only holdback spend counts against the budget goal.
    mixed = tile(
        changes=[('Treatment 1 · campaign on', _HOLDBACK, 7000.0, None),
                 ('Treatment 2 · ads paused', _GO_DARK, -4000.0, -1.0)],
        total_spend=11_000.0, budget=6000.0, meets_budget=False,
    )
    self.assertEqual(mixed.detail, '$1.0k over your $6.0k goal')
    unknown = tile(changes=holdback, total_spend=8000.0, budget=6000.0)
    self.assertIsNone(unknown.detail)
    multi = tile(
        changes=[('Treatment 1 · ads paused', _GO_DARK, -4000.0, -1.0),
                 ('Treatment 2 · spend increased', _HEAVY_UP, 3000.0, 0.5)],
        total_spend=7000.0,
    )
    self.assertEqual(
        multi.formula,
        'Treatment 1: -$4.0k (-100%) · Treatment 2: +$3.0k (+50%)',
    )

  @parameterized.named_parameters(
      ('holdback', [('T', _HOLDBACK, 8000.0, None)],
       ('It needs about $8.0k of campaign spend in the treatment geos over the'
        ' 30-day test.')),
      ('go_dark', [('T', _GO_DARK, -5000.0, -0.5)],
       ('It pauses $5.0k of spend in the treatment geos (50% of their BAU'
        ' spend) over the 30-day test.')),
      ('heavy_up', [('T', _HEAVY_UP, 2000.0, None)],
       ('It adds +$2.0k of extra spend in the treatment geos over the 30-day'
        ' test.')),
      ('multi', [('Treatment 1 · x', _GO_DARK, -5000.0, None),
                 ('Treatment 2 · y', _HOLDBACK, 1000.0, None)],
       ('Spend over the 30-day test: Treatment 1 pauses $5.0k of spend in its'
        ' geos; Treatment 2 needs about $1.0k of campaign spend in its geos.')),
  )
  def test_spend_sentence(self, changes, expected):
    self.assertEqual(
        report_text.DesignText('$').spend_sentence(
            changes, 8000.0, 'Campaign spend', 30
        ),
        expected,
    )

  @parameterized.named_parameters(
      ('budget', 5000.0, True, True, 'Spending your $5.0k budget goal'),
      ('target', None, True, True, 'costs your target iCPA of $2.00'),
      ('no_target', None, False, False, 'At the planned spend and an iCPA of'),
      ('budget_no_holdback', 5000.0, False, True, 'your target iCPA'),
  )
  def test_chance_sentence(self, budget, holdback, has_target, phrase):
    sentence = report_text.DesignText('$').chance_sentence(
        0.75, 2.0, budget, holdback, has_target
    )
    self.assertIn(phrase, sentence)
    self.assertIn('75%', sentence)

  @parameterized.named_parameters(
      ('no_goal', False, [], 'Set goals to rank candidates against them.'),
      ('missed', True, ['budget', 'MDE'],
       ('This candidate misses your budget goal and MDE goal; 1 of 3'
        ' candidates meet all of them.')),
      ('met', True, [],
       'This candidate meets your goals; 1 of 3 candidates meet all of them.'),
  )
  def test_goals_sentence(self, any_goal, missed, expected):
    self.assertEqual(
        report_text.goals_sentence(any_goal, missed, 1, 3), expected
    )

  @parameterized.named_parameters(
      ('budget', True, True, '60% chance at your budget goal'),
      ('target', False, True, '60% chance at your target iCPA'),
      ('planned', False, False, '60% chance at the planned spend'),
  )
  def test_chance_detail(self, has_budget, has_target, expected):
    self.assertEqual(
        report_text.chance_detail(0.6, has_budget, has_target),
        expected,
    )

  def test_chance_wording_keeps_bands(self):
    self.assertEqual(
        report_text.chance_detail(0.796, True, True),
        '79.6% chance at your budget goal',
    )
    sentence = report_text.DesignText('$').chance_sentence(
        float('nan'), 2.0, None, True, True
    )
    self.assertIn('is n/a.', sentence)

  def test_tile_details(self):
    self.assertEqual(
        report_text.mde_detail(None, None),
        'Smallest lift the test reliably detects',
    )
    self.assertEqual(report_text.mde_detail(0.05, True), 'meets your 5.0% goal')
    self.assertEqual(
        report_text.mde_detail(0.05, False), 'misses your 5.0% goal'
    )
    self.assertEqual(
        report_text.mde_formula([('T', 0.04)], 120.0, 30),
        '≈ 120 incremental conversions over 30 days',
    )
    self.assertEqual(
        report_text.mde_formula([('T1 · a', 0.04), ('T2 · b', 0.05)], 0.0, 30),
        'T1: 4.0% · T2: 5.0%',
    )
    self.assertEqual(
        report_text.aa_text(False, None),
        'Treatment and control geos already differed.',
    )
    self.assertEqual(
        report_text.split_detail(0.25, [('T1 · a', 3), ('T2 · b', 4)]),
        'Treatment geos hold 25.0% of conversions (T1: 3, T2: 4)',
    )
    self.assertEqual(
        report_text.split_detail(float('nan'), [('T', 3)]),
        'Treatment geos hold n/a of conversions',
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
