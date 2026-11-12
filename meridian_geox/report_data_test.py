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

"""Tests for GeoX report data."""

import math

from absl.testing import absltest
from absl.testing import parameterized
from meridian_geox import api
from meridian_geox import report_data
from meridian_geox import report_fixtures
from meridian_geox import report_text
import pandas as pd


class DesignGoalsTest(parameterized.TestCase):

  @parameterized.parameters(
      dict(target_mde=0.0),
      dict(target_mde=1.0),
      dict(target_budget=0.0),
      dict(target_cpic=-1.0),
  )
  def test_invalid(self, **kwargs):
    with self.assertRaises(ValueError):
      report_data.DesignGoals(**kwargs)

  def test_any_set(self):
    self.assertFalse(report_data.DesignGoals().any_set)
    self.assertTrue(report_data.DesignGoals(target_mde=0.1).any_set)


class GeoNamesTest(absltest.TestCase):

  def test_resolve(self):
    names = report_data.resolve_geo_metadata(
        pd.DataFrame({
            'location': ['501', 'Berlin'],
            'geo_name': ['New York, NY', 'Berlin'],
            'latitude': [1.0, 2.0],
        })
    )
    self.assertEqual(names('200501'), 'New York, NY')
    self.assertEqual(names('geoTargetConstants/200501'), 'New York, NY')
    self.assertEqual(names('Berlin'), 'Berlin')
    self.assertEqual(names('unknown'), 'unknown')
    self.assertLen(names, 2)

  def test_name_column(self):
    names = report_data.resolve_geo_metadata(
        pd.DataFrame({'Location': ['1'], 'Name': ['One']})
    )
    self.assertEqual(names('1'), 'One')

  def test_missing_columns(self):
    with self.assertRaises(ValueError):
      report_data.resolve_geo_metadata(pd.DataFrame({'location': ['1']}))

  def test_none(self):
    self.assertEqual(report_data.resolve_geo_metadata(None)('x'), 'x')


class DesignSummariesTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.design_set = report_fixtures.make_design_set(n_candidates=4)
    self.design = next(iter(self.design_set.designs.values()))

  def test_default_goals(self):
    goals = report_data.default_goals(self.design, report_data.DesignGoals())
    self.assertEqual(goals.target_budget, 12_000.0)
    self.assertEqual(goals.target_cpic, 1.0)
    self.assertIsNone(goals.target_mde)

  def test_overrides_win(self):
    goals = report_data.default_goals(
        self.design, report_data.DesignGoals(target_budget=5.0)
    )
    self.assertEqual(goals.target_budget, 5.0)

  def test_ranking_puts_feasible_first(self):
    goals = report_data.DesignGoals(
        target_mde=0.03, target_budget=12_000, target_cpic=1.0
    )
    summaries = report_data.design_summaries(self.design_set, goals)
    self.assertLen(summaries, 4)
    self.assertEqual(
        [s.name for s in summaries[:2]],
        ['Option 1 (Recommended)', 'Option 2'],
    )
    self.assertTrue(summaries[0].recommended)
    feasible = [s.feasible for s in summaries]
    # Budgets 8k, 10.5k, 13k, 15.5k; MDEs 1.5%, 2%, 2.5%, 3%.
    self.assertEqual(feasible, [True, True, False, False])
    self.assertEqual(summaries[0].short_id, 'cand0000')
    self.assertEqual(summaries[0].cells['cell_1'].label, 'Treatment')
    self.assertAlmostEqual(summaries[0].max_detectable_cpic, 12_000 / 8_000)
    self.assertEqual(
        report_data.chance_label(summaries[0].success_chance), 'High'
    )

  def test_infeasible_goal_moves_candidates_down(self):
    goals = report_data.DesignGoals(target_mde=0.018)
    summaries = report_data.design_summaries(self.design_set, goals)
    self.assertTrue(summaries[0].feasible)
    self.assertFalse(summaries[1].feasible)
    self.assertTrue(summaries[0].design_id.startswith('cand0000'))

  def test_no_goals(self):
    summaries = report_data.design_summaries(self.design_set)
    self.assertTrue(all(s.feasible for s in summaries))
    self.assertIsNone(summaries[0].success_chance)

  def test_detection_power(self):
    self.assertAlmostEqual(
        report_data.detection_power(100, 100, 1.645, 0.8416), 0.8, places=3
    )
    self.assertTrue(math.isnan(report_data.detection_power(1, 0, 1, 1)))

  def test_chance_label(self):
    self.assertEqual(report_data.chance_label(0.9), 'High')
    self.assertEqual(report_data.chance_label(0.6), 'Medium')
    self.assertEqual(report_data.chance_label(0.1), 'Low')
    self.assertEqual(report_data.chance_label(None), 'Unknown')


class MultiCellTest(absltest.TestCase):

  def test_multicell(self):
    types = {
        'cell_1': api.ExperimentType.GO_DARK,
        'cell_2': api.ExperimentType.HEAVY_UP,
    }
    design_set = report_fixtures.make_design_set(
        n_candidates=2, cells=2, experiment_types=types, target_budget=None
    )
    design = next(iter(design_set.designs.values()))
    self.assertIsNone(report_data.single_experiment_type(design))
    self.assertEqual(
        report_data.group_labels(design),
        {
            'cell_1': 'Treatment 1 · ads paused',
            'cell_2': 'Treatment 2 · spend increased',
        },
    )
    changes = report_data.spend_changes(design)
    self.assertEqual([c.planned_change for c in changes], [-1.0, 1.0])
    summaries = report_data.design_summaries(
        design_set, report_data.DesignGoals(target_mde=0.05, target_cpic=6.5)
    )
    self.assertLen(summaries[0].cells, 2)
    self.assertIsNone(summaries[0].meets_budget)

  def test_budget_pct_plan(self):
    design_set = report_fixtures.make_design_set(
        n_candidates=1,
        experiment_types={'cell_1': api.ExperimentType.HEAVY_UP},
        budget_pct={'cell_1': 0.34},
    )
    design = next(iter(design_set.designs.values()))
    planned = report_data.spend_changes(design)[0].planned_change
    assert planned is not None
    self.assertAlmostEqual(planned, 0.34)


class GeoTableTest(absltest.TestCase):

  def test_geo_table(self):
    geos = report_fixtures.locations(6)
    data = report_fixtures.make_data(geos)
    design = report_fixtures.make_design(
        geos, {'cell_1': geos[:2]}, data=data, excluded=[geos[5]]
    )
    names = report_data.resolve_geo_metadata(
        pd.DataFrame({'location': ['501'], 'geo_name': ['New York, NY']})
    )
    table = report_data.geo_table(design, names)
    self.assertEqual(list(table.columns), report_data.GEO_TABLE_COLUMNS)
    self.assertAlmostEqual(table['spend_share'].sum(), 1.0)
    self.assertTrue(table['stratum'].isna().all())
    self.assertEqual(list(table['group']), ['Treatment'] * 2 + ['Control'] * 3
                     + ['Excluded'])
    self.assertEqual(table.loc[0, 'geo_name'], 'New York, NY')
    self.assertAlmostEqual(table['share'].sum(), 1.0)
    self.assertEqual(table.loc[5, 'share'], 0.0)

  def test_strata(self):
    geos = report_fixtures.locations(6)
    data = report_fixtures.make_data(geos)
    design = report_fixtures.make_design(
        geos, {'cell_1': geos[:2]}, data=data, excluded=[geos[5]]
    )
    remaining = sorted(geos[:5])
    labels = [0, 1, 0, 1, 2]
    design = report_fixtures.with_strata(design, labels)
    strata = report_data.stratum_labels(design)
    self.assertEqual(
        strata, {g: v + 1 for g, v in zip(remaining, labels)}
    )
    table = report_data.geo_table(design, report_data.GeoNames())
    self.assertEqual(
        dict(zip(table['location'], table['stratum']))[remaining[4]], 3
    )
    wrong = report_fixtures.with_strata(design, [0, 1])
    self.assertEqual(report_data.stratum_labels(wrong), {})

  def test_aa_drift(self):
    design_set = report_fixtures.make_design_set(n_candidates=1)
    design = next(iter(design_set.designs.values()))
    drift = report_data.aa_drift(design, 'cell_1')
    assert drift is not None
    self.assertLen(drift.frame, report_fixtures.TEST_DAYS)
    self.assertGreater(drift.final, 0)
    self.assertLess(drift.start, drift.end)
    self.assertIsNone(report_data.aa_drift(design, 'cell_9'))

  def test_plan_numbers_match_power_math(self):
    design_set = report_fixtures.make_design_set(n_candidates=1)
    design = next(iter(design_set.designs.values()))
    goals = report_data.DesignGoals(target_budget=12_000, target_cpic=1.0)
    (summary,) = report_data.design_summaries(design_set, goals)
    numbers = report_data.plan_numbers(summary, design, goals)
    assert numbers is not None
    sigma = summary.cells['cell_1'].mde_abs / (1.644854 + 0.841621)
    self.assertAlmostEqual(numbers.expected, 12_000)
    self.assertAlmostEqual(
        numbers.worst_case, 12_000 - 1.644854 * sigma, places=0
    )
    assert summary.success_chance is not None
    self.assertAlmostEqual(numbers.chance, summary.success_chance, places=6)
    chance = report_data.chance_at(summary, design, 12_000, 1.0)
    assert chance is not None
    self.assertAlmostEqual(chance, numbers.chance, places=6)
    self.assertAlmostEqual(
        numbers.spend_for_power, summary.cells['cell_1'].mde_abs
    )
    self.assertAlmostEqual(numbers.power, 0.8, places=3)

  def test_design_helpers(self):
    design_set = report_fixtures.make_design_set(n_candidates=1)
    design = next(iter(design_set.designs.values()))
    self.assertEqual(report_data.alpha_and_sides(design), (0.1, 'two-sided'))
    self.assertEqual(report_data.min_r2(design), 0.8)
    pretest = report_data.pretest_range(design)
    assert pretest is not None
    self.assertEqual(pretest[0], report_fixtures.PRETEST_START)

  def test_weekly_quality(self):
    geos = report_fixtures.locations(20)
    data = report_fixtures.make_data(geos)
    data.loc[data['location'] == geos[-1], 'conversions'] = 0.0
    weekly = report_data.weekly_quality(data, max_geos=3)
    self.assertEqual(
        list(weekly.columns), ['week', 'location', 'conversions', 'zero', 'z']
    )
    self.assertIn(geos[-1], set(weekly['location']))
    self.assertEqual(weekly['location'].nunique(), 4)
    self.assertTrue(weekly.loc[weekly['location'] == geos[-1], 'zero'].all())
    self.assertTrue(report_data.weekly_quality(None).empty)

  def test_market_spend_and_national(self):
    design_set = report_fixtures.make_design_set(n_candidates=1)
    design = next(iter(design_set.designs.values()))
    total = report_data.market_spend(design)
    assert total is not None
    self.assertGreater(total, 0)
    self.assertTrue(report_data.looks_national(design, total))
    self.assertFalse(report_data.looks_national(design, total * 0.5))
    self.assertFalse(report_data.looks_national(design, None))


class ScheduleTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    design_set = report_fixtures.make_design_set(n_candidates=1)
    self.design = next(iter(design_set.designs.values()))

  def test_dated(self):
    schedule = report_data.experiment_schedule(self.design, '2020-04-01', 7)
    self.assertTrue(schedule.dated)
    self.assertEqual(schedule.test_end, pd.Timestamp('2020-04-30'))
    self.assertEqual(schedule.cooldown_end, pd.Timestamp('2020-05-07'))
    days = schedule.days()
    self.assertLen(days, 37)
    self.assertEqual(days['phase'].value_counts()['Cooldown'], 7)

  def test_undated(self):
    schedule = report_data.experiment_schedule(self.design, None, 0)
    self.assertFalse(schedule.dated)
    self.assertIsNone(schedule.test_end)
    self.assertTrue(schedule.days().empty)

  def test_start_inside_pretest(self):
    with self.assertRaisesRegex(ValueError, 'after the last pretest date'):
      report_data.experiment_schedule(self.design, '2020-02-01')

  def test_negative_cooldown(self):
    with self.assertRaises(ValueError):
      report_data.experiment_schedule(self.design, None, -1)

  def test_ramp_up(self):
    schedule = report_data.experiment_schedule(
        self.design, '2020-04-01', 7, ramp_up_days=3
    )
    phases = schedule.days()['phase']
    self.assertEqual(list(phases[:4]), ['Ramp-up'] * 3 + ['Test'])
    self.assertEqual(phases.value_counts()['Test'], 27)

  def test_ramp_up_too_long(self):
    with self.assertRaisesRegex(ValueError, 'ramp_up_days'):
      report_data.experiment_schedule(self.design, None, 7, ramp_up_days=30)


class AnalysisDataTest(parameterized.TestCase):

  def _design(self, types=None, cells=1):
    geos = report_fixtures.locations(10)
    treatment = {
        f'cell_{c}': geos[2 * (c - 1): 2 * c] for c in range(1, cells + 1)
    }
    return report_fixtures.make_design(
        geos, treatment, experiment_types=types, target_budget=None
    )

  def test_window_and_verdicts(self):
    design = self._design()
    result = report_fixtures.make_analysis_result(design)
    window = report_data.analysis_window(result)
    self.assertTrue(window.complete)
    self.assertEqual(window.cooldown_days, 7)
    self.assertAlmostEqual(report_data.confidence_level(result), 0.9)
    metrics = result.results['cell_1']
    self.assertEqual(
        report_data.verdict(metrics, window, 0.1), report_text.Verdict.YES
    )
    self.assertEqual(
        report_data.verdict(metrics, window, 0.1, execution_ok=False),
        report_text.Verdict.NEEDS_REVIEW,
    )

  def test_no_lift(self):
    result = report_fixtures.make_analysis_result(
        self._design(), lift=100, half_width=400
    )
    window = report_data.analysis_window(result)
    self.assertEqual(
        report_data.verdict(result.results['cell_1'], window, 0.1),
        report_text.Verdict.NO,
    )

  def test_too_early(self):
    result = report_fixtures.make_analysis_result(
        self._design(), observed_days=14
    )
    window = report_data.analysis_window(result)
    self.assertFalse(window.complete)
    self.assertEqual(
        report_data.verdict(result.results['cell_1'], window, 0.1),
        report_text.Verdict.TOO_EARLY,
    )
    self.assertIsNone(
        report_data.effect_split(result.results['cell_1'], window)
    )

  def test_inconsistent_estimate(self):
    bad = api.Estimate(10.0, 20.0, 30.0, 1.0, 0.01)
    self.assertFalse(report_data.estimate_consistent(bad))
    self.assertFalse(
        report_data.estimate_consistent(
            api.Estimate(float('nan'), 0, 1, 1, 1)
        )
    )
    self.assertTrue(
        report_data.estimate_consistent(
            api.Estimate(1.0, float('-inf'), float('inf'), 1, 1)
        )
    )

  def test_efficiency(self):
    result = report_fixtures.make_analysis_result(self._design())
    metrics = result.results['cell_1']
    eff = report_data.efficiency(metrics)
    assert eff is not None
    self.assertEqual(eff.label, 'iCPA')
    self.assertAlmostEqual(eff.point, 2.0)
    self.assertLess(eff.lower, eff.point)
    self.assertGreater(eff.upper, eff.point)
    roas = report_data.efficiency(metrics, response_is_value=True)
    assert roas is not None
    self.assertEqual(roas.label, 'iROAS')
    self.assertAlmostEqual(roas.point, 0.5)
    spend = report_data.implied_spend(metrics)
    assert spend is not None
    self.assertAlmostEqual(spend, 2_000.0)

  def test_efficiency_infinite_bound(self):
    result = report_fixtures.make_analysis_result(
        self._design(), lift=100, half_width=400
    )
    eff = report_data.efficiency(result.results['cell_1'])
    assert eff is not None
    self.assertTrue(math.isinf(eff.upper))

  def test_efficiency_without_spend(self):
    result = report_fixtures.make_analysis_result(self._design(), icpd=None)
    self.assertIsNone(report_data.efficiency(result.results['cell_1']))
    self.assertIsNone(report_data.implied_spend(result.results['cell_1']))

  def test_effect_split(self):
    result = report_fixtures.make_analysis_result(self._design())
    window = report_data.analysis_window(result)
    split = report_data.effect_split(result.results['cell_1'], window)
    assert split is not None
    self.assertAlmostEqual(split.test_effect, 1_000 * 30 / 37)
    self.assertAlmostEqual(split.cooldown_effect, 1_000 * 7 / 37)
    self.assertEqual(split.cooldown_days, 7)

  def test_execution_checks_holdback(self):
    design = self._design()
    result = report_fixtures.make_analysis_result(design)
    data = report_fixtures.make_analysis_data(design)
    (check,) = report_data.execution_checks(result, data)
    self.assertTrue(check.has_spend)
    assert check.observed_change is not None
    self.assertAlmostEqual(check.observed_change, 0.5)
    self.assertIsNone(check.planned_change)
    self.assertIsNone(check.verified)
    self.assertTrue(check.ok)

  def test_execution_checks_multicell(self):
    types = {
        'cell_1': api.ExperimentType.GO_DARK,
        'cell_2': api.ExperimentType.HEAVY_UP,
    }
    design = self._design(types, cells=2)
    result = report_fixtures.make_analysis_result(design)
    data = report_fixtures.make_analysis_data(design)
    checks = report_data.execution_checks(result, data)
    self.assertEqual([c.verified for c in checks], [True, True])
    issue_geo = sorted(design.designs['cell_1'].treatment_geos)[0]
    bad = report_fixtures.make_analysis_data(design, spend_issue_geo=issue_geo)
    checks = report_data.execution_checks(result, bad)
    self.assertEqual(checks[0].issue_geos, (issue_geo,))
    self.assertFalse(checks[0].ok)
    self.assertTrue(checks[1].ok)

  def test_execution_checks_relative_to_control(self):
    types = {
        'cell_1': api.ExperimentType.GO_DARK,
        'cell_2': api.ExperimentType.HEAVY_UP,
    }
    design = self._design(types, cells=2)
    result = report_fixtures.make_analysis_result(design)
    data = report_fixtures.make_analysis_data(design)
    # Seasonality: every geo spends 40% less during the test window.
    in_test = data['date'] >= report_fixtures.PRETEST_START + pd.Timedelta(
        days=report_fixtures.PRETEST_DAYS
    )
    for column in ('spend_cell_1', 'spend_cell_2'):
      data.loc[in_test, column] *= 0.6
    checks = report_data.execution_checks(result, data)
    heavy_up = checks[1]
    assert heavy_up.observed_change is not None
    self.assertAlmostEqual(heavy_up.observed_change, 1.0)
    self.assertTrue(heavy_up.ok)
    self.assertEqual([c.verified for c in checks], [True, True])

  def test_execution_checks_without_spend(self):
    design = self._design()
    result = report_fixtures.make_analysis_result(design)
    (check,) = report_data.execution_checks(result, None)
    self.assertFalse(check.has_spend)
    self.assertTrue(check.ok)

  def test_daily_spend_by_group(self):
    design = self._design()
    result = report_fixtures.make_analysis_result(design)
    data = report_fixtures.make_analysis_data(design)
    frame = report_data.daily_spend_by_group(result, data, 'cell_1')
    self.assertEqual(set(frame['group']), {'Treatment', 'Control'})
    self.assertEqual(list(frame.columns), ['date', 'group', 'spend'])

  def test_robust_z_scores(self):
    geos = report_fixtures.locations(3)
    data = report_fixtures.make_data(geos)
    data.loc[5, 'conversions'] = 10_000.0
    z = report_data.robust_z_scores(data, max_geos=2)
    self.assertEqual(z['location'].nunique(), 2)
    self.assertGreater(z['z'].max(), 10)


if __name__ == '__main__':
  absltest.main()
