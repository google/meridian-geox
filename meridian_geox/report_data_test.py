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

import dataclasses
import math
import statistics

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
    feasible = [s.feasible for s in summaries]
    # Budgets 8k, 10.5k, 13k, 15.5k; MDEs 1.5%, 2%, 2.5%, 3%.
    self.assertEqual(feasible, [True, True, False, False])
    self.assertEqual(summaries[0].cells['cell_1'].label, 'Treatment')
    self.assertAlmostEqual(summaries[0].max_detectable_cpic, 12_000 / 8_000)

  def test_infeasible_goal_moves_candidates_down(self):
    goals = report_data.DesignGoals(target_mde=0.018)
    summaries = report_data.design_summaries(self.design_set, goals)
    self.assertTrue(summaries[0].feasible)
    self.assertFalse(summaries[1].feasible)
    self.assertTrue(summaries[0].design_id.startswith('cand0000'))

  def test_no_goals(self):
    summaries = report_data.design_summaries(self.design_set)
    self.assertTrue(all(s.feasible for s in summaries))

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
    self.assertEqual(
        report_data.design_params(design), (0.1, 0.8, api.TestType.TWO_SIDED)
    )
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


class ScheduleTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    design_set = report_fixtures.make_design_set(n_candidates=1)
    self.design = next(iter(design_set.designs.values()))

  def test_dated(self):
    schedule = report_data.experiment_schedule(
        self.design, '2020-04-01', 7, ramp_up_days=3
    )
    self.assertTrue(schedule.dated)
    self.assertEqual(schedule.test_end, pd.Timestamp('2020-04-30'))
    self.assertEqual(schedule.cooldown_start, pd.Timestamp('2020-05-01'))
    self.assertEqual(schedule.cooldown_end, pd.Timestamp('2020-05-07'))
    phases = schedule.days()['phase']
    self.assertEqual(list(phases[:4]), ['Ramp-up'] * 3 + ['Test'])
    self.assertEqual(
        phases.value_counts().to_dict(),
        {'Ramp-up': 3, 'Test': 27, 'Cooldown': 7},
    )

  def test_undated(self):
    schedule = report_data.experiment_schedule(self.design, None, 0)
    self.assertFalse(schedule.dated)
    self.assertIsNone(schedule.test_end)
    self.assertTrue(schedule.days().empty)

  @parameterized.parameters(
      (dict(test_start='2020-02-01'), 'after the last pretest date'),
      (dict(cooldown_days=-1), 'cooldown_days'),
      (dict(cooldown_days=7, ramp_up_days=30), 'ramp_up_days'),
  )
  def test_invalid(self, kwargs, message):
    with self.assertRaisesRegex(ValueError, message):
      report_data.experiment_schedule(self.design, **kwargs)


class AnalysisDataTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.design = self._design()
    self.result = report_fixtures.make_analysis_result(self.design)
    self.metrics = self.result.results['cell_1']

  def _design(self, types=None, cells=1):
    geos = report_fixtures.locations(10)
    treatment = {
        f'cell_{c}': geos[2 * (c - 1): 2 * c] for c in range(1, cells + 1)
    }
    return report_fixtures.make_design(
        geos, treatment, experiment_types=types, target_budget=None
    )

  def test_window(self):
    window = report_data.analysis_window(self.result)
    self.assertTrue(window.complete)
    self.assertEqual(window.cooldown_days, 7)
    self.assertEqual(window.test_days, window.planned_days)
    self.assertEqual(window.test_end, window.planned_end)
    self.assertEqual(
        window.cooldown_start, window.planned_end + pd.Timedelta(days=1)
    )
    self.assertAlmostEqual(report_data.confidence_level(self.result), 0.9)
    self.assertEqual(
        report_data.verdict(self.metrics, window, 0.1, execution_ok=False),
        report_text.Verdict.NEEDS_REVIEW,
    )

  @parameterized.parameters(
      (dict(), report_text.Verdict.YES),
      (dict(lift=100, half_width=400), report_text.Verdict.NO),
      (dict(observed_days=14), report_text.Verdict.TOO_EARLY),
  )
  def test_verdict(self, kwargs, expected):
    result = report_fixtures.make_analysis_result(self.design, **kwargs)
    window = report_data.analysis_window(result)
    metrics = result.results['cell_1']
    self.assertEqual(report_data.verdict(metrics, window, 0.1), expected)
    early = expected == report_text.Verdict.TOO_EARLY
    self.assertEqual(window.complete, not early)
    if early:
      self.assertIsNone(report_data.effect_split(metrics, window))

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
    eff = report_data.efficiency(self.metrics)
    assert eff is not None
    self.assertEqual(eff.label, 'iCPA')
    self.assertAlmostEqual(eff.point, 2.0)
    self.assertLess(eff.lower, eff.point)
    self.assertGreater(eff.upper, eff.point)
    roas = report_data.efficiency(self.metrics, response_is_value=True)
    assert roas is not None
    self.assertEqual(roas.label, 'iROAS')
    self.assertAlmostEqual(roas.point, 0.5)
    spend = report_data.implied_spend(self.metrics)
    assert spend is not None
    self.assertAlmostEqual(spend, 2_000.0)
    # A CI that reaches zero lift gives an unbounded iCPA.
    weak = report_fixtures.make_analysis_result(
        self.design, lift=100, half_width=400
    )
    weak_eff = report_data.efficiency(weak.results['cell_1'])
    assert weak_eff is not None
    self.assertTrue(math.isinf(weak_eff.upper))

  @parameterized.parameters(
      (0.5, 0.3, 0.7, True),
      (0.5, 0.7, 0.3, True),  # Flipped by a spend cut, not across zero.
      (5.0, 0.3, 0.7, False),
      (-1120.55, 1932.77, -3667.16, False),
      (0.0, -0.1, 0.1, False),
      (math.nan, 0.3, 0.7, False),
  )
  def test_icpd_usable(self, point, low, high, usable):
    icpd = api.Estimate(point, low, high, 1.0, 0.5)
    self.assertEqual(report_data.icpd_usable(icpd), usable)

  def test_unusable_icpd(self):
    no_spend = report_fixtures.make_analysis_result(self.design, icpd=None)
    degenerate = report_fixtures.with_degenerate_icpd(self.result)
    for result in (no_spend, degenerate):
      metrics = result.results['cell_1']
      self.assertIsNone(report_data.efficiency(metrics))
      self.assertIsNone(report_data.implied_spend(metrics))
    _, rows = report_data.estimate_rows(
        report_data.cell_readouts(degenerate), report_text.AnalysisText()
    )
    self.assertIn(
        [report_text.ICPD_METRIC, '— not estimable', '0.500'], rows
    )

  def test_effect_split(self):
    window = report_data.analysis_window(self.result)
    split = report_data.effect_split(self.metrics, window)
    assert split is not None
    self.assertAlmostEqual(split.test_effect, 1_000 * 30 / 37)
    self.assertAlmostEqual(split.cooldown_effect, 1_000 * 7 / 37)
    self.assertEqual(split.cooldown_days, 7)

  def test_execution_checks_holdback(self):
    data = report_fixtures.make_analysis_data(self.design)
    (check,) = report_data.execution_checks(self.result, data)
    self.assertTrue(check.has_spend)
    assert check.observed_change is not None
    self.assertAlmostEqual(check.observed_change, 0.5)
    self.assertIsNone(check.planned_change)
    self.assertIsNone(check.verified)
    self.assertTrue(check.ok)
    (check,) = report_data.execution_checks(self.result, None)
    self.assertFalse(check.has_spend)
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
    # Seasonality: every geo spends 40% less during the test window; changes
    # are measured relative to the control geos.
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


class DesignHelpersTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.single = next(iter(
        report_fixtures.make_design_set(n_candidates=1).designs.values()
    ))
    self.multi = next(iter(
        report_fixtures.make_design_set(n_candidates=1, cells=2)
        .designs.values()
    ))

  def test_single_experiment_type(self):
    self.assertEqual(
        report_data.single_experiment_type(self.single),
        api.ExperimentType.HOLDBACK,
    )
    self.assertIsNone(report_data.single_experiment_type(self.multi))

  def test_group_labels(self):
    self.assertEqual(
        report_data.group_labels(self.single), {'cell_1': 'Treatment'}
    )
    labels = report_data.group_labels(self.multi)
    self.assertEqual(list(labels), ['cell_1', 'cell_2'])
    self.assertTrue(labels['cell_1'].startswith('Treatment 1 · '))
    self.assertTrue(labels['cell_2'].startswith('Treatment 2 · '))

  def test_explicit_cpic_survives_defaults(self):
    goals = report_data.default_goals(
        self.single, report_data.DesignGoals(target_cpic=5.0)
    )
    self.assertEqual(goals.target_cpic, 5.0)


class FixturesTest(absltest.TestCase):

  def test_locations(self):
    self.assertEqual(report_fixtures.locations(2), ['200501', '200803'])
    self.assertEqual(
        report_fixtures.locations(2, dma=False), ['City 1', 'City 2']
    )

  def test_make_data(self):
    geos = report_fixtures.locations(3)
    data = report_fixtures.make_data(geos)
    self.assertLen(data, 3 * report_fixtures.PRETEST_DAYS)
    self.assertCountEqual(set(data[api.LOCATION]), geos)
    self.assertIn(api.SPEND, data.columns)
    multi = report_fixtures.make_data(geos, cells=2)
    self.assertIn('spend_cell_2', multi.columns)
    self.assertNotIn(api.SPEND, multi.columns)
    no_spend = report_fixtures.make_data(geos, with_spend=False)
    self.assertNotIn(api.SPEND, no_spend.columns)

  def test_control_skips_treatment_and_excluded(self):
    geos = report_fixtures.locations(5)
    design = report_fixtures.make_design(
        geos, {'cell_1': geos[:2]}, excluded=geos[2:3]
    )
    self.assertEqual(design.control_geos, set(geos[3:]))
    self.assertEqual(design.excluded_geos, {geos[2]})


class PlanAndChecksTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.design_set = report_fixtures.make_design_set(n_candidates=1)
    self.design = next(iter(self.design_set.designs.values()))
    self.goals = report_data.default_goals(
        self.design, report_data.DesignGoals()
    )
    self.summary = report_data.design_summaries(self.design_set, self.goals)[0]

  def _with_config(self, **kwargs):
    config = self.design.design_config
    assert config is not None
    return dataclasses.replace(
        self.design, design_config=dataclasses.replace(config, **kwargs)
    )

  def _with_cells(self, summary, **kwargs):
    cells = {
        name: dataclasses.replace(cell, **kwargs.get(name, {}))
        for name, cell in summary.cells.items()
    }
    return dataclasses.replace(summary, cells=cells)

  def test_plan_numbers(self):
    plan = report_data.plan_numbers(self.summary, self.design, self.goals)
    assert plan is not None
    cell = self.summary.cells['cell_1']
    self.assertEqual(plan.cpic, 1.0)
    self.assertAlmostEqual(plan.power, 0.8)
    self.assertEqual(plan.mde_abs, cell.mde_abs)
    normal = statistics.NormalDist()
    z_crit, _ = report_data.z_scores(self.design)
    sigma = cell.mde_abs / (z_crit + normal.inv_cdf(0.8))
    self.assertAlmostEqual(
        plan.chance, normal.cdf(plan.spend / plan.cpic / sigma - z_crit)
    )

  def test_plan_numbers_none_without_mde_or_cpic(self):
    no_mde = self._with_cells(self.summary, cell_1={'mde_abs': math.nan})
    self.assertIsNone(
        report_data.plan_numbers(no_mde, self.design, self.goals)
    )
    no_cpic = self._with_cells(self.summary, cell_1={'implied_cpic': math.nan})
    self.assertIsNone(report_data.plan_numbers(
        no_cpic, self.design, report_data.DesignGoals()
    ))

  def test_plan_numbers_one_sided(self):
    one_sided = self._with_config(test_type=api.TestType.ONE_SIDED)
    self.assertLess(
        report_data.z_scores(one_sided)[0], report_data.z_scores(self.design)[0]
    )
    two = report_data.plan_numbers(self.summary, self.design, self.goals)
    one = report_data.plan_numbers(self.summary, one_sided, self.goals)
    assert one is not None and two is not None
    self.assertNotAlmostEqual(one.chance, two.chance)

  def test_plan_numbers_degenerate_power(self):
    design = self._with_config(power=0.01)
    self.assertIsNone(
        report_data.plan_numbers(self.summary, design, self.goals)
    )

  def test_design_checks(self):
    checks = report_data.design_checks(self.summary, self.design)
    drift = report_data.aa_drift(self.design, 'cell_1')
    assert drift is not None
    self.assertTrue(checks.good_fit)
    self.assertTrue(checks.aa_passed)
    self.assertEqual(checks.aa_window, (drift.start, drift.end))
    self.assertEqual(
        checks.treatment_share,
        sum(c.treatment_share for c in self.summary.cells.values()),
    )
    failing = dataclasses.replace(self.summary, min_aa_p_value=0.01)
    self.assertFalse(report_data.design_checks(failing, self.design).aa_passed)

  def test_design_checks_nan_r2_in_any_cell(self):
    design_set = report_fixtures.make_design_set(n_candidates=1, cells=2)
    design = next(iter(design_set.designs.values()))
    summary = report_data.design_summaries(design_set)[0]
    for cell in ('cell_1', 'cell_2'):
      nan_cell = self._with_cells(summary, **{cell: {'r2': math.nan}})
      checks = report_data.design_checks(nan_cell, design)
      self.assertTrue(math.isnan(checks.r2))
      self.assertFalse(checks.good_fit)

  def test_bau_spend(self):
    def change(amount, planned):
      return report_data.SpendChange(
          'cell_1', api.ExperimentType.GO_DARK, 'Treatment', amount, planned
      )

    self.assertIsNone(report_data.bau_spend(change(-500.0, None)))
    self.assertIsNone(report_data.bau_spend(change(-500.0, 0.0)))
    self.assertEqual(report_data.bau_spend(change(-500.0, -0.5)), 1000.0)

  def test_missed_goals(self):
    summary = dataclasses.replace(
        self.summary, meets_mde=False, meets_budget=None, meets_cpic=True
    )
    self.assertEqual(report_data.missed_goals(summary), ['MDE'])
    summary = dataclasses.replace(
        self.summary, meets_mde=False, meets_budget=False, meets_cpic=False
    )
    self.assertEqual(
        report_data.missed_goals(summary), ['MDE', 'budget', 'iCPA']
    )

  def test_chance_at_degenerate_power(self):
    design = self._with_config(power=0.01)
    self.assertIsNone(
        report_data.chance_at(self.summary, design, 1_000.0, 1.0)
    )


class SummaryFallbacksTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.design_set = report_fixtures.make_design_set(n_candidates=1)
    self.design_id, self.design = next(iter(self.design_set.designs.items()))

  def test_default_goals_without_config(self):
    goals = report_data.DesignGoals(target_mde=0.05)
    for design in (
        dataclasses.replace(self.design, design_config=None),
        dataclasses.replace(self.design, constraints=None),
    ):
      self.assertEqual(report_data.default_goals(design, goals), goals)

  def test_default_goals_use_positive_holdback_values_only(self):
    design_set = report_fixtures.make_design_set(
        n_candidates=1,
        cells=3,
        experiment_types={
            'cell_1': api.ExperimentType.HOLDBACK,
            'cell_2': api.ExperimentType.GO_DARK,
            'cell_3': api.ExperimentType.HOLDBACK,
        },
    )
    design = next(iter(design_set.designs.values()))
    assert design.design_config is not None
    design = dataclasses.replace(
        design,
        design_config=dataclasses.replace(
            design.design_config,
            cost_per_incremental_conversion={
                'cell_1': 3.0, 'cell_2': 0.5, 'cell_3': 0.0
            },
        ),
        constraints=api.Constraints(budget_constraint={
            'cell_1': api.Budget(budget=12_000.0),
            'cell_2': api.Budget(budget=5_000.0),
            'cell_3': api.Budget(budget=0.0),
        }),
    )
    goals = report_data.default_goals(design, report_data.DesignGoals())
    self.assertEqual(goals.target_budget, 12_000.0)
    self.assertEqual(goals.target_cpic, 3.0)

  def test_budget_goal_splits_across_holdback_cells_only(self):
    design_set = report_fixtures.make_design_set(
        n_candidates=1,
        cells=2,
        experiment_types={
            'cell_1': api.ExperimentType.HOLDBACK,
            'cell_2': api.ExperimentType.GO_DARK,
        },
    )
    summary = report_data.design_summaries(
        design_set, report_data.DesignGoals(target_budget=12_000.0)
    )[0]
    # Holdback: 12,000 / 8,000 = 1.5; go-dark: 8,000 / 8,000 = 1.0.
    self.assertAlmostEqual(summary.max_detectable_cpic, 1.0)


class CandidateTotalsTest(absltest.TestCase):

  def test_totals(self):
    design_set = report_fixtures.make_design_set(n_candidates=1, cells=2)
    summary = report_data.design_summaries(design_set)[0]
    self.assertEqual(summary.treatment_geo_count, 4)
    self.assertEqual(summary.total_mde_abs, 16_000.0)


class HoldbackSpendTest(absltest.TestCase):

  def test_budget_goal_counts_holdback_cells_only(self):
    design_set = report_fixtures.make_design_set(
        n_candidates=1,
        cells=3,
        experiment_types={
            'cell_1': api.ExperimentType.HOLDBACK,
            'cell_2': api.ExperimentType.HOLDBACK,
            'cell_3': api.ExperimentType.GO_DARK,
        },
    )
    summary = report_data.design_summaries(
        design_set, report_data.DesignGoals(target_budget=12_000.0)
    )[0]
    # Each holdback cell gets 12,000 / 2 = 6,000 of the goal; MDE is 8,000
    # conversions per cell, so 6,000 / 8,000 = 0.75 beats go-dark's 1.0.
    self.assertAlmostEqual(summary.max_detectable_cpic, 0.75)
    # Holdback spend is 2 x 8,000 = 16,000 > 12,000; go-dark spend is ignored.
    self.assertFalse(summary.meets_budget)


if __name__ == '__main__':
  absltest.main()
