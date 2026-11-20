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

  def test_aa_drift(self):
    design_set = report_fixtures.make_design_set(n_candidates=1)
    design = next(iter(design_set.designs.values()))
    drift = report_data.aa_drift(design, 'cell_1')
    assert drift is not None
    self.assertLess(drift.start, drift.end)
    self.assertIsNone(report_data.aa_drift(design, 'cell_9'))

  def test_design_helpers(self):
    design_set = report_fixtures.make_design_set(n_candidates=1)
    design = next(iter(design_set.designs.values()))
    self.assertEqual(report_data.alpha_and_sides(design), (0.1, 'two-sided'))
    self.assertEqual(report_data.min_r2(design), 0.8)
    pretest = report_data.pretest_range(design)
    assert pretest is not None
    self.assertEqual(pretest[0], report_fixtures.PRETEST_START)


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
    z_crit = report_data.z_alpha(0.1, api.TestType.TWO_SIDED)
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
    self.assertLess(
        report_data.z_alpha(0.1, api.TestType.ONE_SIDED),
        report_data.z_alpha(0.1, api.TestType.TWO_SIDED),
    )
    two = report_data.plan_numbers(self.summary, self.design, self.goals)
    one = report_data.plan_numbers(
        self.summary,
        self._with_config(test_type=api.TestType.ONE_SIDED),
        self.goals,
    )
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


if __name__ == '__main__':
  absltest.main()
