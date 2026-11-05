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


from absl.testing import absltest
from absl.testing import parameterized
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

  def test_infeasible_goal_moves_candidates_down(self):
    goals = report_data.DesignGoals(target_mde=0.018)
    summaries = report_data.design_summaries(self.design_set, goals)
    self.assertTrue(summaries[0].feasible)
    self.assertFalse(summaries[1].feasible)
    self.assertTrue(summaries[0].design_id.startswith('cand0000'))

  def test_no_goals(self):
    summaries = report_data.design_summaries(self.design_set)
    self.assertTrue(all(s.feasible for s in summaries))


if __name__ == '__main__':
  absltest.main()
