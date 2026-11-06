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

"""Tests for the GeoX visualizer facade."""

from absl.testing import absltest
from meridian_geox import report_fixtures
from meridian_geox import visualizer


class VisualizerTest(absltest.TestCase):

  def test_facade_exports(self):
    for name in (
        'MapConfig',
        'byo_dma_map_config',
        'byo_city_map_config',
        'boundary_coverage',
        'plot_geo_assignment',
        'resolve_geo_metadata',
        'design_summaries',
        'spend_changes',
        'experiment_schedule',
        'effect_split',
        'execution_checks',
        'plot_spend_pacing',
        'tradeoff_chart',
        'power_simulator',
        'Framing',
        'framed',
        'DESIGN_SELECTION',
    ):
      self.assertTrue(hasattr(visualizer, name), name)

  def test_end_to_end_charts(self):
    design_set = report_fixtures.make_design_set(n_candidates=2)
    goals = visualizer.DesignGoals(target_mde=0.05)
    summaries = visualizer.design_summaries(design_set, goals)
    design = design_set.designs[summaries[0].design_id]
    chart = visualizer.tradeoff_chart(summaries, goals)
    self.assertIn('layer', chart.to_dict())
    result = report_fixtures.make_analysis_result(design)
    checks = visualizer.execution_checks(result, None)
    self.assertLen(checks, 1)


if __name__ == '__main__':
  absltest.main()
