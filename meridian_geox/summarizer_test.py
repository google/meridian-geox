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

"""Tests for the GeoX design and analysis summarizers."""

import dataclasses
import json
import os
import tempfile

from absl.testing import absltest
from absl.testing import parameterized
from meridian_geox import api
from meridian_geox import maps
from meridian_geox import report_fixtures
from meridian_geox import summarizer
import pandas as pd


def _write_dma_topojson(path: str) -> None:
  geometries = [
      {
          'type': 'Polygon',
          'id': int(code),
          'arcs': [[0]],
          'properties': {'note': 'x' * 100},
      }
      for code in report_fixtures.DMA_CODES[:-1]
  ]
  topo = {
      'type': 'Topology',
      'arcs': [[[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1]]],
      'objects': {
          'nielsen_dma': {'type': 'GeometryCollection',
                          'geometries': geometries}
      },
  }
  with open(path, 'w') as f:
    json.dump(topo, f)


class ChanceToneTest(parameterized.TestCase):

  @parameterized.parameters(
      (0.9, 'success'),
      (0.6, 'warning'),
      (0.2, 'error'),
      (None, 'neutral'),
      (float('nan'), 'neutral'),
  )
  def test_chance_tone(self, chance, tone):
    self.assertEqual(summarizer._chance_tone(chance), tone)


class GoalToneTest(parameterized.TestCase):

  @parameterized.parameters(
      (None, 'neutral'),
      (True, 'success'),
      (False, 'warning'),
  )
  def test_goal_tone(self, value, tone):
    self.assertEqual(summarizer._goal_tone(value), tone)


class DesignSummarizerTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.tmp = self.enter_context(tempfile.TemporaryDirectory())
    self.topo = os.path.join(self.tmp, 'nielsentopo.json')
    _write_dma_topojson(self.topo)
    self.design_set = report_fixtures.make_design_set(n_candidates=4)
    self.geo_metadata = pd.DataFrame({
        'location': ['501', '803'],
        'geo_name': ['New York, NY', 'Los Angeles, CA'],
    })

  def _html(self, **kwargs) -> str:
    return summarizer.DesignSummarizer(
        self.design_set, **kwargs
    ).generate_html()

  def test_goals_come_from_design(self):
    # The budget goal (12,000) comes from Constraints and the target iCPA
    # (1.00) from DesignConfig; only the MDE goal is a report argument.
    html = self._html(target_mde=0.02)
    self.assertIn('your 2.0% goal', html)
    self.assertIn('budget ≤ 12.0k', html)
    self.assertIn('target iCPA 1.00', html)

  def test_national_budget_callout(self):
    design_set = report_fixtures.make_design_set(
        n_candidates=4, target_budget=10_000_000
    )
    html = summarizer.DesignSummarizer(design_set).generate_html()
    self.assertIn('Check the budget scope', html)

  def test_without_map_or_dates(self):
    html = self._html(cooldown_days=0)
    self.assertIn('Map omitted: no boundary file was passed', html)
    self.assertIn('Pass test_start to see calendar dates', html)
    self.assertNotIn('id="geox-resource-', html)

  def test_low_budget_misses_goal(self):
    design_set = report_fixtures.make_design_set(
        n_candidates=4, target_budget=500
    )
    html = summarizer.DesignSummarizer(design_set).generate_html()
    self.assertIn('misses your budget goal', html)
    self.assertIn('Low chance', html)
    self.assertIn('To raise the chance of a conclusive result', html)

  def test_compare_card(self):
    html = self._html()
    compare = html[html.index('id="compare"'):]
    self.assertIn('Compare candidate designs', compare)
    self.assertIn('geox-selectable', compare)
    self.assertIn('Setting the target iCPA', compare)

  def test_launch_plan_card(self):
    codes = report_fixtures.DMA_CODES
    html = self._html(
        test_start='2020-04-01', cooldown_days=7, ramp_up_days=3,
        targeting_map=pd.DataFrame({
            'location': [int(c) for c in codes],
            'postal_code': [f'Z{c}' for c in codes],
        }),
    )
    launch = html[html.index('id="launch"'):]
    self.assertIn('Launch plan', launch)
    self.assertIn('Wed, Apr 1, 2020', launch)
    self.assertIn('7-day cooldown', launch)
    self.assertIn('the first 3 a ramp-up', launch)
    self.assertIn('yellow days at the start are ramp-up', launch)
    self.assertIn('geox_option1_targeting.csv', launch)

  def test_visuals_card(self):
    html = self._html(map_config=maps.byo_dma_map_config(polygon_url=self.topo))
    visuals = html[html.index('id="visuals"'):]
    self.assertIn('Where the test runs and what it can detect', visuals)
    self.assertIn('Where would the test run?', visuals)

  def test_to_iframe(self):
    summarizer_obj = summarizer.DesignSummarizer(self.design_set)
    frame = summarizer_obj.to_iframe(800)
    self.assertTrue(frame.startswith('<iframe srcdoc="'))
    self.assertIn('height:800px', frame)
    self.assertEqual(summarizer_obj._repr_html_()[:7], '<iframe')

  def test_output_design_summary(self):
    path = summarizer.DesignSummarizer(
        self.design_set
    ).output_design_summary('design.html', self.tmp)
    self.assertEqual(path, os.path.join(self.tmp, 'design.html'))
    self.assertTrue(os.path.exists(path))

  def test_empty_metrics(self):
    with self.assertRaisesRegex(ValueError, 'design_metrics'):
      summarizer.DesignSummarizer(
          api.DesignSet(designs={}, design_metrics=pd.DataFrame())
      )

  def test_empty_designs(self):
    with self.assertRaisesRegex(ValueError, 'at least one design'):
      summarizer.DesignSummarizer(
          api.DesignSet(
              designs={}, design_metrics=self.design_set.design_metrics
          )
      )

  @parameterized.parameters(
      dict(targeting_map=pd.DataFrame({'location': ['a']})),
      dict(
          targeting_map=pd.DataFrame(
              {'location': ['a'], 'geo_name': ['b']}
          )
      ),
      dict(targeting_map=pd.DataFrame({'location': ['a'], 'group': ['b']})),
      dict(test_start='2020-02-01'),
      dict(target_mde=2.0),
      dict(cooldown_days=-1),
  )
  def test_invalid_inputs(self, **kwargs):
    with self.assertRaises(ValueError):
      summarizer.DesignSummarizer(self.design_set, **kwargs)


class DesignChipsTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.design_set = report_fixtures.make_design_set(n_candidates=1)

  def _render(self, transform) -> str:
    designs = {k: transform(d) for k, d in self.design_set.designs.items()}
    return summarizer.DesignSummarizer(
        api.DesignSet(
            designs=designs, design_metrics=self.design_set.design_metrics
        )
    ).generate_html()

  def test_region_counts_control_treatment_and_excluded_geos(self):
    def exclude_one(design):
      geo = sorted(design.control_geos)[0]
      return dataclasses.replace(
          design,
          control_geos=set(design.control_geos) - {geo},
          excluded_geos={geo},
      )

    self.assertIn('Geos (12 geos)', self._render(exclude_one))

  def test_pretest_chip(self):
    self.assertIn('<span>Pretest:</span>', self._render(lambda d: d))
    html = self._render(lambda d: dataclasses.replace(d, data=None))
    self.assertNotIn('<span>Pretest:</span>', html)

  def test_duration_chip(self):
    self.assertIn('<span>Duration:</span> 30 days', self._render(lambda d: d))
    html = self._render(lambda d: dataclasses.replace(d, design_config=None))
    self.assertNotIn('<span>Duration:</span>', html)

  def test_type_chips(self):
    single = self._render(lambda d: d)
    self.assertIn('<span>Type:</span>', single)
    self.assertNotIn('<span>Types:</span>', single)
    self.assertNotIn('<span>Cells:</span>', single)
    multi = summarizer.DesignSummarizer(
        report_fixtures.make_design_set(n_candidates=1, cells=2)
    ).generate_html()
    self.assertIn('<span>Types:</span>', multi)
    self.assertIn('<span>Cells:</span> 2', multi)
    self.assertNotIn('<span>Type:</span>', multi)

  def test_method_chip(self):
    config = next(iter(self.design_set.designs.values())).design_config
    assert config is not None
    rule = config.geo_assignment_rule.name.replace('_', ' ').title()
    self.assertIn(
        f'<span>Method:</span> {config.methodology.name} · {rule}',
        self._render(lambda d: d),
    )

  def test_mde_goal_chip(self):
    html = summarizer.DesignSummarizer(
        self.design_set, target_mde=0.05
    ).generate_html()
    self.assertIn('MDE ≤ 5.0%', html)
    self.assertNotIn('MDE ≤', self._render(lambda d: d))

  def test_banner_has_chance_sentence(self):
    self.assertIn(
        'chance of a statistically significant result',
        self._render(lambda d: d),
    )


if __name__ == '__main__':
  absltest.main()
