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

import base64
import json
import os
import re
import tempfile

from absl.testing import absltest
from absl.testing import parameterized
from meridian_geox import api
from meridian_geox import maps
from meridian_geox import report_fixtures
from meridian_geox import summarizer
import pandas as pd

_GO_DARK_HEAVY_UP = {
    'cell_1': api.ExperimentType.GO_DARK,
    'cell_2': api.ExperimentType.HEAVY_UP,
}


def _square(x: float, y: float) -> list[list[list[float]]]:
  return [[[x, y], [x + 1, y], [x + 1, y + 1], [x, y + 1], [x, y]]]


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


def _write_city_geojson(path: str, names: list[str]) -> None:
  features = [
      {
          'type': 'Feature',
          'properties': {'name': name},
          'geometry': {'type': 'Polygon', 'coordinates': _square(9 + i, 50)},
      }
      for i, name in enumerate(names)
  ]
  with open(path, 'w') as f:
    json.dump({'type': 'FeatureCollection', 'features': features}, f)


def _save(html: str, name: str) -> None:
  out_dir = os.environ.get('TEST_UNDECLARED_OUTPUTS_DIR')
  if out_dir:
    with open(os.path.join(out_dir, name), 'w', encoding='utf-8') as f:
      f.write(html)


def _downloads(html: str) -> dict[str, str]:
  out = {}
  for href, name in re.findall(
      r'href="data:text/csv;charset=utf-8;base64,([^"]+)" download="([^"]+)"',
      html,
  ):
    out[name] = base64.b64decode(href).decode('utf-8')
  return out


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
    return summarizer.DesignSummarizer(self.design_set, **kwargs).generate_html()

  def test_us_holdback_report(self):
    html = self._html(
        geo_metadata=self.geo_metadata,
        map_config=maps.byo_dma_map_config(polygon_url=self.topo),
        currency_symbol='$',
        target_mde=0.025,
        target_budget=12_000,
        target_cpic=1.0,
        test_start='2020-04-01',
        cooldown_days=7,
    )
    _save(html, 'design_us_polygon.html')
    self.assertIn('Option 1 (Recommended)', html)
    self.assertIn(summarizer.SELECT_CUE, html)
    self.assertIn('Campaign spend', html)
    self.assertIn('New York, NY', html)
    self.assertIn('Download geo list (CSV)', html)
    self.assertIn('When does the test run?', html)
    self.assertIn('Wed, Apr 1, 2020', html)
    self.assertIn('7-day cooldown', html)
    self.assertIn('United States (DMA boundaries)', html)
    # UX order: recommendation, then the candidate comparison, then next steps.
    # Simulator controls follow the comparison table (UX feedback).
    self.assertLess(
        html.index('Compare candidate designs'),
        html.index('What if the spend or target iCPA changes?'),
    )
    # The ranked table sits at the top of the comparison card.
    compare = html[html.index('id="compare"'):]
    self.assertLess(compare.index('geox-selectable'), compare.index(
        'geox-chart'))
    # Boundary polygons are embedded once for all candidate maps.
    self.assertEqual(html.count('id="geox-resource-'), 1)
    # One unmatched geo (the last DMA has no polygon).
    self.assertIn('Some geos are missing from the map', html)
    downloads = _downloads(html)
    self.assertLen(downloads, 4)
    first_csv = next(iter(downloads.values()))
    self.assertTrue(first_csv.startswith('location,geo_name,group,stratum\n'))
    # v3-level detail.
    for text in (
        'Design checks',
        'Feasibility',
        'It detects a lift of',
        'Where would the test run?',
        'What lift size can this test reliably detect?',
        'tradeoff_axis',
        'Would noise alone have looked like a lift before the test?',
        'Conclusive-result chance',
        'Setting the target iCPA',
        'All candidate designs (ranked)',
        'How were geos split within each stratum?',
        'Are there tracking gaps?',
        'Which geo-weeks behave unusually?',
        'Region',
        'Method',
        'Goals',
    ):
      self.assertIn(text, html)
    self.assertIn('class="geox-takeaway"', html)
    # UX order: recommendation, comparison, visuals, then launch plan and
    # next steps (setup P0).
    self.assertLess(html.index('id="recommended"'), html.index('id="compare"'))
    self.assertLess(html.index('id="compare"'), html.index('id="visuals"'))
    self.assertLess(html.index('id="visuals"'), html.index('id="launch"'))
    launch = html[html.index('id="launch"'):]
    self.assertIn('<h3>Next steps</h3>', launch)
    self.assertIn('How this geo test isolates true ad impact', html)
    self.assertIn('Budget goal = treatment-geo spend', html)
    self.assertIn('class="geox-legend"', html)
    self.assertIn('High-confidence zone', html)
    # v3 launch plan, analyst next steps and documentation links.
    for text in (
        'Deliver the campaign only in the',
        'Run the test from Wed, Apr 1 to Thu, Apr 30, 2020 (30 days)',
        'review the post-test analysis readout',
        '<h3>Next steps</h3>',
        'Run the campaign only in the',
        'Geos (A–Z)',
        'Learn more',
        'Experiment types',
        'Glossary: iCPA',
        'Data quality checks',
    ):
      self.assertIn(text, html)
    # Executive banner: short ID chip with a Copy ID button.
    banner = html[html.index('class="geox-banner"'):]
    self.assertLess(banner.index('geox-id-chip'), banner.index('Copy ID'))

  def test_ramp_up_days(self):
    html = self._html(test_start='2020-04-01', cooldown_days=7, ramp_up_days=3)
    self.assertIn('Ramp-up', html)
    self.assertIn('the first 3 a ramp-up', html)
    self.assertIn('Light yellow days at the start are ramp-up.', html)

  def test_goal_precedence(self):
    html = self._html(
        goals=summarizer.DesignGoals(target_mde=0.05), target_mde=0.02
    )
    self.assertIn('your 2.0% goal', html)

  def test_national_budget_callout(self):
    html = self._html(target_budget=10_000_000)
    self.assertIn('Check the budget scope', html)

  def test_without_map_or_dates(self):
    html = self._html(cooldown_days=0)
    self.assertIn('Map omitted: no boundary file was passed', html)
    self.assertIn('Pass test_start to see calendar dates', html)
    self.assertNotIn('id="geox-resource-', html)

  def test_multicell_city_report(self):
    design_set = report_fixtures.make_design_set(
        n_candidates=3,
        cells=2,
        experiment_types=_GO_DARK_HEAVY_UP,
        dma=False,
        target_budget=None,
    )
    cities = report_fixtures.locations(12, dma=False)
    geojson = os.path.join(self.tmp, 'cities.geojson')
    _write_city_geojson(geojson, cities)
    targeting = pd.DataFrame({
        'location': [cities[0], cities[0], cities[1]],
        'postal_code': ['=10115', '10117', '20095'],
    })
    html = summarizer.DesignSummarizer(
        design_set,
        goals=summarizer.DesignGoals(target_mde=0.05, target_cpic=6.5),
        map_config=maps.byo_city_map_config(
            geojson, '276', 'Germany (city boundaries)'
        ),
        currency_symbol='€',
        targeting_map=targeting,
        test_start='2020-04-01',
        cooldown_days=0,
    ).generate_html()
    _save(html, 'design_multicell_polygon.html')
    self.assertIn('Treatment 1 · ads paused', html)
    self.assertIn('Treatment 2 · spend increased', html)
    self.assertIn('Incremental spend', html)
    self.assertIn('Download targeting list (CSV)', html)
    targeting_csvs = [
        v for k, v in _downloads(html).items() if k.endswith('targeting.csv')
    ]
    self.assertLen(targeting_csvs, 3)
    self.assertTrue(
        targeting_csvs[0].startswith('location,geo_name,group,postal_code\n')
    )
    self.assertIn("'=10115", ''.join(targeting_csvs))
    self.assertIn('Targeting CSV: your targeting map rows', html)
    self.assertIn('Treatment 1: Pause campaign spend in the', html)
    self.assertIn('Treatment 2: Increase campaign spend in the', html)
    self.assertIn('with no cooldown', html)
    self.assertIn('Types:', html)
    self.assertIn('Cells:', html)
    self.assertIn('If each incremental conversion costs your target iCPA', html)
    self.assertNotIn('Some geos are missing from the map', html)

  def test_low_budget_misses_goal(self):
    html = self._html(target_budget=500, target_cpic=1.0)
    self.assertIn('misses your budget goal', html)
    self.assertIn('Low chance', html)
    self.assertIn('To raise the chance of a conclusive result', html)

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

  @parameterized.parameters(
      dict(targeting_map=pd.DataFrame({'location': ['a']})),
      dict(
          targeting_map=pd.DataFrame(
              {'location': ['a'], 'geo_name': ['b']}
          )
      ),
      dict(test_start='2020-02-01'),
      dict(target_mde=2.0),
      dict(cooldown_days=-1),
  )
  def test_invalid_inputs(self, **kwargs):
    with self.assertRaises(ValueError):
      summarizer.DesignSummarizer(self.design_set, **kwargs)


class AnalysisSummarizerTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.tmp = self.enter_context(tempfile.TemporaryDirectory())
    self.topo = os.path.join(self.tmp, 'nielsentopo.json')
    _write_dma_topojson(self.topo)
    geos = report_fixtures.locations(10)
    self.design = report_fixtures.make_design(
        geos,
        {'cell_1': geos[:3]},
        data=report_fixtures.make_data(geos),
    )

  def _html(self, result, **kwargs) -> str:
    return summarizer.AnalysisSummarizer(result, **kwargs).generate_html()

  def test_lift_report(self):
    result = report_fixtures.make_analysis_result(self.design)
    html = self._html(
        result,
        map_config=maps.byo_dma_map_config(polygon_url=self.topo),
        data=report_fixtures.make_analysis_data(self.design),
        currency_symbol='$',
    )
    _save(html, 'analysis_lift_polygon.html')
    # v3 header chips.
    for text in (
        'GeoX experiment analysis', 'Status:', 'Test window complete',
        'Cooldown:', 'Regions split:', 'Holdback · ads show only',
        'Alpha:',
    ):
      self.assertIn(text, html)
    # Verdict banner, tiles and executive sections.
    # Headline leads with incremental conversions and iCPA (UX P0).
    self.assertIn('Yes: +1,000 extra conversions at $2.00 iCPA', html)
    self.assertIn('Each extra conversion cost about $2.00', html)
    self.assertIn('Did it drive extra conversions?', html)
    self.assertIn('Cost per extra conversion', html)
    self.assertIn('Campaign spend (implied)', html)
    self.assertIn('implied by the estimates, no interval', html)
    self.assertIn('<h3>What this means</h3>', html)
    self.assertIn('<h3>Next step</h3>', html)
    self.assertIn('calibrate your marketing mix model', html)
    self.assertIn('iCPA = campaign spend ÷ extra conversions.', html)
    # Geo tests avoid relative-lift terms (AL feedback).
    self.assertNotIn('Relative lift', html)
    self.assertNotIn('relative lift', html)
    self.assertIn('Lift (% vs. counterfactual)', html)
    # No planned spend target: neutral wording, no verification claim (UX P1).
    self.assertIn('Observed spend change', html)
    self.assertIn('not been verified against a planned', html)
    self.assertNotIn('Was the test run as planned?', html)
    self.assertIn('7-day cooldown', html)
    self.assertIn('analysis-outputs', html)
    # Analyst estimates: one row per metric, with a CSV download.
    self.assertIn('Estimates (90% CI in brackets)', html)
    self.assertIn('iCPA (cost per incremental conversion)', html)
    self.assertIn('geox_estimates.csv', html)
    # Detailed results equation strip and intros.
    self.assertIn('class="geox-equation"', html)
    self.assertIn('Observed in treatment geos', html)
    self.assertIn('Before the test start marker', html)
    self.assertIn('Where the test ran', html)
    self.assertTrue(
        summarizer.AnalysisSummarizer(result).to_iframe().startswith(
            '<iframe'
        )
    )
    # Objective, then the verdict banner, then What this means.
    self.assertLess(html.index('Objective:'), html.index('Yes: +1,000'))
    self.assertLess(html.index('Yes: +1,000'), html.index('What this means'))

  def test_no_lift(self):
    result = report_fixtures.make_analysis_result(
        self.design, lift=100, half_width=400
    )
    html = self._html(result)
    self.assertIn('No lift detected', html)
    self.assertIn('not statistically significant', html)
    self.assertIn('needs a significant lift', html)
    self.assertIn('not estimable', html)
    self.assertIn('plan a larger spend change', html)
    self.assertIn('No spend data was passed', html)

  def test_in_progress(self):
    result = report_fixtures.make_analysis_result(
        self.design, observed_days=14
    )
    html = self._html(result)
    _save(html, 'analysis_in_progress_polygon.html')
    self.assertIn('TOO EARLY', html)
    self.assertIn('In progress (day 14 of 30)', html)
    self.assertIn('Interim: ', html)
    self.assertIn('Results cover 14 of 30 planned test days', html)
    self.assertIn('Wait for the planned end', html)

  def test_significant_decrease_is_not_yes(self):
    result = report_fixtures.make_analysis_result(
        self.design, lift=-1_000, half_width=400
    )
    html = self._html(result)
    self.assertIn('NEEDS REVIEW', html)
    self.assertIn('Needs review: significant decrease', html)
    self.assertNotIn('Yes: ', html)

  def test_no_spend_data_is_not_checked(self):
    result = report_fixtures.make_analysis_result(self.design)
    html = self._html(result)
    self.assertIn('Spend execution not checked', html)

  def test_value_response(self):
    result = report_fixtures.make_analysis_result(self.design)
    html = self._html(result, response_is_value=True, currency_symbol='$')
    self.assertIn('iROAS', html)
    self.assertIn('extra conversion value', html)

  def _multicell(self):
    geos = report_fixtures.locations(10, dma=False)
    design = report_fixtures.make_design(
        geos,
        {'cell_1': geos[:2], 'cell_2': geos[2:4]},
        experiment_types=_GO_DARK_HEAVY_UP,
        target_budget=None,
    )
    return design, report_fixtures.make_analysis_result(design)

  def test_multicell(self):
    design, result = self._multicell()
    html = self._html(
        result, data=report_fixtures.make_analysis_data(design),
        currency_symbol='€',
    )
    _save(html, 'analysis_multicell_polygon.html')
    self.assertIn('Types:', html)
    self.assertIn('Treatment 1 go dark, Treatment 2 heavy-up', html)
    self.assertIn('Paused spend (implied)', html)
    self.assertIn('fewer conversions while ads were paused', html)
    self.assertIn('as a go-dark requires', html)
    self.assertIn('Counterfactual (expected had ads kept running)', html)
    self.assertIn('<th scope="col">Cell</th>', html)

  def test_multicell_execution_issue(self):
    design, result = self._multicell()
    issue_geo = sorted(design.designs['cell_1'].treatment_geos)[0]
    html = self._html(
        result,
        data=report_fixtures.make_analysis_data(
            design, spend_issue_geo=issue_geo
        ),
    )
    _save(html, 'analysis_multicell_execution_issue.html')
    self.assertIn('NEEDS REVIEW', html)
    self.assertIn('Spend did not change as planned', html)
    self.assertIn('⚠ Was the test run as planned?', html)
    self.assertIn('did not follow the plan', html)
    self.assertIn(issue_geo, html)

  def test_output_analysis_summary(self):
    result = report_fixtures.make_analysis_result(self.design)
    path = summarizer.AnalysisSummarizer(result).output_analysis_summary(
        'analysis.html', self.tmp
    )
    self.assertTrue(os.path.exists(path))

  def test_invalid_target_cpic(self):
    result = report_fixtures.make_analysis_result(self.design)
    with self.assertRaises(ValueError):
      summarizer.AnalysisSummarizer(result, target_cpic=0)

  def test_empty_results(self):
    result = report_fixtures.make_analysis_result(self.design)
    empty = api.AnalysisResult(
        results={}, analysis_config=result.analysis_config
    )
    with self.assertRaises(ValueError):
      summarizer.AnalysisSummarizer(empty)


if __name__ == '__main__':
  absltest.main()
