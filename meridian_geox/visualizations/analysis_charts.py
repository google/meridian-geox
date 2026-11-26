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

"""Charts for the GeoX analysis report."""

from typing import Optional

import altair as alt
from meridian_geox import api
from meridian_geox.visualizations import charts
from meridian_geox.visualizations import report_data
import pandas as pd

_WIDTH = charts.DEFAULT_WIDTH


def _frame_with_dates(df: pd.DataFrame) -> pd.DataFrame:
  frame = df.copy()
  frame.index = pd.to_datetime(frame.index)
  frame = frame.rename_axis('date').reset_index()
  return frame


def effect_totals_chart(
    result: api.AnalysisResult,
    labels: dict[str, str],
    cell: Optional[str] = None,
) -> alt.HConcatChart:
  """Plots counterfactual vs. observed totals next to the incremental effect.

  Left: total conversions in treatment geos over the analysis window, as
  expected without the change (counterfactual) and as observed. Right: the
  incremental effect with its confidence interval.

  Args:
    result: The analysis result.
    labels: Display label per cell.
    cell: Plot only this cell (default: all cells, one row each).

  Returns:
    The two-panel chart.
  """
  window = report_data.analysis_window(result)
  level = report_data.confidence_level(result)
  cells = [cell] if cell is not None else sorted(result.results)
  multi = len(cells) > 1
  totals, effects = [], []
  for name in cells:
    metrics = result.results[name]
    frame = _frame_with_dates(metrics.counterfactual_conversions)
    in_window = frame[(frame['date'] >= window.start) & (
        frame['date'] <= window.end)]
    group = labels.get(name, name)
    for series, column in (
        ('Counterfactual', 'counterfactual'),
        ('Observed', 'observed'),
    ):
      totals.append({
          'row': f'{group} · {series}' if multi else series,
          'series': series,
          'total': float(in_window[column].sum()),
      })
    lift = metrics.lift
    effects.append({
        'row': group if multi else 'Incremental',
        'point': float(lift.point_estimate),
        'lower': float(lift.lower_bound),
        'upper': float(lift.upper_bound),
    })
  bar_height = 26
  total_frame = pd.DataFrame(totals)
  bars = alt.Chart(total_frame).encode(
      y=alt.Y('row:N', title=None, sort=None),
      x=alt.X(
          'total:Q',
          title=f'Conversions in treatment geos, {window.observed_days} days',
          axis=alt.Axis(format='~s'),
      ),
  )
  left = alt.layer(
      bars.mark_bar(cornerRadiusEnd=3).encode(
          color=alt.Color(
              'series:N',
              legend=None,
              scale=alt.Scale(
                  domain=['Counterfactual', 'Observed'],
                  range=[charts.GREEN, charts.OBSERVED_COLOR],
              ),
          ),
          tooltip=[
              alt.Tooltip('row:N', title='Series'),
              alt.Tooltip('total:Q', title='Total', format=',.0f'),
          ],
      ),
      bars.mark_text(align='left', dx=4, fontSize=10).encode(
          text=alt.Text('total:Q', format='.4~s')
      ),
  ).properties(
      title='Counterfactual vs. observed',
      width=_WIDTH // 2,
      height=bar_height * len(totals),
  )
  effect_frame = pd.DataFrame(effects)
  base = alt.Chart(effect_frame).encode(
      y=alt.Y('row:N', title=None, sort=None)
  )
  right = alt.layer(
      base.mark_bar(color=charts.ORANGE, opacity=0.9, height=14).encode(
          x=alt.X(
              'point:Q', title=f'Incremental conversions ({level:.0%} CI)',
              axis=alt.Axis(format='~s'),
          ),
          tooltip=[
              alt.Tooltip('point:Q', title='Incremental', format=',.0f'),
              alt.Tooltip('lower:Q', title='Lower', format=',.0f'),
              alt.Tooltip('upper:Q', title='Upper', format=',.0f'),
          ],
      ),
      base.mark_rule(color=charts.GREY_900, strokeWidth=2).encode(
          x='lower:Q', x2='upper:Q'
      ),
      base.mark_tick(color=charts.GREY_900, thickness=1.5, size=14).encode(
          x='lower:Q'
      ),
      base.mark_tick(color=charts.GREY_900, thickness=1.5, size=14).encode(
          x='upper:Q'
      ),
      base.mark_text(align='left', dx=6, dy=-10, fontSize=10).encode(
          x='upper:Q', text=alt.Text('point:Q', format='.3~s')
      ),
      charts.rule(pd.DataFrame({'x': [0.0]}), charts.GREY_700, x='x:Q'),
  ).properties(
      title='Incremental = observed − counterfactual',
      width=_WIDTH // 2,
      height=max(bar_height * 2, bar_height * len(effects)),
  )
  return alt.hconcat(left, right, spacing=40)
