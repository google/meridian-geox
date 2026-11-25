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

import math
from typing import Callable, Optional

import altair as alt
from meridian_geox import api
from meridian_geox import charts
from meridian_geox import report_data
import numpy as np
import pandas as pd
from scipy import stats

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


def confidence_curve(
    estimate: api.Estimate,
    x_title: str,
    value_format: str,
    transform: Optional[Callable[[float], float]] = None,
    level: float = 0.9,
    decreasing: bool = False,
) -> Optional[alt.LayerChart]:
  """Plots the confidence that the effect exceeds each value.

  The spread is taken from the reported (placebo-based) confidence interval,
  so the curve passes through the interval bounds at the chosen level; the
  placebo standard deviation is the fallback when the interval is infinite.

  Args:
    estimate: The estimate.
    x_title: Axis title.
    value_format: d3 format of the values.
    transform: Optional monotone map from the estimate's scale to the
      displayed scale (for example iCPD -> iCPA). For a decreasing map the
      curve still shows the confidence that the displayed value is above x.
    level: Confidence level drawn as a band.
    decreasing: Whether `transform` is decreasing on positive values (for
      example the reciprocal iCPD -> iCPA). The displayed value is then above
      x exactly when the estimate is between 0 and transform^-1(x), and only
      positive estimates are plotted.

  Returns:
    The chart, or None if the estimate has no usable standard deviation.
  """
  point = estimate.point_estimate
  sd = _sd_from_interval(estimate, level)
  if sd is None:
    sd = estimate.standard_deviation
  if not sd or math.isnan(sd) or sd <= 0 or math.isnan(point):
    return None
  low, high = point - 4 * sd, point + 4 * sd
  if decreasing:
    # Only positive estimates map to a finite displayed value.
    low = max(low, sd / 50)
    if high <= low:
      return None
  grid = np.linspace(low, high, 121)
  below_zero = float(stats.norm.cdf(-point / sd))
  rows = []
  for x in grid:
    value = transform(float(x)) if transform else float(x)
    if value is None or math.isnan(value) or math.isinf(value):
      continue
    at_most = float(stats.norm.cdf((x - point) / sd))
    # Increasing: P(estimate > x). Decreasing (positive estimates only):
    # P(transform(estimate) >= transform(x)) = P(0 < estimate <= x).
    confidence = (at_most - below_zero) if decreasing else 1 - at_most
    rows.append({'value': value, 'confidence': max(confidence, 0.0)})
  if not rows:
    return None
  frame = pd.DataFrame(rows)
  line = (
      alt.Chart(frame)
      .mark_line(color=charts.OBSERVED_COLOR, strokeWidth=2)
      .encode(
          x=alt.X('value:Q', title=x_title, axis=alt.Axis(format=value_format)),
          y=charts.percent_y(
              'confidence:Q', 'Confidence the true value is above'
          ),
          tooltip=[
              alt.Tooltip('value:Q', format=value_format),
              alt.Tooltip('confidence:Q', format='.0%'),
          ],
      )
  )
  tail = (1 - level) / 2
  rules = alt.Chart(pd.DataFrame({'y': [tail, 1 - tail]})).mark_rule(
      color=charts.GREY_500, strokeDash=[3, 3]).encode(y='y:Q')
  return alt.layer(rules, line).properties(width=_WIDTH // 2 + 40,
                                           height=200)


def _sd_from_interval(estimate: api.Estimate, level: float) -> Optional[float]:
  """Standard deviation implied by a finite two-sided interval."""
  lower, upper = estimate.lower_bound, estimate.upper_bound
  if any(math.isnan(v) or math.isinf(v) for v in (lower, upper)):
    return None
  width = abs(upper - lower)
  if width <= 0:
    return None
  return width / (2 * float(stats.norm.ppf(1 - (1 - level) / 2)))
