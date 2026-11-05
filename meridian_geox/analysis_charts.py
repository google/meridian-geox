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
from typing import Callable, Optional, Sequence

import altair as alt
from meridian_geox import api
from meridian_geox import charts
from meridian_geox import report_data
import numpy as np
import pandas as pd
from scipy import stats

_WIDTH = charts.DEFAULT_WIDTH
_HEIGHT = charts.DEFAULT_HEIGHT


def _phase_layers(window: report_data.AnalysisWindow) -> list[alt.Chart]:
  """Shaded test and cooldown periods."""
  rows = [{
      'start': window.start,
      'end': min(window.end, window.planned_end) + pd.Timedelta(hours=23),
      'phase': 'Test',
  }]
  if window.cooldown_days > 0:
    rows.append({
        'start': window.planned_end + pd.Timedelta(days=1),
        'end': window.end + pd.Timedelta(hours=23),
        'phase': 'Cooldown',
    })
  frame = pd.DataFrame(rows)
  return [
      alt.Chart(frame)
      .mark_rect(opacity=0.6)
      .encode(
          x='start:T',
          x2='end:T',
          color=alt.Color(
              'phase:N',
              title=None,
              scale=alt.Scale(
                  domain=['Test', 'Cooldown'],
                  range=[charts.TEST_COLOR, charts.COOLDOWN_COLOR],
              ),
          ),
          tooltip=[alt.Tooltip('phase:N', title='Period')],
      )
  ]


def _frame_with_dates(df: pd.DataFrame) -> pd.DataFrame:
  frame = df.copy()
  frame.index = pd.to_datetime(frame.index)
  frame = frame.rename_axis('date').reset_index()
  return frame


def observed_counterfactual_chart(
    metrics: api.AnalysisMetrics,
    window: report_data.AnalysisWindow,
    group_label: str = 'Treatment',
) -> alt.LayerChart:
  """Plots treatment conversions against the modelled counterfactual."""
  frame = _frame_with_dates(metrics.counterfactual_conversions)
  long = frame.melt(
      id_vars=['date'],
      value_vars=['observed', 'counterfactual'],
      var_name='series',
      value_name='conversions',
  )
  long['series'] = long['series'].map({
      'observed': f'{group_label} (observed)',
      'counterfactual': 'Counterfactual (no ads change)',
  })
  band = (
      alt.Chart(frame.dropna(subset=['lower_bound', 'upper_bound']))
      .mark_area(color=charts.CI_COLOR, opacity=0.5)
      .encode(
          x='date:T',
          y='lower_bound:Q',
          y2='upper_bound:Q',
          tooltip=[
              alt.Tooltip('date:T'),
              alt.Tooltip('lower_bound:Q', title='CI lower', format=',.0f'),
              alt.Tooltip('upper_bound:Q', title='CI upper', format=',.0f'),
          ],
      )
  )
  lines = (
      alt.Chart(long)
      .mark_line(strokeWidth=1.8)
      .encode(
          x=alt.X('date:T', title=None),
          y=alt.Y('conversions:Q', title='Daily conversions',
                  scale=alt.Scale(zero=False)),
          color=alt.Color(
              'series:N',
              title=None,
              scale=alt.Scale(
                  range=[charts.COUNTERFACTUAL_COLOR, charts.OBSERVED_COLOR]
              ),
          ),
          strokeDash=alt.StrokeDash(
              'series:N', legend=None,
              scale=alt.Scale(range=[[5, 3], [1, 0]]),
          ),
          tooltip=[
              alt.Tooltip('date:T'),
              alt.Tooltip('series:N'),
              alt.Tooltip('conversions:Q', format=',.0f'),
          ],
      )
  )
  return (
      alt.layer(*_phase_layers(window), band, lines)
      .resolve_scale(color='independent')
      .properties(width=_WIDTH * 2, height=_HEIGHT)
  )


def pointwise_chart(
    metrics: api.AnalysisMetrics, window: report_data.AnalysisWindow
) -> alt.LayerChart:
  """Plots the daily gap between observed and counterfactual."""
  frame = _frame_with_dates(metrics.pointwise_difference)
  frame = frame[frame['date'] >= window.start - pd.Timedelta(days=28)]
  band = (
      alt.Chart(frame.dropna(subset=['lower_bound', 'upper_bound']))
      .mark_area(color=charts.CI_COLOR, opacity=0.5)
      .encode(x='date:T', y='lower_bound:Q', y2='upper_bound:Q')
  )
  line = (
      alt.Chart(frame)
      .mark_line(color=charts.OBSERVED_COLOR, strokeWidth=1.8)
      .encode(
          x=alt.X('date:T', title=None),
          y=alt.Y('difference:Q', title='Daily extra conversions'),
          tooltip=[
              alt.Tooltip('date:T'),
              alt.Tooltip('difference:Q', title='Gap', format=',.1f'),
          ],
      )
  )
  zero = alt.Chart(pd.DataFrame({'y': [0]})).mark_rule(
      color=charts.GREY_500).encode(y='y:Q')
  return (
      alt.layer(*_phase_layers(window), band, zero, line)
      .resolve_scale(color='independent')
      .properties(width=_WIDTH, height=_HEIGHT)
  )


def _cumulative_chart(
    frame: pd.DataFrame,
    value_column: str,
    title: str,
    value_format: str,
    window: report_data.AnalysisWindow,
) -> alt.LayerChart:
  frame = _frame_with_dates(frame).replace([np.inf, -np.inf], np.nan)
  band = (
      alt.Chart(frame.dropna(subset=['lower_bound', 'upper_bound']))
      .mark_area(color=charts.CI_COLOR, opacity=0.5)
      .encode(x='date:T', y='lower_bound:Q', y2='upper_bound:Q')
  )
  line = (
      alt.Chart(frame)
      .mark_line(color=charts.OBSERVED_COLOR, strokeWidth=2)
      .encode(
          x=alt.X('date:T', title=None),
          y=alt.Y(f'{value_column}:Q', title=title),
          tooltip=[
              alt.Tooltip('date:T'),
              alt.Tooltip(f'{value_column}:Q', title=title,
                          format=value_format),
              alt.Tooltip('lower_bound:Q', title='CI lower',
                          format=value_format),
              alt.Tooltip('upper_bound:Q', title='CI upper',
                          format=value_format),
          ],
      )
  )
  zero = alt.Chart(pd.DataFrame({'y': [0]})).mark_rule(
      color=charts.GREY_500).encode(y='y:Q')
  return (
      alt.layer(*_phase_layers(window), band, zero, line)
      .resolve_scale(color='independent')
      .properties(width=_WIDTH, height=_HEIGHT)
  )


def cumulative_chart(
    metrics: api.AnalysisMetrics, window: report_data.AnalysisWindow
) -> alt.LayerChart:
  """Plots cumulative incremental conversions with their CI."""
  return _cumulative_chart(
      metrics.cumulative_lift,
      'lift',
      'Cumulative extra conversions',
      ',.0f',
      window,
  )


def cumulative_icpd_chart(
    metrics: api.AnalysisMetrics, window: report_data.AnalysisWindow
) -> Optional[alt.LayerChart]:
  """Plots cumulative iCPD (the statistical estimand) with its CI."""
  if metrics.cumulative_icpd is None or metrics.cumulative_icpd.empty:
    return None
  return _cumulative_chart(
      metrics.cumulative_icpd,
      'icpd',
      'Cumulative iCPD (conversions per unit spend)',
      ',.3f',
      window,
  )


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
      alt.Chart(pd.DataFrame({'x': [0.0]})).mark_rule(
          color=charts.GREY_700
      ).encode(x='x:Q'),
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

  Returns:
    The chart, or None if the estimate has no usable standard deviation.
  """
  point = estimate.point_estimate
  sd = _sd_from_interval(estimate, level)
  if sd is None:
    sd = estimate.standard_deviation
  if not sd or math.isnan(sd) or sd <= 0 or math.isnan(point):
    return None
  grid = np.linspace(point - 4 * sd, point + 4 * sd, 121)
  decreasing = False
  if transform is not None:
    ends = [transform(float(grid[0])), transform(float(grid[-1]))]
    if all(e is not None and math.isfinite(e) for e in ends):
      decreasing = ends[1] < ends[0]
  rows = []
  for x in grid:
    value = transform(float(x)) if transform else float(x)
    if value is None or math.isnan(value) or math.isinf(value):
      continue
    # P(estimate > x); a decreasing transform flips the inequality.
    above = float(1 - stats.norm.cdf((x - point) / sd))
    rows.append({
        'value': value,
        'confidence': 1 - above if decreasing else above,
    })
  if not rows:
    return None
  frame = pd.DataFrame(rows)
  line = (
      alt.Chart(frame)
      .mark_line(color=charts.OBSERVED_COLOR, strokeWidth=2)
      .encode(
          x=alt.X('value:Q', title=x_title, axis=alt.Axis(format=value_format)),
          y=alt.Y('confidence:Q', title='Confidence the true value is above',
                  axis=alt.Axis(format='.0%'), scale=alt.Scale(domain=[0, 1])),
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


def spend_pacing_chart(
    spend: pd.DataFrame,
    window: report_data.AnalysisWindow,
    group_labels: Sequence[str],
) -> Optional[alt.LayerChart]:
  """Plots daily spend of treatment vs. control geos."""
  if spend.empty:
    return None
  frame = spend[spend['date'] >= window.start - pd.Timedelta(days=28)]
  line = (
      alt.Chart(frame)
      .mark_line(strokeWidth=1.8)
      .encode(
          x=alt.X('date:T', title=None),
          y=alt.Y('spend:Q', title='Daily spend'),
          color=alt.Color(
              'group:N', title=None,
              scale=charts.group_scale(list(group_labels)),
          ),
          tooltip=[
              alt.Tooltip('date:T'),
              alt.Tooltip('group:N'),
              alt.Tooltip('spend:Q', format=',.0f'),
          ],
      )
  )
  return (
      alt.layer(*_phase_layers(window), line)
      .resolve_scale(color='independent')
      .properties(width=_WIDTH, height=_HEIGHT)
  )


def anomaly_heatmap(
    z_scores: pd.DataFrame,
    names: report_data.GeoNames,
    window: Optional[report_data.AnalysisWindow] = None,
) -> Optional[alt.LayerChart]:
  """Plots per-geo robust z-scores of daily conversions."""
  if z_scores.empty:
    return None
  frame = z_scores.copy()
  frame['geo_name'] = [names(loc) for loc in frame['location']]
  frame['z_clipped'] = frame['z'].clip(-5, 5)
  heat = (
      alt.Chart(frame)
      .mark_rect()
      .encode(
          x=alt.X('date:T', title=None),
          y=alt.Y('geo_name:N', title=None),
          color=alt.Color(
              'z_clipped:Q',
              title='Robust z',
              scale=alt.Scale(scheme='redblue', domain=[-5, 5], reverse=True),
          ),
          tooltip=[
              alt.Tooltip('geo_name:N', title='Geo'),
              alt.Tooltip('date:T'),
              alt.Tooltip('conversions:Q', format=',.0f'),
              alt.Tooltip('z:Q', title='Robust z', format='.1f'),
          ],
      )
  )
  layers = [heat]
  if window is not None:
    layers.append(
        alt.Chart(pd.DataFrame({'date': [window.start]}))
        .mark_rule(color=charts.GREY_900)
        .encode(x='date:T')
    )
  return alt.layer(*layers).properties(
      width=_WIDTH * 2, height=max(120, 14 * frame['location'].nunique())
  )
