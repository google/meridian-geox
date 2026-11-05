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

"""Charts for the GeoX design report."""

import json
import math
from typing import Optional, Sequence

import altair as alt
from meridian_geox import api
from meridian_geox import charts
from meridian_geox import report_data
import numpy as np
import pandas as pd
from scipy import stats

_WIDTH = charts.DEFAULT_WIDTH
_HEIGHT = charts.DEFAULT_HEIGHT


def _selected_expr() -> str:
  return f'datum.design_id == {charts.DESIGN_SELECTION}'


def _currency_format(currency_symbol: str) -> str:
  return f'{currency_symbol},.0f' if not currency_symbol else ',.0f'


def tradeoff_chart(
    summaries: Sequence[report_data.CandidateSummary],
    goals: report_data.DesignGoals,
    y_axis: str = 'mde',
    currency_symbol: str = '',
    spend_label: str = 'Spend',
) -> alt.LayerChart:
  """Plots spend vs. detectable effect for every candidate.

  Args:
    summaries: Ranked candidate summaries.
    goals: Design goals; set goals add goal lines and an ideal zone.
    y_axis: `'mde'` for relative MDE or `'cpic'` for max detectable iCPA.
    currency_symbol: Currency symbol for axis titles.
    spend_label: Name of the spend amount (for example "Campaign spend").

  Returns:
    The chart. Points carry `design_id`, so clicking one selects it.
  """
  if y_axis not in ('mde', 'cpic'):
    raise ValueError(f"y_axis must be 'mde' or 'cpic', got {y_axis!r}.")
  rows = pd.DataFrame([
      {
          'design_id': s.design_id,
          'option': s.name,
          'short_id': s.short_id,
          'rank': s.rank,
          'spend': s.total_spend,
          'mde': s.max_mde,
          'cpic': (
              None if math.isnan(s.max_detectable_cpic)
              else s.max_detectable_cpic
          ),
          'status': 'Meets goals' if s.feasible else 'Misses a goal',
      }
      for s in summaries
  ])
  money = f' ({currency_symbol})' if currency_symbol else ''
  if y_axis == 'mde':
    y = alt.Y(
        'mde:Q', title='Detectable lift (relative MDE)', axis=alt.Axis(
            format='.1%'), scale=alt.Scale(zero=False)
    )
    y_goal = goals.target_mde
  else:
    y = alt.Y(
        'cpic:Q', title=f'Max detectable iCPA{money}',
        scale=alt.Scale(zero=False),
    )
    y_goal = goals.target_cpic
  x = alt.X('spend:Q', title=f'{spend_label}{money}', scale=alt.Scale(
      zero=False))

  layers = []
  if goals.target_budget is not None and y_goal is not None:
    max_y = float(rows[y_axis].max()) if rows[y_axis].notna().any() else y_goal
    zone = pd.DataFrame([{
        'x0': float(rows['spend'].min()) * 0.98,
        'x1': goals.target_budget,
        'y0': (float(rows[y_axis].min()) * 0.98 if y_axis == 'mde'
               else y_goal),
        'y1': y_goal if y_axis == 'mde' else max(max_y, y_goal) * 1.02,
    }])
    layers.append(
        alt.Chart(zone)
        .mark_rect(color=charts.GREEN, opacity=0.08)
        .encode(x='x0:Q', x2='x1:Q', y='y0:Q', y2='y1:Q')
    )
  if goals.target_budget is not None:
    layers.append(
        alt.Chart(pd.DataFrame({'x': [goals.target_budget]}))
        .mark_rule(color=charts.GREEN, strokeDash=[4, 4])
        .encode(x='x:Q')
    )
  if y_goal is not None:
    layers.append(
        alt.Chart(pd.DataFrame({'y': [y_goal]}))
        .mark_rule(color=charts.GREEN, strokeDash=[4, 4])
        .encode(y='y:Q')
    )
  base = alt.Chart(rows)
  points = base.mark_point(filled=True, size=140, cursor='pointer').encode(
      x=x,
      y=y,
      color=alt.Color(
          'status:N',
          title=None,
          scale=alt.Scale(
              domain=['Meets goals', 'Misses a goal'],
              range=[charts.FEASIBLE_COLOR, charts.INFEASIBLE_COLOR],
          ),
      ),
      stroke=alt.condition(
          _selected_expr(), alt.value(charts.GREY_900), alt.value('white')
      ),
      strokeWidth=alt.condition(_selected_expr(), alt.value(3), alt.value(1)),
      tooltip=[
          alt.Tooltip('option:N', title='Option'),
          alt.Tooltip('short_id:N', title='ID'),
          alt.Tooltip('spend:Q', title=spend_label, format=',.0f'),
          alt.Tooltip('mde:Q', title='Relative MDE', format='.2%'),
          alt.Tooltip('cpic:Q', title='Max detectable iCPA', format=',.2f'),
          alt.Tooltip('status:N', title='Goals'),
      ],
  )
  labels = base.mark_text(dy=-12, fontSize=10, color=charts.GREY_700).encode(
      x=x, y=y, text=alt.Text('rank:Q', format='d')
  )
  layers += [points, labels]
  default_id = summaries[0].design_id if summaries else ''
  return (
      alt.layer(*layers)
      .add_params(charts.selection_param(default_id))
      .properties(width=_WIDTH, height=_HEIGHT)
  )


def _z_alpha(alpha: float, test_type: api.TestType) -> float:
  tail = alpha / 2 if test_type == api.TestType.TWO_SIDED else alpha
  return float(stats.norm.ppf(1 - tail))


def _design_params(design: api.Design) -> tuple[float, float, api.TestType]:
  config = design.design_config
  if config is None:
    return 0.1, 0.8, api.TestType.TWO_SIDED
  return config.alpha, config.power, config.test_type


def sensitivity_curve(
    summary: report_data.CandidateSummary,
    design: api.Design,
    target_mde: Optional[float] = None,
) -> alt.LayerChart:
  """Plots the chance of detecting each true lift size."""
  alpha, power, test_type = _design_params(design)
  z_alpha = _z_alpha(alpha, test_type)
  z_power = float(stats.norm.ppf(power))
  max_lift = 3 * summary.max_mde
  if target_mde is not None:
    # Keep the goal line inside the chart.
    max_lift = max(max_lift, 1.15 * target_mde)
  lifts = np.linspace(0, max_lift, 61)
  rows = []
  for cell in summary.cells.values():
    sigma = cell.mde / (z_alpha + z_power)
    for lift in lifts:
      rows.append({
          'lift': float(lift),
          'power': float(stats.norm.cdf(lift / sigma - z_alpha)),
          'group': cell.label,
      })
  frame = pd.DataFrame(rows)
  labels = [c.label for c in summary.cells.values()]
  line = (
      alt.Chart(frame)
      .mark_line(strokeWidth=2.5)
      .encode(
          x=alt.X(
              'lift:Q',
              title='True lift',
              # One decimal so small MDEs don't repeat labels (1%, 1%, …).
              axis=alt.Axis(format='.1%', labelFlush=True, tickMinStep=0.001),
              scale=alt.Scale(domain=[0, float(max_lift)], nice=False),
          ),
          y=alt.Y(
              'power:Q',
              title='Chance of detecting it',
              axis=alt.Axis(format='.0%'),
              scale=alt.Scale(domain=[0, 1]),
          ),
          color=alt.Color(
              'group:N',
              title=None,
              scale=charts.group_scale(labels),
              legend=alt.Legend(values=labels) if len(labels) > 1 else None,
          ),
          tooltip=[
              alt.Tooltip('group:N', title='Group'),
              alt.Tooltip('lift:Q', title='True lift', format='.1%'),
              alt.Tooltip('power:Q', title='Chance', format='.0%'),
          ],
      )
  )
  power_rule = (
      alt.Chart(pd.DataFrame({'y': [power]}))
      .mark_rule(color=charts.GREY_500, strokeDash=[2, 2])
      .encode(y='y:Q')
  )
  mde_rule = (
      alt.Chart(pd.DataFrame({'x': [summary.max_mde], 'label': ['MDE']}))
      .mark_rule(color=charts.GREY_700)
      .encode(x='x:Q', tooltip=[alt.Tooltip('x:Q', title='MDE',
                                            format='.2%')])
  )
  # Shade lifts the test confirms with at least the design power.
  zone = (
      alt.Chart(pd.DataFrame([{
          'x0': summary.max_mde, 'x1': float(max_lift), 'y0': power, 'y1': 1.0,
          'label': f'High-confidence zone (≥ {power:.0%} chance)',
      }]))
      .mark_rect(color=charts.GREEN, opacity=0.08)
      .encode(x='x0:Q', x2='x1:Q', y='y0:Q', y2='y1:Q',
              tooltip=[alt.Tooltip('label:N', title='Zone')])
  )
  zone_label = (
      alt.Chart(pd.DataFrame([{
          'x': summary.max_mde, 'y': 1.0,
          'text': f'High-confidence zone (≥ {power:.0%} chance)',
      }]))
      .mark_text(align='left', baseline='top', dx=4, dy=4, fontSize=10,
                 color=charts.GREEN)
      .encode(x='x:Q', y='y:Q', text='text:N')
  )
  point = (
      alt.Chart(pd.DataFrame([{
          'x': summary.max_mde, 'y': power,
          'text': f'Detectable lift: {summary.max_mde:.1%}',
      }]))
  )
  mde_point = point.mark_point(
      filled=True, size=80, color=charts.BLUE
  ).encode(x='x:Q', y='y:Q')
  mde_label = point.mark_text(
      align='left', dx=8, dy=-8, fontSize=11, color=charts.BLUE
  ).encode(x='x:Q', y='y:Q', text='text:N')
  layers = [zone, zone_label, power_rule, mde_rule, line, mde_point,
            mde_label]
  if target_mde is not None:
    goal = pd.DataFrame([{
        'x': target_mde, 'text': f'Goal ({target_mde:.1%})'
    }])
    layers.insert(
        0,
        alt.Chart(goal)
        .mark_rule(color=charts.GREEN, strokeDash=[4, 4])
        .encode(x='x:Q'),
    )
    layers.append(
        alt.Chart(goal)
        .mark_text(align='left', dx=4, dy=-6, fontSize=10,
                   color=charts.GREEN)
        .encode(x='x:Q', y=alt.value(_HEIGHT), text='text:N')
    )
  return alt.layer(*layers).properties(width=_WIDTH, height=_HEIGHT)


def pretest_fit_chart(design: api.Design) -> alt.LayerChart:
  """Plots observed vs. counterfactual pretest conversions per cell."""
  labels = report_data.group_labels(design)
  frames = []
  for cell, per_cell in design.designs.items():
    df = per_cell.counterfactual_conversions
    if df is None or df.empty:
      continue
    date_col = 'index' if 'index' in df.columns else df.columns[0]
    for series, column in (
        ('Observed', 'observed'),
        ('Predicted from control', 'counterfactual'),
    ):
      frames.append(
          pd.DataFrame({
              'date': pd.to_datetime(df[date_col]),
              'value': df[column].astype(float),
              'series': series,
              'group': labels[cell],
          })
      )
  frame = (
      pd.concat(frames, ignore_index=True)
      if frames
      else pd.DataFrame(columns=['date', 'value', 'series', 'group'])
  )
  label_list = list(labels.values())
  lines = (
      alt.Chart(frame)
      .mark_line(strokeWidth=1.8)
      .encode(
          x=alt.X('date:T', title=None),
          y=alt.Y('value:Q', title='Daily conversions', scale=alt.Scale(
              zero=False)),
          color=alt.Color(
              'group:N', title=None, scale=charts.group_scale(label_list)
          ),
          strokeDash=alt.StrokeDash(
              'series:N',
              title=None,
              scale=alt.Scale(
                  domain=['Observed', 'Predicted from control'],
                  range=[[1, 0], [5, 3]],
              ),
          ),
          tooltip=[
              alt.Tooltip('date:T'),
              alt.Tooltip('group:N'),
              alt.Tooltip('series:N'),
              alt.Tooltip('value:Q', format=',.0f'),
          ],
      )
  )
  layers = [lines]
  excluded = sorted(pd.to_datetime(list(design.excluded_dates or ())))
  if excluded:
    layers.insert(
        0,
        alt.Chart(pd.DataFrame({'date': excluded}))
        .mark_rule(color=charts.RED, opacity=0.25)
        .encode(x='date:T'),
    )
  return alt.layer(*layers).properties(width=_WIDTH, height=_HEIGHT)


def aa_check_chart(
    summary: report_data.CandidateSummary, alpha: float
) -> alt.LayerChart:
  """Plots the A/A test p-value of each cell against alpha."""
  frame = pd.DataFrame([
      {
          'group': c.label,
          'p_value': c.aa_p_value,
          'result': 'Pass' if c.aa_p_value >= alpha else 'Fail',
      }
      for c in summary.cells.values()
  ])
  bars = (
      alt.Chart(frame)
      .mark_bar(height=18)
      .encode(
          y=alt.Y('group:N', title=None),
          x=alt.X(
              'p_value:Q',
              title='A/A test p-value (higher is better)',
              scale=alt.Scale(domain=[0, 1]),
          ),
          color=alt.Color(
              'result:N',
              title=None,
              scale=alt.Scale(
                  domain=['Pass', 'Fail'], range=[charts.GREEN, charts.RED]
              ),
          ),
          tooltip=[
              alt.Tooltip('group:N'),
              alt.Tooltip('p_value:Q', format='.3f'),
              alt.Tooltip('result:N'),
          ],
      )
  )
  rule = (
      alt.Chart(pd.DataFrame({'x': [alpha]}))
      .mark_rule(color=charts.GREY_700, strokeDash=[4, 4])
      .encode(x='x:Q')
  )
  return alt.layer(bars, rule).properties(
      width=_WIDTH, height=max(40, 30 * len(frame))
  )


def largest_geos_chart(
    geo_table: pd.DataFrame, color_scale: alt.Scale, top_n: int = 22
) -> alt.Chart:
  """Plots the largest geos with their share of conversions and of spend.

  Solid bars are the share of pretest conversions; light bars the share of
  pretest spend, so it is easy to see whether spend follows conversions.

  Args:
    geo_table: Output of `report_data.geo_table`.
    color_scale: Colour scale over group labels.
    top_n: Number of largest geos to show.

  Returns:
    The chart.
  """
  frame = geo_table[geo_table['cell'] != 'excluded']
  frame = frame.sort_values('share', ascending=False).head(top_n)
  has_spend = 'spend_share' in frame and float(frame['spend_share'].sum()) > 0
  columns = ['geo_name', 'group', 'share'] + (
      ['spend_share'] if has_spend else []
  )
  order = list(frame['geo_name'])
  base = alt.Chart(frame[columns])
  if has_spend:
    base = base.transform_fold(
        ['share', 'spend_share'], as_=['measure', 'value']
    ).transform_calculate(
        measure_label=(
            "datum.measure == 'share' ? 'Conversions' : 'Spend'"
        )
    )
    value = 'value:Q'
  else:
    base = base.transform_calculate(measure_label="'Conversions'")
    value = 'share:Q'
  return (
      base.mark_bar()
      .encode(
          y=alt.Y('geo_name:N', title=None, sort=order),
          yOffset=alt.YOffset(
              'measure_label:N', sort=['Conversions', 'Spend']
          ),
          x=alt.X(
              value, title='Share of pretest conversions (solid) and spend',
              axis=alt.Axis(format='.1%'),
          ),
          color=alt.Color('group:N', title=None, scale=color_scale),
          opacity=alt.Opacity(
              'measure_label:N',
              scale=alt.Scale(
                  domain=['Conversions', 'Spend'], range=[1.0, 0.35]
              ),
              title=None,
          ),
          tooltip=[
              alt.Tooltip('geo_name:N', title='Geo'),
              alt.Tooltip('group:N', title='Group'),
              alt.Tooltip('measure_label:N', title='Measure'),
              alt.Tooltip(value, title='Share', format='.2%'),
          ],
      )
      .properties(width=_WIDTH, height=max(80, 22 * len(frame)))
  )


def tradeoff_toggle_chart(
    summaries: Sequence[report_data.CandidateSummary],
    goals: report_data.DesignGoals,
    currency_symbol: str = '',
    spend_label: str = 'Spend',
) -> alt.LayerChart:
  """Spend vs. detectable effect with a chip toggle for the y-axis.

  The toggle switches between relative MDE (lower is better) and max
  detectable iCPA (higher is better). Points carry `design_id`, so clicking
  one selects that candidate.

  Args:
    summaries: Ranked candidate summaries.
    goals: Design goals for goal lines.
    currency_symbol: Currency symbol for labels.
    spend_label: Name of the spend amount.

  Returns:
    The chart.
  """
  rows = pd.DataFrame([
      {
          'design_id': s.design_id,
          'option': ('★ ' if s.rank == 1 else '') + f'Option {s.rank}',
          'short_id': s.short_id,
          'spend': s.total_spend,
          'mde': s.max_mde,
          'cpic': (
              None if math.isnan(s.max_detectable_cpic)
              else s.max_detectable_cpic
          ),
          'share': sum(c.treatment_share for c in s.cells.values()),
          'status': 'Meets goals' if s.feasible else 'Misses a goal',
      }
      for s in summaries
  ])
  mde_label = 'Detectable lift, MDE % (lower is better)'
  cpic_label = 'Max detectable iCPA (higher is better)'
  axis = alt.param(
      name='tradeoff_axis',
      value=mde_label,
      bind=alt.binding_radio(options=[mde_label, cpic_label], name='Show: '),
  )
  is_mde = f"tradeoff_axis == '{mde_label}'"
  y_expr = f'{is_mde} ? datum.mde : datum.cpic'
  money = f' ({currency_symbol})' if currency_symbol else ''
  label_expr = (
      f"{is_mde} ? format(datum.value, '.2%') : "
      f"'{currency_symbol}' + format(datum.value, ',.2f')"
  )
  y = alt.Y(
      'y:Q', title='Selected metric', scale=alt.Scale(zero=False),
      axis=alt.Axis(labelExpr=label_expr),
  )
  x = alt.X(
      'spend:Q', title=f'{spend_label} needed{money}',
      scale=alt.Scale(zero=False),
  )
  base = alt.Chart(rows).transform_calculate(y=y_expr)
  layers = []
  if goals.target_budget is not None:
    layers.append(
        alt.Chart(pd.DataFrame({'x': [goals.target_budget]}))
        .mark_rule(color=charts.GREEN, strokeDash=[4, 4])
        .encode(x='x:Q')
    )
  goal_rows = []
  if goals.target_mde is not None:
    goal_rows.append({'metric': 'mde', 'y': goals.target_mde})
  if goals.target_cpic is not None:
    goal_rows.append({'metric': 'cpic', 'y': goals.target_cpic})
  if goal_rows:
    layers.append(
        alt.Chart(pd.DataFrame(goal_rows))
        .transform_filter(
            f"({is_mde} && datum.metric == 'mde') || "
            f"(!({is_mde}) && datum.metric == 'cpic')"
        )
        .mark_rule(color=charts.GREEN, strokeDash=[4, 4])
        .encode(y='y:Q')
    )
  points = base.mark_point(filled=True, size=160, cursor='pointer').encode(
      x=x,
      y=y,
      color=alt.Color(
          'status:N',
          title=None,
          scale=alt.Scale(
              domain=['Meets goals', 'Misses a goal'],
              range=[charts.FEASIBLE_COLOR, charts.INFEASIBLE_COLOR],
          ),
      ),
      stroke=alt.condition(
          _selected_expr(), alt.value(charts.GREY_900), alt.value('white')
      ),
      strokeWidth=alt.condition(_selected_expr(), alt.value(3), alt.value(1)),
      tooltip=[
          alt.Tooltip('option:N', title='Option'),
          alt.Tooltip('short_id:N', title='ID'),
          alt.Tooltip('spend:Q', title=spend_label, format=',.0f'),
          alt.Tooltip('mde:Q', title='Relative MDE', format='.2%'),
          alt.Tooltip('cpic:Q', title='Max detectable iCPA', format=',.2f'),
          alt.Tooltip('share:Q', title='Treatment share', format='.0%'),
          alt.Tooltip('status:N', title='Goals'),
      ],
  )
  labels = base.transform_calculate(
      label="datum.option + ' · ' + format(datum.share, '.0%')"
  ).mark_text(dy=-13, fontSize=10, color=charts.GREY_700).encode(
      x=x, y=y, text='label:N'
  )
  layers += [points, labels]
  default_id = summaries[0].design_id if summaries else ''
  return (
      alt.layer(*layers)
      .add_params(axis, charts.selection_param(default_id))
      .properties(width=_WIDTH, height=_HEIGHT)
  )


def strata_chart(
    geo_table: pd.DataFrame, color_scale: alt.Scale
) -> Optional[alt.Chart]:
  """Plots how each stratum's geos split into groups (stratified designs)."""
  if 'stratum' not in geo_table or geo_table['stratum'].isna().all():
    return None
  frame = geo_table[
      (geo_table['cell'] != 'excluded') & geo_table['stratum'].notna()
  ].copy()
  frame['stratum_label'] = [f'Stratum {int(s)}' for s in frame['stratum']]
  counts = (
      frame.groupby(['stratum_label', 'group'])
      .agg(geos=('location', 'count'), share=('share', 'sum'))
      .reset_index()
  )
  return (
      alt.Chart(counts)
      .mark_bar()
      .encode(
          y=alt.Y('stratum_label:N', title=None),
          x=alt.X('geos:Q', title='Geos', stack='zero'),
          color=alt.Color('group:N', title=None, scale=color_scale),
          tooltip=[
              alt.Tooltip('stratum_label:N', title='Stratum'),
              alt.Tooltip('group:N', title='Group'),
              alt.Tooltip('geos:Q', title='Geos'),
              alt.Tooltip('share:Q', title='Conversion share', format='.1%'),
          ],
      )
      .properties(
          width=_WIDTH, height=max(60, 26 * counts['stratum_label'].nunique())
      )
  )


def aa_drift_chart(
    drift: report_data.AaDrift, mde_abs: Optional[float]
) -> alt.LayerChart:
  """Cumulative observed minus predicted conversions in the A/A window."""
  line = (
      alt.Chart(drift.frame)
      .mark_line(color=charts.BLUE, strokeWidth=2)
      .encode(
          x=alt.X('date:T', title=None),
          y=alt.Y(
              'drift:Q', title='Cumulative conversions (observed − expected)'
          ),
          tooltip=[
              alt.Tooltip('date:T'),
              alt.Tooltip('drift:Q', title='Drift', format='+,.0f'),
          ],
      )
  )
  layers = [
      alt.Chart(pd.DataFrame({'y': [0.0]}))
      .mark_rule(color=charts.GREY_500)
      .encode(y='y:Q')
  ]
  if mde_abs is not None and not math.isnan(mde_abs) and mde_abs > 0:
    band = pd.DataFrame({'y': [-mde_abs, mde_abs], 'label': ['−MDE', '+MDE']})
    layers.append(
        alt.Chart(band)
        .mark_rule(color=charts.RED, strokeDash=[4, 4])
        .encode(y='y:Q', tooltip=[alt.Tooltip('label:N'),
                                  alt.Tooltip('y:Q', format='+,.0f')])
    )
  layers.append(line)
  return alt.layer(*layers).properties(width=_WIDTH, height=_HEIGHT)


def daily_spend_chart(design: api.Design) -> Optional[alt.Chart]:
  """Plots total daily pretest spend."""
  if design.data is None or design.data.empty:
    return None
  frame = design.data.copy()
  frame.columns = [str(c).lower() for c in frame.columns]
  columns = [
      c for c in frame.columns
      if c == api.SPEND or c.startswith('spend_cell_')
  ]
  if not columns or api.DATE not in frame:
    return None
  frame[api.DATE] = pd.to_datetime(frame[api.DATE])
  frame['total_spend'] = frame[columns].astype(float).sum(axis=1)
  daily = frame.groupby(api.DATE)['total_spend'].sum().reset_index()
  daily.columns = ['date', 'spend']
  return (
      alt.Chart(daily)
      .mark_line(color=charts.GREEN, strokeWidth=1.5)
      .encode(
          x=alt.X('date:T', title=None),
          y=alt.Y('spend:Q', title='Daily spend', scale=alt.Scale(zero=False)),
          tooltip=[alt.Tooltip('date:T'),
                   alt.Tooltip('spend:Q', format=',.0f')],
      )
      .properties(width=_WIDTH * 2, height=160)
  )


def _weekly_base(weekly: pd.DataFrame, names: report_data.GeoNames):
  frame = weekly.copy()
  frame['geo_name'] = [names(loc) for loc in frame['location']]
  frame['week_label'] = frame['week'].dt.strftime('%Y-%m-%d')
  order = (
      frame.groupby('geo_name')['conversions'].sum()
      .sort_values(ascending=False).index.tolist()
  )
  return frame, order


def tracking_gaps_chart(
    weekly: pd.DataFrame, names: report_data.GeoNames
) -> Optional[alt.Chart]:
  """Heatmap of weekly conversions per geo; empty weeks are flagged."""
  if weekly.empty:
    return None
  frame, order = _weekly_base(weekly, names)
  frame['status'] = [
      'No conversions' if zero else 'Has conversions' for zero in frame['zero']
  ]
  return (
      alt.Chart(frame)
      .mark_rect(stroke='white', strokeWidth=1)
      .encode(
          x=alt.X('week_label:O', title='Pretest week'),
          y=alt.Y('geo_name:N', title=None, sort=order),
          color=alt.Color(
              'status:N',
              title=None,
              scale=alt.Scale(
                  domain=['Has conversions', 'No conversions'],
                  range=['#CEEAD6', charts.RED],
              ),
          ),
          tooltip=[
              alt.Tooltip('geo_name:N', title='Geo'),
              alt.Tooltip('week_label:O', title='Week'),
              alt.Tooltip('conversions:Q', format=',.0f'),
          ],
      )
      .properties(width=_WIDTH * 2, height=max(80, 16 * len(order)))
  )


def weekly_z_chart(
    weekly: pd.DataFrame, names: report_data.GeoNames
) -> Optional[alt.Chart]:
  """Heatmap of per-geo robust z-scores of weekly conversions."""
  if weekly.empty:
    return None
  frame, order = _weekly_base(weekly, names)
  frame['z_clipped'] = frame['z'].clip(-3, 3)
  return (
      alt.Chart(frame)
      .mark_rect(stroke='white', strokeWidth=1)
      .encode(
          x=alt.X('week_label:O', title='Pretest week'),
          y=alt.Y('geo_name:N', title=None, sort=order),
          color=alt.Color(
              'z_clipped:Q',
              title='Unusualness (z)',
              scale=alt.Scale(scheme='redblue', domain=[-3, 3], reverse=True),
          ),
          tooltip=[
              alt.Tooltip('geo_name:N', title='Geo'),
              alt.Tooltip('week_label:O', title='Week'),
              alt.Tooltip('conversions:Q', format=',.0f'),
              alt.Tooltip('z:Q', title='Robust z', format='.1f'),
          ],
      )
      .properties(width=_WIDTH * 2, height=max(80, 16 * len(order)))
  )


SIMULATOR_CLASS = 'geox-simulator'
# A target iCPA more than this factor away from the reference trips the
# guardrail warning.
ICPA_GUARDRAIL_FACTOR = 2.0


def power_simulator(
    summary: report_data.CandidateSummary,
    design: api.Design,
    goals: report_data.DesignGoals,
    currency_symbol: str = '',
    spend_label: str = 'Spend',
    reference_cpic: Optional[float] = None,
) -> alt.HConcatChart:
  """Builds an interactive "what if spend or target iCPA changes?" chart.

  Controls (spend, target iCPA, test type, confidence and cell) are Vega
  bindings; the page CSS shows them above the chart they affect.

  Args:
    summary: The candidate to simulate.
    design: The candidate's design.
    goals: Design goals used for the default control values.
    currency_symbol: Currency symbol for labels.
    spend_label: Name of the spend amount.
    reference_cpic: Expected iCPA used by the unrealistic-input guardrail.

  Returns:
    The chart.
  """
  alpha, power, test_type = _design_params(design)
  z_alpha0 = _z_alpha(alpha, test_type)
  z_power = float(stats.norm.ppf(power))
  cells = [
      c for c in summary.cells.values()
      if not math.isnan(c.mde_abs) and c.mde_abs > 0
  ]
  n_holdback = sum(
      1 for c in cells if c.experiment_type == api.ExperimentType.HOLDBACK
  )
  default_spend = (
      goals.target_budget / max(n_holdback, 1)
      if goals.target_budget
      else max((abs(c.budget) for c in cells), default=1.0)
  )
  default_cpic = goals.target_cpic or reference_cpic or 1.0
  reference = reference_cpic or default_cpic
  max_spend = max(
      [default_spend * 3] + [abs(c.budget) * 2 for c in cells]
  )
  constants = pd.DataFrame([
      {
          'group': c.label,
          'sigma': c.mde_abs / (z_alpha0 + z_power),
          'z_power': z_power,
          'reference_cpic': reference,
      }
      for c in cells
  ])
  grid = constants.merge(
      pd.DataFrame({'spend': np.linspace(0, max_spend, 61)}), how='cross'
  )
  step = max(round(max_spend / 200, -1), 1.0)
  spend = alt.param(
      name='sim_spend',
      value=round(default_spend, 2),
      bind=alt.binding_range(
          min=0, max=round(max_spend, 2), step=step,
          name=f'{spend_label} ({currency_symbol or "amount"}) ',
      ),
  )
  cpic = alt.param(
      name='sim_icpa',
      value=round(default_cpic, 4),
      bind=alt.binding(
          input='number', name=f'Target iCPA ({currency_symbol or "amount"}) '
      ),
  )
  tail = alt.param(
      name='sim_tail',
      value=(
          'Two-sided' if test_type == api.TestType.TWO_SIDED else 'One-sided'
      ),
      bind=alt.binding_radio(
          options=['Two-sided', 'One-sided'], name='Test type '
      ),
  )
  confidence = alt.param(
      name='sim_conf',
      value=round(1 - alpha, 4),
      bind=alt.binding_select(
          options=[0.8, 0.9, 0.95],
          labels=['80%', '90%', '95%'],
          name='Confidence ',
      ),
  )
  params = [spend, cpic, tail, confidence]
  labels = [c.label for c in cells]
  cell_filter = 'true'
  if len(cells) > 1:
    cell_param = alt.param(
        name='sim_cell',
        value=labels[0],
        bind=alt.binding_select(options=labels, name='Cell '),
    )
    params.append(cell_param)
    cell_filter = 'datum.group == sim_cell'

  z_alpha_expr = (
      "quantileNormal(1 - (1 - sim_conf) / (sim_tail == 'Two-sided' ? 2 : 1))"
  )
  safe_cpic = 'max(sim_icpa, 1e-9)'
  curve = (
      alt.Chart(grid)
      .transform_filter(cell_filter)
      .transform_calculate(
          power=(
              f'cumulativeNormal(datum.spend / {safe_cpic} / datum.sigma - '
              f'{z_alpha_expr})'
          )
      )
      .mark_line(strokeWidth=2.5)
      .encode(
          x=alt.X('spend:Q', title=f'{spend_label}'),
          y=alt.Y(
              'power:Q', title='Chance of detecting the effect',
              axis=alt.Axis(format='.0%'), scale=alt.Scale(domain=[0, 1]),
          ),
          color=alt.Color(
              'group:N', scale=charts.group_scale(labels), legend=None
          ),
      )
  )
  money = json.dumps(currency_symbol or '')
  money_title = f' ({currency_symbol})' if currency_symbol else ''
  point_data = (
      alt.Chart(constants)
      .transform_filter(cell_filter)
      .transform_calculate(
          spend='sim_spend',
          z_alpha=z_alpha_expr,
          power=(
              f'cumulativeNormal(sim_spend / {safe_cpic} / datum.sigma - '
              'datum.z_alpha)'
          ),
          needed=f'(datum.z_alpha + datum.z_power) * datum.sigma * {safe_cpic}',
          headline=(
              "'Chance of detecting: ' + format(datum.power, '.0%') + "
              f"'  ·  {spend_label} needed for {power:.0%}: ' + {money} + "
              "format(datum.needed, ',.0f')"
          ),
          warning=(
              f'abs(log({safe_cpic} / datum.reference_cpic)) > '
              f'log({ICPA_GUARDRAIL_FACTOR}) ? '
              "'Check the target iCPA: it is more than "
              f"{ICPA_GUARDRAIL_FACTOR:g}x away from the expected iCPA (' + "
              f"{money} + format(datum.reference_cpic, ',.2f') + '), so these "
              "numbers may be unrealistic.' : ''"
          ),
      )
  )
  rule = point_data.mark_rule(color=charts.GREY_900, strokeDash=[3, 3]).encode(
      x='spend:Q'
  )
  dot = point_data.mark_point(
      filled=True, size=90, color=charts.GREY_900
  ).encode(x='spend:Q', y='power:Q')
  headline = point_data.mark_text(
      align='left', baseline='top', fontSize=13, fontWeight=500,
      color=charts.GREY_900,
  ).encode(text='headline:N', x=alt.value(4), y=alt.value(-28))
  warning = point_data.mark_text(
      align='left', baseline='top', fontSize=12, color=charts.RED
  ).encode(text='warning:N', x=alt.value(4), y=alt.value(-12))
  spend_panel = alt.layer(curve, rule, dot, headline, warning).properties(
      width=_WIDTH, height=_HEIGHT
  )
  # Chance of a conclusive result as the target iCPA changes, at the chosen
  # spend: shows how fast power drops if the real iCPA is worse than planned.
  max_icpa = max(3.0 * reference, 0.01)
  icpa_grid = constants.merge(
      pd.DataFrame({'icpa': np.linspace(max_icpa / 60, max_icpa, 60)}),
      how='cross',
  )
  icpa_curve = (
      alt.Chart(icpa_grid)
      .transform_filter(cell_filter)
      .transform_calculate(
          power=(
              'cumulativeNormal(sim_spend / datum.icpa / datum.sigma - '
              f'{z_alpha_expr})'
          )
      )
      .mark_line(strokeWidth=2.5)
      .encode(
          x=alt.X('icpa:Q', title=f'Target iCPA{money_title}'),
          y=alt.Y(
              'power:Q', title='Chance of a conclusive result',
              axis=alt.Axis(format='.0%'), scale=alt.Scale(domain=[0, 1]),
          ),
          color=alt.Color(
              'group:N', scale=charts.group_scale(labels), legend=None
          ),
          tooltip=[
              alt.Tooltip('icpa:Q', title='iCPA', format=',.2f'),
              alt.Tooltip('power:Q', title='Chance', format='.0%'),
          ],
      )
  )
  planning = (
      alt.Chart(pd.DataFrame({'icpa': [reference]}))
      .mark_rule(color=charts.GREY_700, strokeDash=[2, 2])
      .encode(x='icpa:Q', tooltip=[alt.Tooltip('icpa:Q',
                                                 title='Planning iCPA')])
  )
  chosen = (
      alt.Chart(pd.DataFrame({'one': [1]}))
      .transform_calculate(icpa='sim_icpa')
      .mark_rule(color=charts.GREY_900, strokeDash=[3, 3])
      .encode(x='icpa:Q')
  )
  icpa_panel = alt.layer(icpa_curve, planning, chosen).properties(
      width=_WIDTH - 80, height=_HEIGHT
  )
  return (
      alt.hconcat(spend_panel, icpa_panel)
      .add_params(*params)
      .properties(padding={'top': 36})
  )


def calendar_chart(schedule: report_data.ExperimentSchedule) -> Optional[
    alt.FacetChart
]:
  """Plots Monday-start month calendars shading test and cooldown days."""
  if not schedule.dated:
    return None
  phases = schedule.days().set_index('date')['phase']
  start = phases.index.min().replace(day=1)
  end = (phases.index.max() + pd.offsets.MonthEnd(0)).normalize()
  days = pd.DataFrame({'date': pd.date_range(start, end)})
  days['phase'] = [phases.get(d, 'Other days') for d in days['date']]
  days['month'] = days['date'].dt.strftime('%B %Y')
  days['month_order'] = days['date'].dt.year * 12 + days['date'].dt.month
  days['weekday'] = days['date'].dt.dayofweek
  first_weekday = days['date'].dt.to_period('M').dt.start_time.dt.dayofweek
  days['week'] = (days['date'].dt.day - 1 + first_weekday) // 7
  days['day'] = days['date'].dt.day
  days['date_label'] = days['date'].dt.strftime('%a %b %d, %Y')
  weekday_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
  days['weekday_name'] = [weekday_names[d] for d in days['weekday']]
  base = alt.Chart().encode(
      x=alt.X(
          'weekday_name:O', title=None, sort=weekday_names,
          axis=alt.Axis(orient='top', labelAngle=0, ticks=False, domain=False),
      ),
      y=alt.Y('week:O', title=None, axis=None),
  )
  cells = base.mark_rect(stroke='white', strokeWidth=2, cornerRadius=3).encode(
      color=alt.Color(
          'phase:N',
          title=None,
          scale=alt.Scale(
              domain=['Ramp-up', 'Test', 'Cooldown', 'Other days'],
              range=[charts.RAMP_UP_COLOR, charts.BLUE, charts.YELLOW,
                     charts.GREY_100],
          ),
      ),
      tooltip=[alt.Tooltip('date_label:N', title='Date'), 'phase:N'],
  )
  text = base.mark_text(fontSize=10).encode(
      text='day:Q',
      color=alt.condition(
          "datum.phase == 'Test'", alt.value('white'), alt.value(
              charts.GREY_900)
      ),
  )
  return (
      alt.layer(cells, text, data=days)
      .properties(width=210, height=150)
      .facet(
          facet=alt.Facet(
              'month:N', title=None,
              sort=alt.SortField('month_order', order='ascending'),
          ),
          columns=3,
      )
  )


def data_quality_chart(design: api.Design) -> Optional[alt.LayerChart]:
  """Plots total daily pretest conversions with excluded/outlier dates."""
  if design.data is None or design.data.empty:
    return None
  frame = design.data.copy()
  frame.columns = [str(c).lower() for c in frame.columns]
  frame[api.DATE] = pd.to_datetime(frame[api.DATE])
  daily = frame.groupby(api.DATE)[api.CONVERSIONS].sum().reset_index()
  daily.columns = ['date', 'conversions']
  outliers = set()
  if design.quality_check_result is not None:
    outliers = {
        pd.Timestamp(d) for d in design.quality_check_result.outlier_dates
    }
  excluded = {pd.Timestamp(d) for d in (design.excluded_dates or ())}
  marks = daily[daily['date'].isin(excluded | outliers)].copy()
  marks['reason'] = [
      'Outlier date' if d in outliers else 'Excluded date'
      for d in marks['date']
  ]
  line = (
      alt.Chart(daily)
      .mark_line(color=charts.BLUE, strokeWidth=1.5)
      .encode(
          x=alt.X('date:T', title=None),
          y=alt.Y('conversions:Q', title='Total daily conversions',
                  scale=alt.Scale(zero=False)),
          tooltip=[alt.Tooltip('date:T'), alt.Tooltip('conversions:Q',
                                                      format=',.0f')],
      )
  )
  layers = [line]
  if not marks.empty:
    layers.append(
        alt.Chart(marks)
        .mark_point(filled=True, size=60)
        .encode(
            x='date:T',
            y='conversions:Q',
            color=alt.Color(
                'reason:N', title=None,
                scale=alt.Scale(
                    domain=['Excluded date', 'Outlier date'],
                    range=[charts.RED, charts.ORANGE],
                ),
            ),
            tooltip=[alt.Tooltip('date:T'), 'reason:N'],
        )
    )
  return alt.layer(*layers).properties(width=_WIDTH * 2, height=200)
