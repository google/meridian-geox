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
from typing import Any, Optional, Sequence

import altair as alt
from meridian_geox import api
from meridian_geox import charts
from meridian_geox import report_data
import numpy as np
import pandas as pd
from scipy import stats

_WIDTH = charts.DEFAULT_WIDTH
_HEIGHT = charts.DEFAULT_HEIGHT


# Vega test for the candidate selected on the page.
_SELECTED = f'datum.design_id == {charts.DESIGN_SELECTION}'
SIMULATOR_CLASS = 'geox-simulator'
# A target iCPA more than this factor from the reference trips a warning.
ICPA_GUARDRAIL_FACTOR = 2.0


def _padded(values: Sequence[float], pad: float = 0.08) -> tuple[float, float]:
  """A [low, high] domain around `values`, padded so points sit inside."""
  finite = [float(v) for v in values if v is not None and math.isfinite(v)]
  if not finite:
    return 0.0, 1.0
  low, high = min(finite), max(finite)
  span = (high - low) or (abs(high) * 0.1) or 1.0
  return low - span * pad, high + span * pad


def _ideal_zone(
    x_domain: tuple[float, float], y_domain: tuple[float, float],
    budget: Optional[float], y_goal: Optional[float], lower_is_better: bool,
) -> Optional[dict[str, float]]:
  """The goal-meeting rectangle clipped to the plotted domain, or None."""
  if budget is None and y_goal is None:
    return None
  (x0, x1), (y0, y1) = x_domain, y_domain
  if budget is not None:
    x1 = min(budget, x1)
  if y_goal is not None and lower_is_better:
    y1 = min(y_goal, y1)
  elif y_goal is not None:
    y0 = max(y_goal, y0)
  if x1 <= x0 or y1 <= y0:
    return None
  return {'x0': x0, 'x1': x1, 'y0': y0, 'y1': y1}


def _panel(
    layers: Sequence[alt.Chart], title: str, width: int = _WIDTH
) -> alt.LayerChart:
  """A titled simulator panel."""
  return alt.layer(*layers).properties(
      width=width, height=_HEIGHT, title=charts.panel_title(title)
  )


def sensitivity_curve(
    summary: report_data.CandidateSummary,
    design: api.Design,
    target_mde: Optional[float] = None,
) -> alt.LayerChart:
  """Plots the chance of detecting each true lift size."""
  _, power, _ = report_data.design_params(design)
  z_alpha, z_power = report_data.z_scores(design)
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
  power_rule = charts.rule(
      pd.DataFrame({'y': [power]}), charts.GREY_500, [2, 2], y='y:Q'
  )
  mde_rule = charts.rule(
      pd.DataFrame({'x': [summary.max_mde], 'label': ['MDE']}), charts.GREY_700,
      x='x:Q', tooltip=[alt.Tooltip('x:Q', title='MDE', format='.2%')],
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
    layers.insert(0, charts.rule(goal, charts.GREEN, [4, 4], x='x:Q'))
    layers.append(
        alt.Chart(goal)
        .mark_text(align='left', dx=4, dy=-6, fontSize=10,
                   color=charts.GREEN)
        .encode(x='x:Q', y=alt.value(_HEIGHT), text='text:N')
    )
  return alt.layer(*layers).properties(width=_WIDTH, height=_HEIGHT)


_FIT_SERIES = {
    'observed': 'Observed', 'counterfactual': 'Predicted from control'
}


def pretest_fit_chart(design: api.Design) -> alt.LayerChart:
  """Plots observed vs. counterfactual pretest conversions per cell."""
  labels = report_data.group_labels(design)
  frames = []
  for cell, per_cell in design.designs.items():
    df = per_cell.counterfactual_conversions
    if df is None or df.empty:
      continue
    date_col = 'index' if 'index' in df.columns else df.columns[0]
    long = df.rename(columns={date_col: 'date', **_FIT_SERIES}).melt(
        'date', list(_FIT_SERIES.values()), 'series', 'value'
    )
    frames.append(long.assign(
        date=pd.to_datetime(long['date']), value=long['value'].astype(float),
        group=labels[cell],
    ))
  frame = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(
      columns=['date', 'value', 'series', 'group']
  )
  layers = [alt.Chart(frame).mark_line(strokeWidth=1.8).encode(
      x=charts.DATE_X,
      y=alt.Y(
          'value:Q', title='Daily conversions', scale=alt.Scale(zero=False)
      ),
      color=alt.Color('group:N', title=None, scale=charts.group_scale(
          list(labels.values())
      )),
      strokeDash=alt.StrokeDash('series:N', title=None, scale=alt.Scale(
          domain=['Observed', 'Predicted from control'], range=[[1, 0], [5, 3]]
      )),
      tooltip=[
          charts.DATE_TOOLTIP, alt.Tooltip('group:N'),
          alt.Tooltip('series:N'), alt.Tooltip('value:Q', format=',.0f'),
      ],
  )]
  excluded = sorted(pd.to_datetime(list(design.excluded_dates or ())))
  if excluded:
    layers.insert(0, alt.Chart(pd.DataFrame({'date': excluded})).mark_rule(
        color=charts.RED, opacity=0.25
    ).encode(x='date:T'))
  return alt.layer(*layers).properties(width=_WIDTH, height=_HEIGHT)


def largest_geos_chart(
    geo_table: pd.DataFrame, color_scale: alt.Scale, top_n: int = 22
) -> alt.Chart:
  """Largest geos: share of conversions (solid) and of spend (light)."""
  frame = geo_table[geo_table['cell'] != 'excluded']
  frame = frame.sort_values('share', ascending=False).head(top_n)
  has_spend = 'spend_share' in frame and float(frame['spend_share'].sum()) > 0
  columns = ['geo_name', 'group', 'share'] + ['spend_share'] * has_spend
  base = alt.Chart(frame[columns])
  if has_spend:
    base = base.transform_fold(['share', 'spend_share'], ['measure', 'value'])
  base = base.transform_calculate(measure_label=(
      "datum.measure == 'share' ? 'Conversions' : 'Spend'" if has_spend
      else "'Conversions'"
  ))
  value = 'value:Q' if has_spend else 'share:Q'
  measures = ['Conversions', 'Spend']
  return base.mark_bar().encode(
      y=alt.Y('geo_name:N', title=None, sort=list(frame['geo_name'])),
      yOffset=alt.YOffset('measure_label:N', sort=measures),
      x=alt.X(
          value, title='Share of pretest conversions (solid) and spend',
          axis=alt.Axis(format='.1%'),
      ),
      color=alt.Color('group:N', title=None, scale=color_scale),
      opacity=alt.Opacity('measure_label:N', title=None, scale=alt.Scale(
          domain=measures, range=[1.0, 0.35]
      )),
      tooltip=[
          alt.Tooltip('geo_name:N', title='Geo'),
          alt.Tooltip('group:N', title='Group'),
          alt.Tooltip('measure_label:N', title='Measure'),
          alt.Tooltip(value, title='Share', format='.2%'),
      ],
  ).properties(width=_WIDTH, height=max(80, 22 * len(frame)))


def tradeoff_toggle_chart(
    summaries: Sequence[report_data.CandidateSummary],
    goals: report_data.DesignGoals,
    currency_symbol: str = '',
    spend_label: str = 'Spend',
) -> alt.LayerChart:
  """Spend vs. MDE or max detectable iCPA (toggle); clicking selects."""
  rows = pd.DataFrame([{
      'design_id': s.design_id, 'rank': s.rank,
      'option': f"{'★ ' * (s.rank == 1)}Option {s.rank}",
      'spend': s.total_spend, 'mde': s.max_mde, 'cpic': s.max_detectable_cpic,
      'share': sum(c.treatment_share for c in s.cells.values()),
      'status': 'Meets goals' if s.feasible else 'Misses a goal',
  } for s in summaries])
  dup = (
      rows.groupby([rows['spend'].round(2), rows['mde'].round(6)]).cumcount()
      if len(rows) else pd.Series(dtype=int)
  )
  rows['dup'] = dup
  mde_label = 'Detectable lift, MDE % (lower is better)'
  cpic_label = 'Max detectable iCPA (higher is better)'
  axis = alt.param(
      name='tradeoff_axis', value=mde_label,
      bind=alt.binding_radio(options=[mde_label, cpic_label], name='Show: '),
  )
  is_mde = f"tradeoff_axis == '{mde_label}'"
  money = f' ({currency_symbol})' if currency_symbol else ''
  label_expr = (
      f"{is_mde} ? format(datum.value, '.2%') : "
      f"{json.dumps(currency_symbol)} + format(datum.value, ',.2f')"
  )
  # Padded per-metric domains (room right for labels); zones clip to them.
  x_low, x_high = _padded(rows['spend'].tolist(), 0.15)
  x_domain = (x_low, x_high + (x_high - x_low) * 0.3)
  mde_domain, cpic_domain = (
      _padded(rows[m].tolist(), 0.2) for m in ('mde', 'cpic')
  )
  y_scale = alt.Scale(domain={
      'expr': f'{is_mde} ? {list(mde_domain)} : {list(cpic_domain)}'
  }, nice=False, zero=False)
  x_scale = alt.Scale(domain=list(x_domain), nice=False, zero=False)
  y = alt.Y('y:Q', title=None, scale=y_scale,
            axis=alt.Axis(labelExpr=label_expr, tickCount=5))
  x = alt.X('spend:Q', title=f'{spend_label} needed{money}', scale=x_scale,
            axis=alt.Axis(format='~s', tickCount=5))
  zones = pd.DataFrame([
      {**zone, 'metric': metric}
      for metric, domain, goal, lower in (
          ('mde', mde_domain, goals.target_mde, True),
          ('cpic', cpic_domain, goals.target_cpic, False),
      )
      if (zone := _ideal_zone(
          x_domain, domain, goals.target_budget, goal, lower
      ))
  ])
  layers: list[alt.Chart] = []
  if not zones.empty:
    shown = alt.Chart(zones).transform_filter(
        f"({is_mde}) == (datum.metric == 'mde')"
    )
    zone_x = alt.X('x0:Q', title=None, scale=x_scale)
    # Outlined, not filled, so the chart keeps a plain background.
    layers = [
        shown.mark_rect(
            filled=False, stroke=charts.GREEN, strokeDash=[5, 4],
            strokeWidth=1.5, clip=True,
        ).encode(
            x=zone_x, x2='x1:Q',
            y=alt.Y('y0:Q', title=None, scale=y_scale), y2='y1:Q',
        ),
        shown.mark_text(
            align='left', baseline='top', dx=6, dy=6, fontSize=12,
            fontWeight=500, color=charts.GREEN,
        ).encode(
            x=zone_x, y=alt.Y('y1:Q', scale=y_scale, title=None),
            text=alt.value('Ideal zone'),
        ),
    ]

  def sel(on: float | str, off: float | str):
    return alt.condition(_SELECTED, alt.value(on), alt.value(off))

  base = alt.Chart(rows).transform_calculate(
      y=f'{is_mde} ? datum.mde : datum.cpic',
      label="datum.option + ' · ' + format(datum.share, '.0%')",
  )
  tooltip = [
      alt.Tooltip('option:N', title='Option'),
      alt.Tooltip('spend:Q', title=spend_label, format=',.0f'),
      alt.Tooltip('mde:Q', title='MDE (%)', format='.2%'),
      alt.Tooltip('cpic:Q', title='Max detectable iCPA', format=',.2f'),
      alt.Tooltip('share:Q', title='Treatment share', format='.0%'),
      alt.Tooltip('status:N', title='Goals'),
  ]
  # See-through bubbles (identical candidates nudged apart); the selected one
  # gets a thick dark outline.
  points = [
      base.transform_filter(f'datum.dup == {d}').mark_circle(
          size=520, color=charts.LIGHT_BLUE, opacity=0.55, cursor='pointer',
          xOffset=12 * d,
      ).encode(
          x=x, y=y, stroke=sel(charts.GREY_900, charts.DARK_BLUE),
          strokeWidth=sel(3, 1.5), strokeOpacity=alt.value(1), tooltip=tooltip,
      )
      for d in range((int(dup.max()) if len(dup) else 0) + 1)
  ]
  # "Option N · share" labels: odd ranks above, even ranks below.
  labels = [
      base.transform_filter(f'datum.rank % 2 == {parity}').mark_text(
          align='left', dx=16, dy=dy, fontSize=13
      ).encode(
          x=x, y=y, text='label:N', color=sel(charts.GREY_900, charts.GREY_700),
          fontWeight=sel(700, 400),
      )
      for parity, dy in ((1, -14), (0, 18))
  ]
  default_id = summaries[0].design_id if summaries else ''
  return alt.layer(*layers, *points, *labels).add_params(
      axis, charts.selection_param(default_id)
  ).properties(width=_WIDTH, height=_HEIGHT)


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
  layers = [charts.rule(pd.DataFrame({'y': [0.0]}), charts.GREY_500, y='y:Q')]
  if mde_abs is not None and mde_abs > 0:
    band = pd.DataFrame({'y': [-mde_abs, mde_abs], 'label': ['−MDE', '+MDE']})
    layers.append(charts.rule(band, charts.RED, [4, 4], y='y:Q', tooltip=[
        alt.Tooltip('label:N'), alt.Tooltip('y:Q', format='+,.0f')
    ]))
  layers.append(alt.Chart(drift.frame).mark_line(
      color=charts.BLUE, strokeWidth=2
  ).encode(
      x=charts.DATE_X,
      y=alt.Y('drift:Q', title='Cumulative conversions (observed − expected)'),
      tooltip=[
          charts.DATE_TOOLTIP,
          alt.Tooltip('drift:Q', title='Drift', format='+,.0f'),
      ],
  ))
  return alt.layer(*layers).properties(width=_WIDTH, height=_HEIGHT)


def _daily_line(
    daily: pd.DataFrame, field: str, title: str, color: str
) -> alt.Chart:
  """Line of a daily pretest total (`field`)."""
  return alt.Chart(daily[['date', field]]).mark_line(
      color=color, strokeWidth=1.5
  ).encode(
      x=charts.DATE_X,
      y=alt.Y(f'{field}:Q', title=title, scale=alt.Scale(zero=False)),
      tooltip=[charts.DATE_TOOLTIP, alt.Tooltip(f'{field}:Q', format=',.0f')],
  )


def daily_spend_chart(design: api.Design) -> Optional[alt.Chart]:
  """Plots total daily pretest spend."""
  daily = report_data.daily_totals(design.data)
  if 'spend' not in daily:
    return None
  return _daily_line(daily, 'spend', 'Daily spend', charts.GREEN).properties(
      width=_WIDTH * 2, height=160
  )


def _weekly_heatmap(
    weekly: pd.DataFrame,
    names: report_data.GeoNames,
    column: tuple[str, Any],
    color: alt.Color,
    tooltip: Sequence[alt.Tooltip] = (),
) -> Optional[alt.Chart]:
  """Week × geo heatmap (geos by total conversions) coloured by `column`."""
  if weekly.empty:
    return None
  frame = weekly.copy()
  frame['geo_name'] = [names(loc) for loc in frame['location']]
  frame['week_label'] = frame['week'].dt.strftime('%Y-%m-%d')
  frame[column[0]] = column[1]
  order = (
      frame.groupby('geo_name')['conversions'].sum()
      .sort_values(ascending=False).index.tolist()
  )
  return (
      alt.Chart(frame)
      .mark_rect(stroke='white', strokeWidth=1)
      .encode(
          x=alt.X('week_label:O', title='Pretest week'),
          y=alt.Y('geo_name:N', title=None, sort=order),
          color=color,
          tooltip=[
              alt.Tooltip('geo_name:N', title='Geo'),
              alt.Tooltip('week_label:O', title='Week'),
              alt.Tooltip('conversions:Q', format=',.0f'),
              *tooltip,
          ],
      )
      .properties(width=_WIDTH * 2, height=max(80, 16 * len(order)))
  )


def tracking_gaps_chart(
    weekly: pd.DataFrame, names: report_data.GeoNames
) -> Optional[alt.Chart]:
  """Heatmap of weekly conversions per geo; empty weeks are flagged."""
  status = [
      'No conversions' if zero else 'Has conversions' for zero in weekly['zero']
  ]
  return _weekly_heatmap(weekly, names, ('status', status), alt.Color(
      'status:N',
      title=None,
      scale=alt.Scale(
          domain=['Has conversions', 'No conversions'],
          range=[charts.LIGHT_GREEN, charts.RED],
      ),
  ))


def weekly_z_chart(
    weekly: pd.DataFrame, names: report_data.GeoNames
) -> Optional[alt.Chart]:
  """Heatmap of per-geo robust z-scores of weekly conversions."""
  return _weekly_heatmap(
      weekly,
      names,
      ('z_clipped', weekly['z'].clip(-3, 3)),
      alt.Color(
          'z_clipped:Q',
          title='Unusualness (z)',
          scale=alt.Scale(scheme='redblue', domain=[-3, 3], reverse=True),
      ),
      [alt.Tooltip('z:Q', title='Robust z', format='.1f')],
  )


def power_simulator(
    summary: report_data.CandidateSummary,
    design: api.Design,
    goals: report_data.DesignGoals,
    currency_symbol: str = '',
    spend_label: str = 'Spend',
    reference_cpic: Optional[float] = None,
) -> alt.HConcatChart:
  """Interactive "what if spend or target iCPA changes?" chart."""
  alpha, power, test_type = report_data.design_params(design)
  z_alpha0, z_power = report_data.z_scores(design)
  cells = [c for c in summary.cells.values() if c.mde_abs > 0]  # Drops NaN.
  holdback, go_dark = api.ExperimentType.HOLDBACK, api.ExperimentType.GO_DARK
  n_holdback = sum(c.experiment_type == holdback for c in cells)
  default_spend = (
      goals.target_budget / max(n_holdback, 1) if goals.target_budget
      else max((abs(c.budget) for c in cells), default=1.0)
  )
  default_cpic = goals.target_cpic or reference_cpic or 1.0
  reference = reference_cpic or default_cpic
  max_spend = max([default_spend * 3] + [abs(c.budget) * 2 for c in cells])
  # A go-dark can pause at most the business-as-usual spend.
  bau = [
      b for change in report_data.spend_changes(design)
      if change.experiment_type == go_dark
      and (b := report_data.bau_spend(change)) is not None
  ]
  if cells and bau and all(c.experiment_type == go_dark for c in cells):
    max_spend = max(bau)
    default_spend = min(default_spend, max_spend)
  constants = pd.DataFrame([{
      'group': c.label, 'sigma': c.mde_abs / (z_alpha0 + z_power),
      'z_power': z_power, 'reference_cpic': reference,
      'baseline': c.mde_abs / c.mde if c.mde else 0.0,
  } for c in cells])
  amount = currency_symbol or 'amount'
  tails = ['Two-sided', 'One-sided']
  levels = sorted({0.8, 0.9, 0.95, round(1 - alpha, 4)})
  labels = [c.label for c in cells]
  bindings = [
      ('spend', round(default_spend, 2), alt.binding_range(
          min=0, max=round(max_spend, 2),
          step=max(round(max_spend / 200, -1), 1.0),
          name=f'{spend_label} ({amount}) ',
      )),
      ('icpa', round(default_cpic, 4),
       alt.binding(input='number', name=f'Target iCPA ({amount}) ')),
      ('tail', tails[test_type != api.TestType.TWO_SIDED],
       alt.binding_radio(options=tails, name='Test type ')),
      ('conf', round(1 - alpha, 4), alt.binding_select(
          options=levels, labels=[f'{level:.0%}' for level in levels],
          name='Confidence ',
      )),
  ]
  cell_filter = 'true'
  if len(cells) > 1:
    bindings.append(
        ('cell', labels[0], alt.binding_select(options=labels, name='Cell '))
    )
    cell_filter = 'datum.group == sim_cell'
  group_color = alt.Color(
      'group:N', scale=charts.group_scale(labels), legend=None
  )
  z_a = (
      "quantileNormal(1 - (1 - sim_conf) / (sim_tail == 'Two-sided' ? 2 : 1))"
  )
  safe_cpic = 'max(sim_icpa, 1e-9)'
  expected = f'sim_spend / {safe_cpic}'
  money = json.dumps(currency_symbol or '')

  def filtered(frame: pd.DataFrame) -> alt.Chart:
    return alt.Chart(frame).transform_filter(cell_filter)

  def cross(col: str, values: np.ndarray) -> alt.Chart:
    return filtered(constants.merge(pd.DataFrame({col: values}), how='cross'))

  def chance(numerator: str, z: str) -> str:
    return f'cumulativeNormal({numerator} / datum.sigma - {z})'

  def curve(grid, numerator, x, y_title, tooltip=alt.Undefined) -> alt.Chart:
    return grid.transform_calculate(power=chance(numerator, z_a)).mark_line(
        strokeWidth=2.5
    ).encode(
        x=x, y=charts.percent_y('power:Q', y_title), color=group_color,
        tooltip=tooltip,
    )

  point_data = filtered(constants).transform_calculate(
      spend='sim_spend',
      z_alpha=z_a,
      power=chance(expected, 'datum.z_alpha'),
      warning=(
          f'abs(log({safe_cpic} / datum.reference_cpic)) > '
          f"log({ICPA_GUARDRAIL_FACTOR}) ? '⚠ Target iCPA is over "
          f"{ICPA_GUARDRAIL_FACTOR:g}x the planned iCPA (' + "
          f"{money} + format(datum.reference_cpic, ',.2f') + '): numbers "
          "may be unrealistic.' : ''"
      ),
  )
  spend_panel = _panel([
      curve(cross('spend', np.linspace(0, max_spend, 61)),
            f'datum.spend / {safe_cpic}', alt.X('spend:Q', title=spend_label),
            'Chance of detecting the effect'),
      charts.rule(point_data, charts.GREY_900, [3, 3], x='spend:Q'),
      point_data.mark_point(filled=True, size=90, color=charts.GREY_900).encode(
          x='spend:Q', y='power:Q'
      ),
      point_data.mark_text(
          align='left', baseline='top', fontSize=12, color=charts.RED,
          limit=_WIDTH - 8,
      ).encode(text='warning:N', x=alt.value(4), y=alt.value(-12)),
  ], 'How does the chance change with spend?')
  # Chance at the chosen spend as the target iCPA moves away from the plan.
  max_icpa = max(3.0 * reference, 0.01)
  money_title = f' ({currency_symbol})' if currency_symbol else ''
  icpa_panel = _panel([
      curve(cross('icpa', np.linspace(max_icpa / 60, max_icpa, 60)),
            'sim_spend / datum.icpa',
            alt.X('icpa:Q', title=f'Target iCPA{money_title}'),
            'Chance of a conclusive result', [
                alt.Tooltip('icpa:Q', title='iCPA', format=',.2f'),
                alt.Tooltip('power:Q', title='Chance', format='.0%'),
            ]),
      charts.rule(
          pd.DataFrame({'icpa': [reference]}), charts.GREY_700, [2, 2],
          x='icpa:Q', tooltip=[alt.Tooltip('icpa:Q', title='Planning iCPA')],
      ),
      charts.rule(
          alt.Chart(pd.DataFrame({'one': [1]})).transform_calculate(
              icpa='sim_icpa'
          ), charts.GREY_900, [3, 3], x='icpa:Q',
      ),
  ], 'How fast does the chance drop as iCPA rises?', _WIDTH - 80)
  # Spread with no effect (grey) vs around the expected lift (blue) = power.
  threshold = f'({z_a}) * datum.sigma'
  noise = cross('z', np.round(np.linspace(-4, 4, 81), 2)).transform_calculate(
      density='densityNormal(datum.z, 0, 1)',
      noise_x='datum.z * datum.sigma',
      lift_x=f'{expected} + datum.z * datum.sigma',
      threshold=threshold,
  )
  conversions_x = alt.X('x:Q', title='Incremental conversions over the test',
                        axis=alt.Axis(format='~s'))
  density_y = alt.Y('density:Q', title=None, axis=None)
  noise_line, lift_line = (
      noise.transform_calculate(x=f'datum.{x}').mark_line(
          color=color, strokeWidth=2
      ).encode(x=conversions_x, y=density_y)
      for x, color in (('noise_x', charts.GREY_500), ('lift_x', charts.BLUE))
  )
  lift_area = (
      noise.transform_calculate(x='datum.lift_x')
      .transform_filter('datum.x >= datum.threshold')
      .mark_area(color=charts.BLUE, opacity=0.25)
      .encode(x=conversions_x, y=density_y)
  )
  markers = filtered(constants).transform_calculate(
      threshold=threshold, lift=expected, peak='densityNormal(0, 0, 1)'
  )
  # Threshold label at the bottom, curve labels at the peaks: no overlap.
  noise_panel = _panel([
      noise_line, lift_area, lift_line,
      charts.rule(markers, charts.RED, [4, 3], x=alt.X('threshold:Q')),
      markers.mark_text(
          align='left', dx=4, dy=-4, fontSize=10, color=charts.RED,
          baseline='bottom',
      ).encode(
          x='threshold:Q', y=alt.value(_HEIGHT),
          text=alt.value('Significance threshold'),
      ),
      *(
          markers.mark_text(
              dy=-8, fontSize=11, color=color, fontWeight=weight
          ).encode(x=x, y=alt.Y('peak:Q', axis=None), text=alt.value(text))
          for x, color, weight, text in (
              (alt.datum(0), charts.GREY_700, alt.Undefined, 'Noise alone'),
              ('lift:Q', charts.BLUE, 500, 'Expected lift'),
          )
      ),
  ], 'Would the expected lift stand out from noise?', _WIDTH - 80)
  # Headline numbers, recomputed from the controls (plan_numbers formulas).
  high, medium = report_data.HIGH_CHANCE, report_data.MEDIUM_CHANCE

  def tier(a: str, b: str, c: str) -> str:
    return (
        f'datum.chance >= {high} ? {a} : datum.chance >= {medium} ? {b} : {c}'
    )

  def pct(prefix: str, num: str, fmt: str) -> str:
    return (
        f"datum.baseline > 0 ? '{prefix}' + format({num} / datum.baseline, "
        f"'{fmt}') : ''"
    )

  def signed(field: str) -> str:
    return f"(datum.{field} >= 0 ? '+' : '') + format(datum.{field}, ',.0f')"

  def cash(field: str, fmt: str) -> str:
    return f"{money} + format(datum.{field}, '{fmt}')"

  numbers = filtered(constants).transform_calculate(
      z_alpha=z_a,
      expected=expected,
      chance=chance('datum.expected', 'datum.z_alpha'),
      worst='datum.expected - datum.z_alpha * datum.sigma',
      mde_abs='(datum.z_alpha + datum.z_power) * datum.sigma',
      needed=f'datum.mde_abs * {safe_cpic}',
      max_cpic='sim_spend / datum.mde_abs',
      chance_value="format(datum.chance, '.0%')",
      chance_detail=tier("'High'", "'Medium'", "'Low'"),
      chance_color=tier(
          f"'{charts.GREEN}'", f"'{charts.GREY_900}'", f"'{charts.RED}'"
      ),
      expected_value=signed('expected'),
      expected_detail=pct('Lift: +', 'datum.expected', '.2%'),
      worst_value=signed('worst'),
      worst_detail=(
          "datum.worst <= 0 ? 'could be zero or negative' : 'iCPA up to ' + "
          f"{money} + format(sim_spend / datum.worst, ',.2f')"
      ),
      worst_label="'Worst case (' + format(sim_conf, '.0%') + ' conf.)'",
      needed_value=cash('needed', ',.0f'),
      needed_detail=pct('MDE ', 'datum.mde_abs', '.1%'),
      max_value=cash('max_cpic', ',.2f'),
      max_detail=(
          f"datum.max_cpic >= {safe_cpic} ? '≥ target iCPA ✓' : "
          "'below target iCPA ⚠'"
      ),
      max_color=(
          f"datum.max_cpic >= {safe_cpic} ? '{charts.GREEN}' : "
          f"'{charts.AMBER}'"
      ),
  )
  layers = []
  for col, (key, label, color) in enumerate([
      ('chance', "'Conclusive-result chance'", 'chance_color'),
      ('expected', "'Expected extra conversions'", None),
      ('worst', 'datum.worst_label', None),
      ('needed', json.dumps(f'{spend_label} for a {power:.0%} chance'), None),
      ('max', "'Max detectable iCPA'", 'max_color'),
  ]):
    layers += [
        _kpi_text(numbers.transform_calculate(label=label), col, 0, 'label',
                  12, color=charts.GREY_900, weight=500),
        _kpi_text(numbers, col, 18, f'{key}_value', 24, encoded_color=(
            alt.Color(f'{color}:N', scale=None) if color
            else alt.value(charts.GREY_900)
        )),
        _kpi_text(numbers, col, 48, f'{key}_detail', 12, color=charts.GREY_700),
    ]
  return alt.vconcat(
      alt.layer(*layers).properties(width=200 * 5, height=64),
      alt.hconcat(noise_panel, icpa_panel, spacing=72), spend_panel,
      spacing=28,
  ).add_params(*(
      alt.param(name=f'sim_{name}', value=value, bind=bind)
      for name, value, bind in bindings
  ))


def _kpi_text(
    chart: alt.Chart, col: int, y: int, field: str, size: int,
    color: Optional[str] = None, weight: Optional[int] = None,
    encoded_color=None,
) -> alt.Chart:
  """One left-aligned text row of a simulator headline-number column."""
  return chart.mark_text(
      align='left', baseline='top', fontSize=size,
      color=alt.Undefined if color is None else color,
      fontWeight=alt.Undefined if weight is None else weight,
  ).encode(
      text=f'{field}:N', x=alt.value(col * 200), y=alt.value(y),
      color=alt.Undefined if encoded_color is None else encoded_color,
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
  # Ramp-up and cooldown share the yellow (stakeholder feedback: mark both
  # ends of the test in yellow); the legend lists only phases that occur.
  present = set(days['phase'])
  edges = [p for p in ('Ramp-up', 'Cooldown') if p in present]
  edge_label = ' / '.join(edges)
  days['legend'] = [edge_label if p in edges else p for p in days['phase']]
  legend_domain = (
      (['Test'] if 'Test' in present else [])
      + ([edge_label] if edges else [])
      + (['Other days'] if 'Other days' in present else [])
  )
  legend_range = [
      {'Test': charts.BLUE, 'Other days': charts.GREY_100}.get(p, charts.YELLOW)
      for p in legend_domain
  ]
  # Weekday names are a header row inside each month, below its title.
  header = pd.DataFrame([
      {'month': m, 'month_order': o, 'week': -1, 'weekday_name': w}
      for (m, o) in days[['month', 'month_order']].drop_duplicates().itertuples(
          index=False
      )
      for w in weekday_names
  ])
  data = pd.concat([days, header], ignore_index=True)
  base = alt.Chart().encode(
      x=alt.X('weekday_name:O', title=None, sort=weekday_names, axis=None),
      y=alt.Y('week:O', title=None, axis=None),
  )
  in_month = base.transform_filter('datum.week >= 0')
  cells = in_month.mark_rect(
      stroke='white', strokeWidth=2, cornerRadius=3
  ).encode(
      color=alt.Color(
          'legend:N',
          title=None,
          scale=alt.Scale(domain=legend_domain, range=legend_range),
          legend=alt.Legend(orient='bottom', direction='horizontal'),
      ),
      tooltip=[alt.Tooltip('date_label:N', title='Date'),
               alt.Tooltip('phase:N', title='Phase')],
  )
  text = in_month.mark_text(fontSize=10).encode(
      text='day:Q',
      color=alt.condition(
          "datum.phase == 'Test'", alt.value('white'), alt.value(
              charts.GREY_900)
      ),
  )
  weekdays = base.transform_filter('datum.week < 0').mark_text(
      fontSize=10, fontWeight=500, color=charts.GREY_700
  ).encode(text='weekday_name:N')
  return (
      alt.layer(cells, text, weekdays, data=data)
      .properties(width=210, height=170)
      .facet(
          facet=alt.Facet(
              'month:N', title=None,
              sort=alt.SortField('month_order', order='ascending'),
              header=alt.Header(labelFontSize=13, labelFontWeight=500),
          ),
          columns=3,
      )
  )


def data_quality_chart(design: api.Design) -> Optional[alt.LayerChart]:
  """Plots total daily pretest conversions with excluded/outlier dates."""
  daily = report_data.daily_totals(design.data)
  if 'conversions' not in daily:
    return None
  daily = daily[['date', 'conversions']]
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
  layers = [_daily_line(
      daily, 'conversions', 'Total daily conversions', charts.BLUE
  )]
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
            tooltip=[charts.DATE_TOOLTIP, 'reason:N'],
        )
    )
  return alt.layer(*layers).properties(width=_WIDTH * 2, height=200)
