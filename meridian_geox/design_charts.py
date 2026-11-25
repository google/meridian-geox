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


# Vega test for the candidate selected on the page.
_SELECTED = f'datum.design_id == {charts.DESIGN_SELECTION}'


def _padded(values: Sequence[float], pad: float = 0.08) -> tuple[float, float]:
  """A [low, high] domain around `values`, padded so points sit inside."""
  finite = [float(v) for v in values if v is not None and math.isfinite(v)]
  if not finite:
    return 0.0, 1.0
  low, high = min(finite), max(finite)
  span = (high - low) or (abs(high) * 0.1) or 1.0
  return low - span * pad, high + span * pad


def _ideal_zone(
    x_domain: tuple[float, float],
    y_domain: tuple[float, float],
    budget: Optional[float],
    y_goal: Optional[float],
    lower_is_better: bool,
) -> Optional[dict[str, float]]:
  """The goal-meeting rectangle clipped to the plotted domain, or None.

  Spend at or below the budget goal, and MDE at or below (or max detectable
  iCPA at or above) its goal. Clipping keeps x0 <= x1 and y0 <= y1.
  """
  if budget is None and y_goal is None:
    return None
  x0, x1 = x_domain[0], x_domain[1]
  if budget is not None:
    x1 = min(budget, x1)
  y0, y1 = y_domain
  if y_goal is not None:
    if lower_is_better:
      y1 = min(y_goal, y1)
    else:
      y0 = max(y_goal, y0)
  if x1 <= x0 or y1 <= y0:
    return None
  return {'x0': x0, 'x1': x1, 'y0': y0, 'y1': y1}


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
    frames.append(pd.DataFrame({
        'date': pd.to_datetime(long['date']),
        'value': long['value'].astype(float),
        'series': long['series'],
        'group': labels[cell],
    }))
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
    base = base.transform_fold(
        ['share', 'spend_share'], as_=['measure', 'value']
    ).transform_calculate(
        measure_label="datum.measure == 'share' ? 'Conversions' : 'Spend'"
    )
  else:
    base = base.transform_calculate(measure_label="'Conversions'")
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
  rows = pd.DataFrame([
      {
          'design_id': s.design_id,
          'rank': s.rank,
          'option': ('★ ' if s.rank == 1 else '') + f'Option {s.rank}',
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
  dup = (
      rows.groupby([rows['spend'].round(2), rows['mde'].round(6)]).cumcount()
      if len(rows) else pd.Series(dtype=int)
  )
  rows['dup'] = dup
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
      f"{json.dumps(currency_symbol)} + format(datum.value, ',.2f')"
  )
  # Padded per-metric domains (room right for labels); zones clip to them.
  x_low, x_high = _padded(rows['spend'].tolist(), 0.15)
  x_domain = (x_low, x_high + (x_high - x_low) * 0.3)
  mde_domain = _padded(rows['mde'].tolist(), 0.2)
  cpic_domain = _padded(rows['cpic'].dropna().tolist(), 0.2)
  y_domain_expr = (
      f'{is_mde} ? [{mde_domain[0]}, {mde_domain[1]}] :'
      f' [{cpic_domain[0]}, {cpic_domain[1]}]'
  )
  y_scale = alt.Scale(domain={'expr': y_domain_expr}, nice=False, zero=False)
  x_scale = alt.Scale(domain=list(x_domain), nice=False, zero=False)
  y = alt.Y(
      'y:Q', title=None, scale=y_scale,
      axis=alt.Axis(labelExpr=label_expr, tickCount=5),
  )
  x = alt.X(
      'spend:Q', title=f'{spend_label} needed{money}', scale=x_scale,
      axis=alt.Axis(format='~s', tickCount=5),
  )
  zones = []
  for metric, domain, goal, lower in (
      ('mde', mde_domain, goals.target_mde, True),
      ('cpic', cpic_domain, goals.target_cpic, False),
  ):
    zone = _ideal_zone(x_domain, domain, goals.target_budget, goal, lower)
    if zone is not None:
      zones.append({**zone, 'metric': metric})
  layers: list[alt.Chart] = []
  if zones:
    zone_frame = pd.DataFrame(zones)
    shown = alt.Chart(zone_frame).transform_filter(
        f"({is_mde}) == (datum.metric == 'mde')"
    )
    zone_x = alt.X('x0:Q', title=None, scale=x_scale)
    # Outlined, not filled, so the chart keeps a plain background.
    layers += [
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
  base = alt.Chart(rows).transform_calculate(
      y=y_expr,
      label="datum.option + ' · ' + format(datum.share, '.0%')",
  )
  # See-through bubbles (identical candidates nudged apart); the selected one
  # gets a thick dark outline.
  points = [
      base.transform_filter(f'datum.dup == {d}').mark_circle(
          size=520, color=charts.LIGHT_BLUE, opacity=0.55, cursor='pointer',
          xOffset=12 * d,
      ).encode(
          x=x,
          y=y,
          stroke=alt.condition(
              _SELECTED, alt.value(charts.GREY_900),
              alt.value(charts.DARK_BLUE),
          ),
          strokeWidth=alt.condition(_SELECTED, alt.value(3), alt.value(1.5)),
          strokeOpacity=alt.value(1),
          tooltip=[
              alt.Tooltip('option:N', title='Option'),
              alt.Tooltip('spend:Q', title=spend_label, format=',.0f'),
              alt.Tooltip('mde:Q', title='MDE (%)', format='.2%'),
              alt.Tooltip('cpic:Q', title='Max detectable iCPA', format=',.2f'),
              alt.Tooltip('share:Q', title='Treatment share', format='.0%'),
              alt.Tooltip('status:N', title='Goals'),
          ],
      )
      for d in range((int(dup.max()) if len(dup) else 0) + 1)
  ]
  # "Option N · share" labels: odd ranks above, even ranks below.
  labels = [
      base.transform_filter(f'datum.rank % 2 == {parity}').mark_text(
          align='left', dx=16, dy=dy, fontSize=13
      ).encode(
          x=x,
          y=y,
          text='label:N',
          color=alt.condition(
              _SELECTED, alt.value(charts.GREY_900), alt.value(charts.GREY_700)
          ),
          fontWeight=alt.condition(_SELECTED, alt.value(700), alt.value(400)),
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
  if mde_abs is not None and not math.isnan(mde_abs) and mde_abs > 0:
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
  """Interactive "what if spend or target iCPA changes?" chart."""
  alpha, power, test_type = report_data.design_params(design)
  z_alpha0, z_power = report_data.z_scores(design)
  cells = [
      c for c in summary.cells.values()
      if not math.isnan(c.mde_abs) and c.mde_abs > 0
  ]
  n_holdback = sum(
      c.experiment_type == api.ExperimentType.HOLDBACK for c in cells
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
  # A go-dark can pause at most the business-as-usual spend.
  bau = [
      b for change in report_data.spend_changes(design)
      if change.experiment_type == api.ExperimentType.GO_DARK
      and (b := report_data.bau_spend(change)) is not None
  ]
  if cells and bau and all(
      c.experiment_type == api.ExperimentType.GO_DARK for c in cells
  ):
    max_spend = max(bau)
    default_spend = min(default_spend, max_spend)
  constants = pd.DataFrame([
      {
          'group': c.label,
          'sigma': c.mde_abs / (z_alpha0 + z_power),
          'z_power': z_power,
          'reference_cpic': reference,
          'baseline': c.mde_abs / c.mde if c.mde else 0.0,
      }
      for c in cells
  ])
  amount = currency_symbol or 'amount'
  tails = ['Two-sided', 'One-sided']
  levels = sorted({0.8, 0.9, 0.95, round(1 - alpha, 4)})
  params = [
      alt.param(name='sim_spend', value=round(default_spend, 2), bind=(
          alt.binding_range(
              min=0, max=round(max_spend, 2),
              step=max(round(max_spend / 200, -1), 1.0),
              name=f'{spend_label} ({amount}) ',
          )
      )),
      alt.param(name='sim_icpa', value=round(default_cpic, 4), bind=(
          alt.binding(input='number', name=f'Target iCPA ({amount}) ')
      )),
      alt.param(
          name='sim_tail', value=tails[test_type != api.TestType.TWO_SIDED],
          bind=alt.binding_radio(options=tails, name='Test type '),
      ),
      alt.param(name='sim_conf', value=round(1 - alpha, 4), bind=(
          alt.binding_select(
              options=levels, labels=[f'{level:.0%}' for level in levels],
              name='Confidence ',
          )
      )),
  ]
  labels = [c.label for c in cells]
  cell_filter = 'true'
  if len(cells) > 1:
    params.append(alt.param(
        name='sim_cell', value=labels[0],
        bind=alt.binding_select(options=labels, name='Cell '),
    ))
    cell_filter = 'datum.group == sim_cell'
  group_color = alt.Color(
      'group:N', scale=charts.group_scale(labels), legend=None
  )

  def cell_chart(frame: pd.DataFrame) -> alt.Chart:
    return alt.Chart(frame).transform_filter(cell_filter)

  def cross(col: str, values: np.ndarray) -> pd.DataFrame:
    return constants.merge(pd.DataFrame({col: values}), how='cross')

  def chance(numerator: str, z: str) -> str:
    return f'cumulativeNormal({numerator} / datum.sigma - {z})'

  def dashed(chart: alt.Chart, x: str) -> alt.Chart:
    return chart.mark_rule(color=charts.GREY_900, strokeDash=[3, 3]).encode(x=x)

  def panel(*layers: alt.Chart, width: int, title: str) -> alt.LayerChart:
    return alt.layer(*layers).properties(
        width=width, height=_HEIGHT, title=charts.panel_title(title)
    )

  def curve(frame, power, x, y_title, tooltip=alt.Undefined) -> alt.Chart:
    return cell_chart(frame).transform_calculate(power=power).mark_line(
        strokeWidth=2.5
    ).encode(
        x=x, y=charts.percent_y('power:Q', y_title), color=group_color,
        tooltip=tooltip,
    )

  z_alpha_expr = (
      "quantileNormal(1 - (1 - sim_conf) / (sim_tail == 'Two-sided' ? 2 : 1))"
  )
  safe_cpic = 'max(sim_icpa, 1e-9)'
  expected = f'sim_spend / {safe_cpic}'
  threshold = f'({z_alpha_expr}) * datum.sigma'
  spend_curve = curve(
      cross('spend', np.linspace(0, max_spend, 61)),
      chance(f'datum.spend / {safe_cpic}', z_alpha_expr),
      alt.X('spend:Q', title=spend_label),
      'Chance of detecting the effect',
  )
  money = json.dumps(currency_symbol or '')
  money_title = f' ({currency_symbol})' if currency_symbol else ''
  point_data = (
      cell_chart(constants)
      .transform_calculate(
          spend='sim_spend',
          z_alpha=z_alpha_expr,
          power=chance(expected, 'datum.z_alpha'),
          warning=(
              f'abs(log({safe_cpic} / datum.reference_cpic)) > '
              f'log({ICPA_GUARDRAIL_FACTOR}) ? '
              "'⚠ Target iCPA is over "
              f"{ICPA_GUARDRAIL_FACTOR:g}x the planned iCPA (' + "
              f"{money} + format(datum.reference_cpic, ',.2f') + '): numbers "
              "may be unrealistic.' : ''"
          ),
      )
  )
  dot = point_data.mark_point(
      filled=True, size=90, color=charts.GREY_900
  ).encode(x='spend:Q', y='power:Q')
  warning = point_data.mark_text(
      align='left', baseline='top', fontSize=12, color=charts.RED,
      limit=_WIDTH - 8,
  ).encode(text='warning:N', x=alt.value(4), y=alt.value(-12))
  spend_panel = panel(
      spend_curve, dashed(point_data, 'spend:Q'), dot, warning, width=_WIDTH,
      title='How does the chance change with spend?',
  )
  # Chance at the chosen spend as the target iCPA moves away from the plan.
  max_icpa = max(3.0 * reference, 0.01)
  icpa_curve = curve(
      cross('icpa', np.linspace(max_icpa / 60, max_icpa, 60)),
      chance('sim_spend / datum.icpa', z_alpha_expr),
      alt.X('icpa:Q', title=f'Target iCPA{money_title}'),
      'Chance of a conclusive result',
      [
          alt.Tooltip('icpa:Q', title='iCPA', format=',.2f'),
          alt.Tooltip('power:Q', title='Chance', format='.0%'),
      ],
  )
  planning = alt.Chart(pd.DataFrame({'icpa': [reference]})).mark_rule(
      color=charts.GREY_700, strokeDash=[2, 2]
  ).encode(x='icpa:Q', tooltip=[alt.Tooltip('icpa:Q', title='Planning iCPA')])
  chosen = alt.Chart(pd.DataFrame({'one': [1]})).transform_calculate(
      icpa='sim_icpa'
  )
  icpa_panel = panel(
      icpa_curve, planning, dashed(chosen, 'icpa:Q'), width=_WIDTH - 80,
      title='How fast does the chance drop as iCPA rises?',
  )
  # Spread with no effect (grey) vs around the expected lift (blue) = power.
  z_grid = cross('z', np.round(np.linspace(-4, 4, 81), 2))
  noise_base = cell_chart(z_grid).transform_calculate(
      density='densityNormal(datum.z, 0, 1)',
      noise_x='datum.z * datum.sigma',
      lift_x=f'{expected} + datum.z * datum.sigma',
      threshold=threshold,
  )
  conversions_x = alt.X('x:Q', title='Incremental conversions over the test',
                        axis=alt.Axis(format='~s'))
  density_y = alt.Y('density:Q', title=None, axis=None)
  noise_line, lift_line = (
      noise_base.transform_calculate(x=f'datum.{x}').mark_line(
          color=color, strokeWidth=2
      ).encode(x=conversions_x, y=density_y)
      for x, color in (('noise_x', charts.GREY_500), ('lift_x', charts.BLUE))
  )
  lift_area = (
      noise_base.transform_calculate(x='datum.lift_x')
      .transform_filter('datum.x >= datum.threshold')
      .mark_area(color=charts.BLUE, opacity=0.25)
      .encode(x=conversions_x, y=density_y)
  )
  marker_data = cell_chart(constants).transform_calculate(
      threshold=threshold,
      lift=expected,
      peak='densityNormal(0, 0, 1)',
  )
  threshold_rule = marker_data.mark_rule(
      color=charts.RED, strokeDash=[4, 3]
  ).encode(x=alt.X('threshold:Q'))
  # Threshold label at the bottom, curve labels at the peaks: no overlap.
  threshold_label = marker_data.mark_text(
      align='left', dx=4, dy=-4, fontSize=10, color=charts.RED,
      baseline='bottom',
  ).encode(
      x='threshold:Q', y=alt.value(_HEIGHT),
      text=alt.value('Significance threshold'),
  )
  noise_label, lift_label = (
      marker_data.mark_text(
          dy=-8, fontSize=11, color=color, fontWeight=weight
      ).encode(x=x, y=alt.Y('peak:Q', axis=None), text=alt.value(text))
      for x, color, weight, text in (
          (alt.datum(0), charts.GREY_700, alt.Undefined, 'Noise alone'),
          ('lift:Q', charts.BLUE, 500, 'Expected lift'),
      )
  )
  noise_panel = panel(
      noise_line, lift_area, lift_line, threshold_rule, threshold_label,
      noise_label, lift_label, width=_WIDTH - 80,
      title='Would the expected lift stand out from noise?',
  )
  # Headline numbers, recomputed from the controls (plan_numbers formulas).
  high, medium = report_data.HIGH_CHANCE, report_data.MEDIUM_CHANCE

  def signed(f: str) -> str:
    return f"(datum.{f} >= 0 ? '+' : '') + format(datum.{f}, ',.0f')"

  def per_baseline(prefix: str, num: str, fmt: str) -> str:
    return (
        f"datum.baseline > 0 ? '{prefix}' + format({num} / datum.baseline,"
        f" '{fmt}') : ''"
    )

  def tier(a: str, b: str, c: str) -> str:
    return (
        f'datum.chance >= {high} ? {a} : datum.chance >= {medium} ? {b} : {c}'
    )

  numbers_base = cell_chart(constants).transform_calculate(
      z_alpha=z_alpha_expr,
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
      expected_detail=per_baseline('Lift: +', 'datum.expected', '.2%'),
      worst_value=signed('worst'),
      worst_detail=(
          "datum.worst <= 0 ? 'could be zero or negative' : 'iCPA up to ' +"
          f" {money} + format(sim_spend / datum.worst, ',.2f')"
      ),
      worst_label="'Worst case (' + format(sim_conf, '.0%') + ' conf.)'",
      needed_value=f"{money} + format(datum.needed, ',.0f')",
      needed_detail=per_baseline('MDE ', 'datum.mde_abs', '.1%'),
      max_value=f"{money} + format(datum.max_cpic, ',.2f')",
      max_detail=(
          f"datum.max_cpic >= {safe_cpic} ? '≥ target iCPA ✓' :"
          " 'below target iCPA ⚠'"
      ),
      max_color=(
          f"datum.max_cpic >= {safe_cpic} ? '{charts.GREEN}' :"
          f" '{charts.AMBER}'"
      ),
  )
  column = 200
  layers = []
  for i, (label, value, detail, color) in enumerate([
      ("'Conclusive-result chance'", 'chance_value', 'chance_detail',
       'chance_color'),
      ("'Expected extra conversions'", 'expected_value', 'expected_detail',
       None),
      ('datum.worst_label', 'worst_value', 'worst_detail', None),
      (json.dumps(f'{spend_label} for a {power:.0%} chance'), 'needed_value',
       'needed_detail', None),
      ("'Max detectable iCPA'", 'max_value', 'max_detail', 'max_color'),
  ]):
    x = alt.value(i * column)
    layers.append(
        numbers_base.transform_calculate(label=label)
        .mark_text(align='left', baseline='top', fontSize=12, fontWeight=500,
                   color=charts.GREY_900)
        .encode(text='label:N', x=x, y=alt.value(0))
    )
    value_color = (
        alt.Color(f'{color}:N', scale=None) if color
        else alt.value(charts.GREY_900)
    )
    layers.append(
        numbers_base.mark_text(align='left', baseline='top', fontSize=24)
        .encode(text=f'{value}:N', x=x, y=alt.value(18), color=value_color)
    )
    layers.append(
        numbers_base.mark_text(align='left', baseline='top', fontSize=12,
                               color=charts.GREY_700)
        .encode(text=f'{detail}:N', x=x, y=alt.value(48))
    )
  numbers = alt.layer(*layers).properties(width=column * 5, height=64)
  return alt.vconcat(
      numbers, alt.hconcat(noise_panel, icpa_panel, spacing=72), spend_panel,
      spacing=28,
  ).add_params(*params)


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
