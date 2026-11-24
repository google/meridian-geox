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

from typing import Optional

import altair as alt
from meridian_geox import api
from meridian_geox import charts
from meridian_geox import report_data
import numpy as np
import pandas as pd
from scipy import stats

_WIDTH = charts.DEFAULT_WIDTH
_HEIGHT = charts.DEFAULT_HEIGHT


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
