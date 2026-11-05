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

"""Interactive HTML design and analysis summaries for GeoX.

`DesignSummarizer` turns a `DesignSet` into a design 2-pager that compares
candidates against the user's goals. `AnalysisSummarizer` turns an
`AnalysisResult` into an analysis readout. Both write a single self-contained
HTML file and derive everything from existing objects; no model is re-run.
"""

import dataclasses
import datetime
import math
from typing import Callable, Optional, Sequence, Union

import altair as alt
from meridian_geox import analysis_charts
from meridian_geox import api
from meridian_geox import charts
from meridian_geox import design_charts
from meridian_geox import maps
from meridian_geox import report_data
from meridian_geox import report_text
from meridian_geox.templates import formatter
import pandas as pd

DesignGoals = report_data.DesignGoals
DateLike = Union[None, str, datetime.date, pd.Timestamp]

# Maximum number of candidates with their own detail panels.
MAX_SCOPED_CANDIDATES = 10
_RESERVED_TARGETING_COLUMNS = ('geo_name', 'group')

SELECT_CUE = 'Select a candidate to view its geo split and trade-offs.'


def _goal_tone(value: Optional[bool]) -> formatter.Tone:
  if value is None:
    return 'neutral'
  return 'success' if value else 'warning'


def _date(value: Optional[pd.Timestamp]) -> str:
  if value is None:
    return ''
  return report_text.short_date(value)


def _date_span(
    start: Optional[pd.Timestamp], end: Optional[pd.Timestamp]
) -> str:
  """Formats a date range, e.g. 'Wed, Apr 1 to Thu, Apr 30, 2020'."""
  if start is None or end is None:
    return ''
  if start.year == end.year:
    return f'{start:%a, %b} {start.day} to {_date(end)}'
  return f'{_date(start)} to {_date(end)}'


def _doc_link(label: str, url: str) -> str:
  return (
      f'<a href="{formatter.escape(url)}" target="_blank" rel="noopener">'
      f'{formatter.escape(label)}</a>'
  )


def _chart(
    chart: Optional[alt.TopLevelMixin],
    framing: charts.Framing,
    candidate: Optional[str] = None,
    audience: formatter.Audience = 'all',
    css_class: str = '',
) -> Optional[formatter.Block]:
  if chart is None:
    return None
  return formatter.Chart(
      charts.framed(chart, framing),
      framing=framing,
      candidate=candidate,
      audience=audience,
      css_class=css_class,
  )


def _reciprocal(value: float) -> float:
  return 1.0 / value if value > 0 else float('inf')


def _ordered_list(items: Sequence[str]) -> str:
  return (
      '<ol class="geox-steps">'
      + ''.join(f'<li>{formatter.escape(item)}</li>' for item in items)
      + '</ol>'
  )


def _id_chip(
    name: str, design_id: str, short_id: str, copy: str = ''
) -> str:
  """Option name with its short ID chip and an optional copy button."""
  return (
      f'<p class="geox-option-banner"><strong>{formatter.escape(name)}</strong>'
      f' <span class="geox-id-chip" title="{formatter.escape(design_id)}">'
      f'ID {formatter.escape(short_id)}</span>'
      + (f' {copy}' if copy else '')
      + '</p>'
  )


def _coverage_callout(
    coverage: maps.BoundaryCoverage, names: report_data.GeoNames
) -> Optional[formatter.Block]:
  if not coverage.configured or not coverage.checked or not coverage.unmatched:
    return None
  listed = ', '.join(names(g) for g in coverage.unmatched[:10])
  more = len(coverage.unmatched) - 10
  if more > 0:
    listed += f' and {more} more'
  return formatter.Callout(
      f'{len(coverage.unmatched)} geos have no matching boundary polygon and '
      f'are not drawn on the map: {listed}.',
      title='Some geos are missing from the map',
      tone='warning',
  )


def _map_block(
    design: api.Design,
    names: report_data.GeoNames,
    map_config: maps.MapConfig,
    framing: charts.Framing,
    candidate: Optional[str] = None,
    volumes: Optional[pd.Series] = None,
    show_legend: bool = True,
) -> formatter.Block:
  """Returns the boundary map, or a setup note without boundaries."""
  if not map_config.polygon_url:
    return formatter.Callout(
        maps.SETUP_NOTE, title='Map', tone='info', candidate=candidate
    )
  table = report_data.geo_table(design, names, volumes)
  scale = charts.group_scale(list(report_data.group_labels(design).values()))
  chart = maps.plot_geo_assignment(
      table, map_config, scale, title='', show_legend=show_legend
  )
  block = _chart(chart, framing, candidate=candidate)
  assert block is not None
  return block


def _geo_csv(table: pd.DataFrame) -> str:
  rows = table[table['cell'] != 'excluded'].to_dict('records')
  return formatter.to_csv(rows, ['location', 'geo_name', 'group', 'stratum'])


def _targeting_csv(table: pd.DataFrame, targeting_map: pd.DataFrame) -> str:
  targeting = targeting_map.copy()
  targeting['_key'] = [maps.normalize_geo_id(v) for v in targeting['location']]
  geos = table[table['cell'] != 'excluded'][['location', 'geo_name', 'group']]
  geos = geos.assign(_key=[maps.normalize_geo_id(v) for v in geos['location']])
  merged = geos.drop(columns='location').merge(
      targeting, on='_key', how='inner'
  )
  target_columns = [
      c for c in targeting_map.columns if c != 'location'
  ]
  columns = ['location', 'geo_name', 'group'] + target_columns
  return formatter.to_csv(merged[columns].to_dict('records'), columns)


def _map_legend(design: api.Design, table: pd.DataFrame) -> str:
  """Legend listing each group's size, share and largest geos."""
  experiment_type = report_data.single_experiment_type(design)
  treat_desc, control_desc = report_text.GROUP_DESCRIPTIONS.get(
      experiment_type, ('spend changed', 'business as usual')
  ) if experiment_type is not None else ('spend changed',
                                         'business as usual')
  labels = list(report_data.group_labels(design).values())
  colors = {label: charts.treatment_color(i)
            for i, label in enumerate(labels)}
  colors['Control'] = charts.CONTROL_COLOR
  colors['Excluded'] = charts.EXCLUDED_COLOR
  total = float(table['conversions'].sum()) or 1.0
  items = []
  for group, frame in table.groupby('group', sort=False):
    if group == 'Control':
      title = f"Control · {control_desc}"
    elif group == 'Excluded':
      title = 'Excluded · not in the test'
    elif len(labels) == 1:
      title = f'Treatment · {treat_desc}'
    else:
      title = str(group)
    names = list(frame.sort_values('share', ascending=False)['geo_name'])
    share = float(frame['conversions'].sum()) / total
    items.append(
        '<li><span class="geox-dot" style="background:'
        f'{colors.get(str(group), charts.GREY_500)}"></span><span><strong>'
        f'{formatter.escape(title)}</strong>: {len(frame)} geos,'
        f' {share:.0%} of conversions<br><span class="geox-detail">'
        + formatter.escape(report_text.name_list(
            [str(n) for n in names], 4))
        + '</span></span></li>'
    )
  return '<ul class="geox-legend">' + ''.join(items) + '</ul>'


# Tiles shown only in the Full report.
_ANALYST_TILES = ('Pretest fit', 'Pre-test AA check')

_CHECK_ICONS = {'pass': '✓', 'warn': '⚠', 'fail': '✗', 'info': 'ℹ'}


@dataclasses.dataclass(frozen=True)
class _Check:
  status: str
  text: str
  detail: Optional[str] = None


def _checks_html(checks: Sequence[_Check]) -> str:
  items = []
  for check in checks:
    detail = (
        f' <span class="geox-detail">({formatter.escape(check.detail)})</span>'
        if check.detail
        else ''
    )
    items.append(
        f'<li class="geox-check-{check.status}"><span class="geox-icon">'
        f'{_CHECK_ICONS[check.status]}</span><span>'
        f'{formatter.escape(check.text)}{detail}</span></li>'
    )
  return '<ul class="geox-checks">' + ''.join(items) + '</ul>'


def _chance_tone(chance: Optional[float]) -> formatter.Tone:
  label = report_data.chance_label(chance)
  if label == 'High':
    return 'success'
  if label == 'Medium':
    return 'warning'
  if label == 'Low':
    return 'error'
  return 'neutral'


def _download_link(label: str, payload: str, filename: str) -> str:
  button = formatter.Button(label, 'download', payload, filename)
  return (
      f'<a class="geox-button" href="{formatter.escape(button.href)}" '
      f'download="{formatter.escape(filename)}">{formatter.escape(label)}</a>'
  )


def _copy_button(label: str, payload: str) -> str:
  return (
      '<button type="button" class="geox-button geox-copy" '
      f'data-copy="{formatter.escape(payload)}">{formatter.escape(label)}'
      '</button>'
  )


class DesignSummarizer:
  """Builds the interactive design 2-pager for a `DesignSet`."""

  def __init__(
      self,
      design_set: api.DesignSet,
      goals: Optional[DesignGoals] = None,
      geo_metadata: Optional[pd.DataFrame] = None,
      map_config: maps.MapConfig = maps.MapConfig(),
      currency_symbol: str = '',
      target_mde: Optional[float] = None,
      target_budget: Optional[float] = None,
      target_cpic: Optional[float] = None,
      targeting_map: Optional[pd.DataFrame] = None,
      test_start: DateLike = None,
      cooldown_days: int = 14,
      ramp_up_days: int = 0,
  ):
    """Initializes the summarizer.

    Args:
      design_set: Output of `geox.run_design(...)`.
      goals: Optional bundle of target_mde, target_budget and target_cpic.
        Individual keyword arguments take precedence.
      geo_metadata: Optional display-name table (`location` plus `geo_name`
        or `name`).
      map_config: Boundary polygon configuration for map panels.
      currency_symbol: Symbol prefixed to monetary values.
      target_mde: Largest acceptable relative MDE (for example 0.05).
      target_budget: Budget goal; defaults to `Constraints.budget_constraint`.
      target_cpic: Target iCPA; defaults to the design's
        `cost_per_incremental_conversion`.
      targeting_map: Optional `location` to ad-platform target rows (for
        example postal codes), adding a targeting CSV download.
      test_start: Planned first test day (after the pretest window).
      ramp_up_days: First test days used to ramp spend up; shaded as
        ramp-up in the calendar and named in the launch plan.
      cooldown_days: Planned cooldown length in days.

    Raises:
      ValueError: On invalid inputs.
    """
    if design_set.design_metrics is None or design_set.design_metrics.empty:
      raise ValueError('design_set must have non-empty design_metrics.')
    if not design_set.designs:
      raise ValueError('design_set must contain at least one design.')
    base = goals or DesignGoals()
    self._goals = DesignGoals(
        target_mde=target_mde if target_mde is not None else base.target_mde,
        target_budget=(
            target_budget if target_budget is not None else base.target_budget
        ),
        target_cpic=(
            target_cpic if target_cpic is not None else base.target_cpic
        ),
    )
    if targeting_map is not None:
      columns = [str(c) for c in targeting_map.columns]
      if 'location' not in columns or len(columns) < 2:
        raise ValueError(
            'targeting_map must have a `location` column and at least one '
            'target column.'
        )
      reserved = [c for c in columns if c in _RESERVED_TARGETING_COLUMNS]
      if reserved:
        raise ValueError(
            f'targeting_map must not use reserved column names: {reserved}.'
        )
    self._design_set = design_set
    self._names = report_data.resolve_geo_metadata(geo_metadata)
    self._map_config = map_config
    self._currency = currency_symbol
    self._targeting_map = targeting_map
    self._cooldown_days = cooldown_days
    first = next(iter(design_set.designs.values()))
    self._schedule = report_data.experiment_schedule(
        first, test_start, cooldown_days, ramp_up_days
    )

  def _money(self, value: Optional[float]) -> str:
    return report_text.format_money(value, self._currency)

  def _short(self, value: Optional[float]) -> str:
    return report_text.short_money(value, self._currency)

  def generate_html(self) -> str:
    """Returns the self-contained HTML report."""
    return formatter.render_report(self._build_report())

  def to_iframe(self, height: int = 1100) -> str:
    """Returns the report wrapped in an iframe for notebook display.

    Use `display(HTML(summarizer.to_iframe()))` in Colab, Cider or Jupyter so
    charts, candidate selection and the Executive/Full toggle work.

    Args:
      height: Frame height in pixels.
    """
    return formatter.iframe(self.generate_html(), height)

  def _repr_html_(self) -> str:
    return self.to_iframe()

  def output_design_summary(self, filename: str, filepath: str) -> str:
    """Writes the report to `<filepath>/<filename>` and returns the path."""
    return formatter.write_html(self.generate_html(), filename, filepath)

  # ------------------------------------------------------------------------

  def _build_report(self) -> formatter.Report:
    designs = {str(k): v for k, v in self._design_set.designs.items()}
    first = next(iter(designs.values()))
    goals = report_data.default_goals(first, self._goals)
    summaries = report_data.design_summaries(self._design_set, goals)
    if not summaries:
      raise ValueError('No candidate designs could be summarized.')
    scoped = summaries[:MAX_SCOPED_CANDIDATES]
    best = summaries[0]
    best_design = designs[best.design_id]
    experiment_type = report_data.single_experiment_type(best_design)
    terms = report_text.spend_terms(experiment_type)
    all_geos = sorted(
        {str(g) for d in designs.values() for g in d.control_geos}
        | {
            str(g)
            for d in designs.values()
            for c in d.designs.values()
            for g in c.treatment_geos
        }
        | {str(g) for d in designs.values() for g in (d.excluded_geos or ())}
    )
    coverage = maps.boundary_coverage(all_geos, self._map_config)
    context = _DesignContext(
        designs=designs,
        summaries=summaries,
        scoped=scoped,
        goals=goals,
        terms=terms,
        coverage=coverage,
        n_feasible=sum(s.feasible for s in summaries),
    )
    # UX order: recommendation, then the comparison and its evidence, then
    # the launch plan and next steps, then details.
    cards = [
        self._recommended_card(context),
        self._compare_card(context),
        self._visuals_card(context),
        self._launch_card(context),
        self._details_card(context),
        self._quality_card(best_design),
    ]
    return formatter.Report(
        title='GeoX experiment design',
        subtitle=(
            'Compare candidate geo splits against your goals, then pick one to'
            ' launch.'
        ),
        kind='design',
        cards=cards,
        chips=self._chips(best_design, experiment_type, goals, all_geos),
        default_candidate=best.design_id,
    )

  def _chips(
      self,
      design: api.Design,
      experiment_type: Optional[api.ExperimentType],
      goals: DesignGoals,
      all_geos: Sequence[str],
  ) -> list[formatter.Chip]:
    chips = []
    region = self._map_config.region_name or 'Geos'
    chips.append(formatter.Chip('Region', f'{region} ({len(all_geos)} geos)'))
    pretest = report_data.pretest_range(design)
    if pretest is not None:
      chips.append(
          formatter.Chip('Pretest', report_text.date_range(*pretest))
      )
    config = design.design_config
    if config is not None:
      chips.append(
          formatter.Chip(
              'Duration', f'{config.experiment_duration.days} days'
          )
      )
    if experiment_type is None:
      types = ', '.join(
          f'{self._short_label(c.label)}'
          f' {report_text.EXPERIMENT_TYPE_LABELS[c.experiment_type].lower()}'
          for c, _ in self._changes(design)
      )
      chips.append(formatter.Chip('Types', types))
      chips.append(formatter.Chip('Cells', str(len(design.designs))))
    else:
      chips.append(
          formatter.Chip(
              'Type',
              report_text.EXPERIMENT_TYPE_LABELS[experiment_type]
              + ' · '
              + report_text.EXPERIMENT_TYPE_EXPLANATIONS[experiment_type],
          )
      )
    if config is not None:
      method = config.methodology.name
      rule = config.geo_assignment_rule.name.replace('_', ' ').title()
      chips.append(formatter.Chip('Method', f'{method} · {rule}'))
    goal_parts = []
    if goals.target_mde is not None:
      goal_parts.append(f'MDE ≤ {report_text.format_percent(goals.target_mde)}')
    if goals.target_budget is not None:
      goal_parts.append(f'budget ≤ {self._short(goals.target_budget)}')
    if goals.target_cpic is not None:
      goal_parts.append(
          'target iCPA '
          + report_text.format_money(goals.target_cpic, self._currency, 2)
      )
    if goal_parts:
      chips.append(formatter.Chip('Goals', ', '.join(goal_parts)))
    return chips

  # ---------------------------------------------------------------- recommend

  @staticmethod
  def _short_label(label: str) -> str:
    """'Treatment 1 · ads paused' -> 'Treatment 1'."""
    return label.split(' · ')[0]

  def _changes(
      self, design: api.Design
  ) -> list[tuple[report_data.SpendChange, Optional[float]]]:
    """Planned spend change per cell with its treatment BAU spend."""
    out = []
    for change in report_data.spend_changes(design):
      bau = (
          abs(change.amount / change.planned_change)
          if change.planned_change
          else None
      )
      out.append((change, bau))
    return out

  def _change_clause(
      self, change: report_data.SpendChange, where: str
  ) -> str:
    """'adds +€14.1k of extra spend in the treatment geos (+34% …)'."""
    amount = self._short(abs(change.amount))
    pct = (
        report_text.format_percent(abs(change.planned_change), 0)
        if change.planned_change is not None
        else ''
    )
    if change.experiment_type == api.ExperimentType.HEAVY_UP:
      share = f' (+{pct} of their BAU spend)' if pct else ''
      return f'adds +{amount} of extra spend in {where}{share}'
    if change.experiment_type == api.ExperimentType.GO_DARK:
      share = f' ({pct} of their BAU spend)' if pct else ''
      return f'pauses {amount} of spend in {where}{share}'
    return f'needs about {amount} of campaign spend in {where}'

  def _narrative(
      self,
      s: report_data.CandidateSummary,
      design: api.Design,
      context: '_DesignContext',
      plan: Optional[report_data.PlanNumbers],
  ) -> str:
    goals = context.goals
    alpha, sides = report_data.alpha_and_sides(design)
    days = design.design_config.experiment_duration.days if (
        design.design_config
    ) else 0
    power = plan.power if plan else 0.8
    parts = [
        f'It detects a lift of {report_text.format_percent(s.max_mde)} or more'
        f' in treatment-geo conversions {power:.0%} of the time (alpha'
        f' {alpha:g}, {sides}).',
    ]
    changes = self._changes(design)
    all_holdback = all(
        c.experiment_type == api.ExperimentType.HOLDBACK for c, _ in changes
    )
    if all_holdback:
      parts.append(
          f'It needs about {self._short(s.total_spend)} of'
          f' {context.terms.label.lower()} in the treatment geos over the'
          f' {days}-day test.'
      )
    elif len(changes) == 1:
      parts.append(
          f'It {self._change_clause(changes[0][0], "the treatment geos")}'
          f' over the {days}-day test.'
      )
    else:
      parts.append(
          f'Spend over the {days}-day test: '
          + '; '.join(
              f'{self._short_label(c.label)} {self._change_clause(c, "its geos")}'
              for c, _ in changes
          )
          + '.'
      )
    if plan is not None:
      cpic = report_text.format_money(plan.cpic, self._currency, 2)
      if goals.target_budget is not None and any(
          c.experiment_type == api.ExperimentType.HOLDBACK
          for c in s.cells.values()
      ):
        parts.append(
            f'Spending your {self._short(goals.target_budget)} budget goal at'
            f' an iCPA of {cpic} gives a {plan.chance:.0%} chance of a'
            ' statistically significant result.'
        )
      else:
        parts.append(
            f'If each incremental conversion costs your target iCPA of {cpic},'
            f' the chance of a statistically significant result is'
            f' {plan.chance:.0%}.'
        )
    missed = [
        name
        for name, value in (
            ('MDE goal', s.meets_mde),
            ('budget goal', s.meets_budget),
            ('iCPA goal', s.meets_cpic),
        )
        if value is False
    ]
    n = len(context.summaries)
    if not goals.any_set:
      parts.append('Set goals to rank candidates against them.')
    elif missed:
      parts.append(
          f'This candidate misses your {" and ".join(missed)};'
          f' {context.n_feasible} of {n} candidates meet all of them.'
      )
    else:
      parts.append(
          f'This candidate meets your goals; {context.n_feasible} of {n}'
          ' candidates meet all of them.'
      )
    return ' '.join(parts)

  def _design_tiles(
      self,
      s: report_data.CandidateSummary,
      design: api.Design,
      context: '_DesignContext',
      plan: Optional[report_data.PlanNumbers],
  ) -> list[formatter.Tile]:
    goals = context.goals
    tiles = []
    if plan is not None:
      label = report_data.chance_label(plan.chance)
      detail = (
          f'{plan.chance:.0%} chance at your budget goal'
          if goals.target_budget is not None
          else f'{plan.chance:.0%} chance at your target iCPA'
          if goals.target_cpic is not None
          else f'{plan.chance:.0%} chance at the planned spend'
      )
      tiles.append(
          formatter.Tile(
              'Feasibility',
              f'{label} chance',
              detail=detail,
              ci=report_text.CHANCE_LEGEND,
              tone=_chance_tone(plan.chance),
          )
      )
    spend_detail = None
    if s.meets_budget is not None and goals.target_budget is not None:
      gap = s.total_spend - goals.target_budget
      spend_detail = (
          f'within your {self._short(goals.target_budget)} goal'
          if gap <= 0
          else f'{self._short(gap)} over your'
          f' {self._short(goals.target_budget)} goal'
      )
    spend_formula = None
    spend_value = self._short(s.total_spend)
    changes = self._changes(design)
    days = design.design_config.experiment_duration.days if (
        design.design_config
    ) else 0
    if len(changes) > 1:
      spend_formula = ' · '.join(
          f'{self._short_label(c.label)}: '
          + ('−' if c.amount < 0 or c.experiment_type
             == api.ExperimentType.GO_DARK else '+')
          + self._short(abs(c.amount))
          + (
              f' ({report_text.format_percent(c.planned_change, 0, True)})'
              if c.planned_change is not None
              else ''
          )
          for c, _ in changes
      )
    elif changes and changes[0][0].experiment_type in (
        api.ExperimentType.HEAVY_UP, api.ExperimentType.GO_DARK
    ):
      change, bau = changes[0]
      pct = report_text.format_percent(abs(change.planned_change or 0), 0)
      if change.experiment_type == api.ExperimentType.HEAVY_UP:
        spend_value = '+' + self._short(abs(change.amount))
      spend_detail = f'{pct} of treatment BAU spend'
      if bau is not None:
        spend_formula = (
            f'= {self._short(bau)} BAU spend in treatment geos over'
            f' {days} days × {pct}'
        )
    elif plan is not None and len(s.cells) == 1:
      spend_formula = (
          f'= {report_text.format_number(plan.mde_abs)} incremental'
          ' conversions (MDE) × '
          + report_text.format_money(plan.cpic, self._currency, 2)
          + ' iCPA'
      )
    tiles.append(
        formatter.Tile(
            context.terms.label,
            spend_value,
            detail=spend_detail,
            ci=spend_formula,
            tone=_goal_tone(s.meets_budget),
        )
    )
    mde_detail = 'Smallest lift the test reliably detects'
    if goals.target_mde is not None:
      verb = 'meets' if s.meets_mde else 'misses'
      mde_detail = (
          f'{verb} your {report_text.format_percent(goals.target_mde)} goal'
      )
    days = design.design_config.experiment_duration.days if (
        design.design_config
    ) else 0
    mde_abs = sum(
        c.mde_abs for c in s.cells.values() if not math.isnan(c.mde_abs)
    )
    tiles.append(
        formatter.Tile(
            'Minimum detectable effect',
            report_text.format_percent(s.max_mde),
            detail=mde_detail,
            ci=(
                ' · '.join(
                    f'{self._short_label(c.label)}:'
                    f' {report_text.format_percent(c.mde)}'
                    for c in s.cells.values()
                )
                if len(s.cells) > 1
                else f'≈ {report_text.format_number(mde_abs)} incremental'
                f' conversions over {days} days'
            ),
            tone=_goal_tone(s.meets_mde),
        )
    )
    r2 = min(c.r2 for c in s.cells.values())
    minimum = report_data.min_r2(design)
    good_fit = not math.isnan(r2) and r2 >= minimum
    tiles.append(
        formatter.Tile(
            'Pretest fit',
            '✓ Good fit' if good_fit else '⚠ Weak fit',
            detail=(
                'Control geos predict treatment-geo conversions closely'
                ' before the test.'
                if good_fit
                else 'Control geos predict treatment geos less closely than'
                ' required.'
            ),
            ci=f'R² {r2:.2f}; minimum {minimum:g}',
            tone='success' if good_fit else 'warning',
        )
    )
    alpha, _ = report_data.alpha_and_sides(design)
    passed = s.min_aa_p_value >= alpha
    drift = report_data.aa_drift(design, next(iter(s.cells)))
    window = (
        f' over {report_text.date_range(drift.start, drift.end)}'
        if drift is not None
        else ''
    )
    tiles.append(
        formatter.Tile(
            'Pre-test AA check',
            '✓ Passed' if passed else '✗ Failed',
            detail=(
                'No pre-existing difference between treatment and control'
                f' geos{window}.'
                if passed
                else f'Treatment and control geos already differed{window}.'
            ),
            ci=f'AA p-value {s.min_aa_p_value:.2f}; passes at ≥ {alpha:g}',
            tone='success' if passed else 'error',
        )
    )
    n_treatment = sum(c.treatment_geo_count for c in s.cells.values())
    n_excluded = len(design.excluded_geos or ())
    share = sum(c.treatment_share for c in s.cells.values())
    tiles.append(
        formatter.Tile(
            'Regions split',
            f'{n_treatment} / {s.control_geo_count} / {n_excluded}',
            detail='treatment / control / excluded',
            ci=f'Treatment geos hold {share:.1%} of conversions'
            + (
                ' ('
                + ', '.join(
                    f'{self._short_label(c.label)}: {c.treatment_geo_count}'
                    for c in s.cells.values()
                )
                + ')'
                if len(s.cells) > 1
                else ''
            ),
        )
    )
    return tiles

  def _checks(
      self,
      s: report_data.CandidateSummary,
      design: api.Design,
      context: '_DesignContext',
      plan: Optional[report_data.PlanNumbers],
  ) -> list[_Check]:
    goals = context.goals
    checks = []
    r2 = min(c.r2 for c in s.cells.values())
    minimum = report_data.min_r2(design)
    if not math.isnan(r2) and r2 >= minimum:
      checks.append(_Check(
          'pass',
          'Control geos predict treatment-geo conversions closely before the'
          ' test.',
          f'R² {r2:.2f}; minimum {minimum:g}',
      ))
    else:
      checks.append(_Check(
          'warn',
          'Control geos predict treatment geos less closely than required.',
          f'R² {r2:.2f}; minimum {minimum:g}',
      ))
    alpha, _ = report_data.alpha_and_sides(design)
    drift = report_data.aa_drift(design, next(iter(s.cells)))
    window = (
        f' over {report_text.date_range(drift.start, drift.end)}'
        if drift is not None
        else ''
    )
    if s.min_aa_p_value >= alpha:
      checks.append(_Check(
          'pass',
          'No pre-existing difference between treatment and control geos'
          f'{window}.',
          f'AA p-value {s.min_aa_p_value:.2f}; passes at ≥ {alpha:g}',
      ))
    else:
      checks.append(_Check(
          'fail',
          f'Treatment and control geos already differed{window}.',
          f'AA p-value {s.min_aa_p_value:.2f}; passes at ≥ {alpha:g}',
      ))
    if goals.target_mde is not None:
      checks.append(_Check(
          'pass' if s.meets_mde else 'warn',
          f'MDE {report_text.format_percent(s.max_mde)} vs. goal'
          f' {report_text.format_percent(goals.target_mde)}',
      ))
    holdback = [
        c for c in s.cells.values()
        if c.experiment_type == api.ExperimentType.HOLDBACK
    ]
    if goals.target_budget is not None and holdback:
      checks.append(_Check(
          'pass' if s.meets_budget else 'warn',
          f'{context.terms.label} {self._short(s.total_spend)} vs. budget goal'
          f' {self._short(goals.target_budget)}',
      ))
      share = sum(c.treatment_share for c in holdback)
      if plan is not None and share > 0:
        treatment_budget = goals.target_budget * share
        chance = report_data.chance_at(
            s, design, treatment_budget, plan.cpic
        )
        chance_text = (
            f', for a {chance:.0%} chance of a conclusive result'
            if chance is not None
            else ''
        )
        checks.append(_Check(
            'info',
            'Enter the spend planned for the treatment geos only. If'
            f' {self._short(goals.target_budget)} were a national budget, only'
            f' about {self._short(treatment_budget)} would run in the'
            f' treatment geos ({share:.0%} of conversions){chance_text}.',
        ))
    share = sum(c.treatment_share for c in s.cells.values())
    checks.append(_Check(
        'pass',
        f'Treatment geos hold {share:.1%} of conversions in the test geos'
        ' (excluded geos aside)',
    ))
    coverage = context.coverage
    if coverage.configured and coverage.checked:
      total = len(coverage.matched) + len(coverage.unmatched)
      if coverage.unmatched:
        checks.append(_Check(
            'warn',
            f'{len(coverage.matched)} of {total} geos matched a boundary'
            ' polygon; missing: '
            + report_text.name_list(
                [self._names(g) for g in coverage.unmatched], 5
            ),
        ))
      else:
        checks.append(_Check(
            'pass', f'{total} of {total} geos matched a boundary polygon'
        ))
    elif not coverage.configured:
      checks.append(_Check(
          'info', 'No boundary file was passed, so maps are omitted.'
      ))
    quality = design.quality_check_result
    flagged_geos = sorted(str(g) for g in (quality.outlier_geos if quality
                                           else ()))
    user_excluded = sorted(
        str(g)
        for g in (design.constraints.excluded_geos if design.constraints
                  else ())
        if str(g) not in flagged_geos
    )
    if user_excluded:
      checks.append(_Check(
          'info',
          'Excluded by you: '
          + report_text.name_list([self._names(g) for g in user_excluded], 5),
      ))
    if flagged_geos:
      checks.append(_Check(
          'warn',
          f'{report_text.plural(len(flagged_geos), "geo")} flagged by data'
          ' quality checks (see Pretest data quality)',
      ))
    excluded_dates = design.excluded_dates or ()
    if excluded_dates:
      checks.append(_Check(
          'warn',
          f'{report_text.plural(len(excluded_dates), "pretest date")} excluded'
          ' (see Pretest data quality)',
      ))
    return checks

  def _budget_tip(
      self,
      s: report_data.CandidateSummary,
      design: api.Design,
      context: '_DesignContext',
      plan: Optional[report_data.PlanNumbers],
  ) -> Optional[str]:
    """Explains that a holdback budget goal is treatment-geo spend only."""
    budget = context.goals.target_budget
    holdback = [
        c for c in s.cells.values()
        if c.experiment_type == api.ExperimentType.HOLDBACK
    ]
    if budget is None or not holdback or plan is None:
      return None
    share = sum(c.treatment_share for c in holdback)
    if share <= 0:
      return None
    treatment_budget = budget * share
    chance = report_data.chance_at(s, design, treatment_budget, plan.cpic)
    chance_text = (
        f', for a {chance:.0%} chance of a conclusive result'
        if chance is not None
        else ''
    )
    return (
        'Enter the spend planned for the treatment geos only. If'
        f' {self._short(budget)} were a national budget, only about'
        f' {self._short(treatment_budget)} would run in the treatment geos'
        f' ({share:.0%} of conversions){chance_text}.'
    )

  def _recommended_card(self, context: '_DesignContext') -> formatter.Card:
    blocks: list[formatter.Block] = []
    experiment_type = report_data.single_experiment_type(
        context.designs[context.summaries[0].design_id]
    )
    for s in context.scoped:
      design = context.designs[s.design_id]
      plan = report_data.plan_numbers(s, design, context.goals)
      chance_text = (
          f': {report_data.chance_label(plan.chance).lower()} chance of a'
          ' conclusive result'
          if plan is not None
          else ''
      )
      banner_class = 'geox-banner' + (
          ' geox-banner-warn'
          if plan is not None and plan.chance < 0.8
          else ''
      )
      blocks.append(formatter.Html(
          f'<div class="{banner_class}">'
          + _id_chip(
              s.name + chance_text,
              s.design_id,
              s.short_id,
              copy=_copy_button('Copy ID', s.design_id),
          )
          + '<p class="geox-narrative">'
          + formatter.escape(self._narrative(s, design, context, plan))
          + ' '
          + _doc_link('Learn more', report_text.DOCS_INTRO_TO_DESIGN)
          + '</p></div>',
          candidate=s.design_id,
      ))
      tiles = self._design_tiles(s, design, context, plan)
      blocks.append(formatter.Tiles(
          [t for t in tiles if t.label not in _ANALYST_TILES],
          candidate=s.design_id,
      ))
      blocks.append(formatter.Tiles(
          [t for t in tiles if t.label in _ANALYST_TILES],
          candidate=s.design_id,
          audience='technical',
      ))
      if experiment_type is not None:
        blocks.append(formatter.Callout(
            report_text.SPEND_DEFINITIONS[experiment_type],
            title=context.terms.label,
            tone='info',
            candidate=s.design_id,
            links=[('Experiment types', report_text.DOCS_EXPERIMENT_TYPES)],
        ))
      tip = self._budget_tip(s, design, context, plan)
      if tip is not None:
        blocks.append(formatter.Callout(
            tip, title='Budget goal = treatment-geo spend', tone='neutral',
            candidate=s.design_id,
        ))
      if report_data.looks_national(design, context.goals.target_budget):
        blocks.append(formatter.Callout(
            'The budget goal is at least 90% of all spend across the market'
            ' over the test window. It looks like a national budget; a'
            ' holdback test only needs the spend for the treatment geos.',
            title='Check the budget scope',
            tone='warning',
            candidate=s.design_id,
        ))
      blocks.append(formatter.Html(
          '<h3>Design checks</h3>'
          + _checks_html(self._checks(s, design, context, plan)),
          candidate=s.design_id,
          audience='technical',
      ))
    return formatter.Card(
        'recommended',
        'Recommended design',
        blocks,
        subtitle='Which design to run, why, and whether it meets your goals.',
    )

  # ------------------------------------------------------------------ visuals

  def _regions_sentence(
      self, s: report_data.CandidateSummary, table: pd.DataFrame
  ) -> str:
    treatment = table[~table['cell'].isin(['control', 'excluded'])]
    share = float(treatment['share'].sum())
    largest = (
        treatment.sort_values('share', ascending=False).iloc[0]
        if not treatment.empty
        else None
    )
    largest_text = ''
    if largest is not None and share > 0:
      largest_text = (
          f' (largest, {largest["geo_name"]}, is'
          f' {float(largest["share"]) / share:.0%} of the group)'
      )
    return (
        f'{s.name}: treatment has {len(treatment)} geos with {share:.0%} of'
        f' pretest conversions{largest_text}; {s.control_geo_count} control'
        ' geos.'
    )

  def _visuals_card(self, context: '_DesignContext') -> formatter.Card:
    blocks: list[formatter.Block] = [
        formatter.Callout(
            'Matched treatment and control regions run side by side while the'
            ' sensitivity curve shows the smallest campaign lift this design'
            ' can reliably confirm.',
            tone='neutral',
        )
    ]
    for s in context.scoped:
      design = context.designs[s.design_id]
      table = report_data.geo_table(design, self._names)
      regions = self._regions_sentence(s, table)
      map_block = _map_block(
          design,
          self._names,
          self._map_config,
          charts.Framing(
              'Where would the test run?',
              reading_guide=(
                  'Treatment geos change spend during the test; control geos'
                  ' stay business as usual and predict what treatment would'
                  ' have done otherwise.'
              ),
              takeaway=regions,
          ),
          candidate=s.design_id,
          # The group list next to the map doubles as its legend.
          show_legend=False,
      )
      plan = report_data.plan_numbers(s, design, context.goals)
      power = plan.power if plan else 0.8
      sensitivity = _chart(
          design_charts.sensitivity_curve(
              s, design, context.goals.target_mde
          ),
          charts.Framing(
              'What lift size can this test reliably detect?',
              reading_guide=(
                  'The curve shows the chance of a significant result for each'
                  ' true lift. Where it crosses the dotted line is the minimum'
                  ' detectable effect (MDE).'
              ),
              takeaway=(
                  f'{s.name} detects lifts of'
                  f' {report_text.format_percent(s.max_mde)} or more'
                  f' {power:.0%} of the time; smaller lifts may be missed.'
              ),
          ),
      )
      left: list[formatter.Block] = [
          map_block,
          formatter.Html(_map_legend(design, table)),
      ]
      row: list[formatter.Block] = [formatter.Row(left)]
      if sensitivity is not None:
        row.append(sensitivity)
      blocks.append(formatter.Row(row, candidate=s.design_id))
    coverage = _coverage_callout(context.coverage, self._names)
    if coverage is not None:
      blocks.append(coverage)
    return formatter.Card(
        'visuals',
        'Where the test runs and what it can detect',
        blocks,
        cue=SELECT_CUE,
    )

  # ------------------------------------------------------------------ compare

  def _compare_card(self, context: '_DesignContext') -> formatter.Card:
    goals = context.goals
    rows, tones = [], []
    for s in context.scoped:
      design = context.designs[s.design_id]
      plan = report_data.plan_numbers(s, design, goals)
      alpha, _ = report_data.alpha_and_sides(design)
      chance = (
          f'{report_data.chance_label(plan.chance)} · {plan.chance:.0%}'
          if plan is not None
          else '—'
      )
      r2 = min(c.r2 for c in s.cells.values())
      rows.append([
          ('★ ' if s.rank == 1 else '') + s.name,
          s.short_id,
          chance,
          report_text.format_percent(s.max_mde),
          self._short(s.total_spend),
          report_text.format_money(s.max_detectable_cpic, self._currency, 2),
          f'{sum(c.treatment_share for c in s.cells.values()):.0%}',
          f'{r2:.2f}',
          ('✓ ' if s.min_aa_p_value >= alpha else '✗ ')
          + f'{s.min_aa_p_value:.2f}',
          'Meets all goals' if s.feasible else 'Misses a goal',
      ])
      tones.append('success' if s.feasible else 'warning')
    table = formatter.Table(
        columns=[
            'Option', 'ID', 'Chance', 'MDE', context.terms.label, 'Max iCPA',
            'Share', 'Fit R²', 'A/A', 'Goals',
        ],
        rows=rows,
        row_ids=[s.design_id for s in context.scoped],
        selectable=True,
        row_tones=tones,
        caption=(
            'Click a row of the ranked table or a bubble to inspect that'
            ' candidate. The map, largest geos, pretest fit, AA check and'
            ' simulator update, and so do the design summary above and the'
            ' export list below. ★ marks the recommended design.'
        ),
    )
    best = context.summaries[0]
    if best.feasible:
      zone = 'meets your goals'
    elif best.meets_budget is False and goals.target_budget is not None:
      zone = (
          f'needs more than your {self._short(goals.target_budget)} spend goal'
      )
    else:
      zone = 'misses at least one goal'
    tradeoff = _chart(
        design_charts.tradeoff_toggle_chart(
            context.scoped, goals, self._currency, context.terms.label
        ),
        charts.Framing(
            'Which design detects the smallest lift for the least spend?',
            reading_guide=(
                'Each dot is a candidate. Use the toggle to switch between'
                ' detectable lift (lower is better) and max detectable iCPA'
                ' (higher is better). Green dots meet every goal; dashed lines'
                ' are your goals. Click a dot to select it.'
            ),
            takeaway=(
                f'{best.name} detects lifts of'
                f' {report_text.format_percent(best.max_mde)} or more with'
                f' {self._short(best.total_spend)} of'
                f' {context.terms.label.lower()}; it {zone}.'
            ),
        ),
    )
    blocks: list[formatter.Block] = [table]
    if tradeoff is not None:
      blocks.append(tradeoff)
    for s in context.scoped:
      blocks.extend(self._evidence_blocks(s, context))
    blocks.append(formatter.Html(
        '<h3>What if the spend or target iCPA changes?</h3>'
        '<p>Use the controls first; the chance of a conclusive result and'
        ' the spend needed update below them.</p>'
    ))
    for s in context.scoped:
      blocks.extend(self._simulator_blocks(s, context))
    blocks.append(formatter.Callout(
        report_text.ICPA_TIP,
        title='Setting the target iCPA',
        tone='info',
        links=[
            ('How to choose it', report_text.DOCS_TROUBLESHOOTING),
            ('Glossary: iCPA', report_text.DOCS_ICPA_GLOSSARY),
        ],
    ))
    return formatter.Card(
        'compare',
        'Compare candidate designs',
        blocks,
        subtitle=(
            'Why the recommendation, and how it compares with the other'
            ' options.'
        ),
        audience='technical',
        cue=SELECT_CUE,
    )

  def _evidence_blocks(
      self, s: report_data.CandidateSummary, context: '_DesignContext'
  ) -> list[formatter.Block]:
    design = context.designs[s.design_id]
    table = report_data.geo_table(design, self._names)
    scale = charts.group_scale(list(report_data.group_labels(design).values()))
    in_test = table[table['cell'] != 'excluded'].sort_values(
        'share', ascending=False
    ).head(22)
    n_largest = len(in_test)
    n_treated = int((~in_test['cell'].isin(['control'])).sum())
    tracks = in_test[
        (in_test['share'] > 0)
        & ((in_test['spend_share'] - in_test['share']).abs()
           <= 0.5 * in_test['share'])
    ]
    spend_text = ''
    if float(in_test['spend_share'].sum()) > 0:
      spend_text = (
          ' Spend tracks conversions in all of them.'
          if len(tracks) == n_largest
          else f' Spend tracks conversions in {len(tracks)} of them.'
      )
    r2 = min(c.r2 for c in s.cells.values())
    minimum = report_data.min_r2(design)
    row: list[Optional[formatter.Block]] = [
        _chart(
            design_charts.largest_geos_chart(table, scale),
            charts.Framing(
                'Do the biggest geos split sensibly, and does spend follow?',
                reading_guide=(
                    'Solid bars are each geo\'s share of pretest conversions,'
                    ' light bars its share of spend. A few very large geos can'
                    ' dominate the result.'
                ),
                takeaway=(
                    f'{s.name} puts {n_treated} of the {n_largest} largest geos'
                    f' in treatment.{spend_text}'
                ),
            ),
        ),
        _chart(
            design_charts.pretest_fit_chart(design),
            charts.Framing(
                'Does the model track treatment geos before the test?',
                reading_guide=(
                    'Before the test, control geos should predict treatment'
                    ' closely. Red lines mark excluded dates.'
                ),
                takeaway=(
                    f'{s.name}: '
                    + (
                        'control geos predict treatment-geo conversions'
                        ' closely before the test'
                        if r2 >= minimum
                        else 'the pretest fit is weaker than required'
                    )
                    + f' (R² {r2:.2f}; minimum {minimum:g}).'
                ),
            ),
        ),
    ]
    blocks: list[formatter.Block] = [
        formatter.Row([b for b in row if b is not None], candidate=s.design_id)
    ]
    cell = next(iter(s.cells))
    drift = report_data.aa_drift(design, cell)
    if drift is not None:
      mde_abs = s.cells[cell].mde_abs
      ratio = abs(drift.final) / mde_abs if mde_abs and mde_abs > 0 else None
      ratio_text = f' ({ratio:.0%} of the MDE' if ratio is not None else ' ('
      verdict = (
          'Noise stays well inside what the test can detect.'
          if ratio is not None and ratio < 0.5
          else 'Noise alone comes close to the detectable effect.'
      )
      block = _chart(
          design_charts.aa_drift_chart(drift, mde_abs),
          charts.Framing(
              'Would noise alone have looked like a lift before the test?',
              reading_guide=(
                  'Cumulative gap between observed and predicted treatment-geo'
                  ' conversions over the last test-length of pretest, when'
                  ' nothing changed. Dashed red lines mark ± the MDE.'
              ),
              takeaway=(
                  f'{s.name}: over'
                  f' {report_text.date_range(drift.start, drift.end)},'
                  ' treatment geos drifted'
                  f' {report_text.format_signed(drift.final)} conversions from'
                  f' the counterfactual{ratio_text}; AA p ='
                  f' {s.min_aa_p_value:.2f}). {verdict}'
              ),
          ),
          candidate=s.design_id,
      )
      if block is not None:
        blocks.append(block)
    return blocks

  def _simulator_blocks(
      self, s: report_data.CandidateSummary, context: '_DesignContext'
  ) -> list[formatter.Block]:
    design = context.designs[s.design_id]
    goals = context.goals
    plan = report_data.plan_numbers(s, design, goals)
    blocks: list[formatter.Block] = []
    reference = goals.target_cpic
    config = design.design_config
    if config is not None and isinstance(
        config.cost_per_incremental_conversion, dict
    ):
      values = list(config.cost_per_incremental_conversion.values())
      if values:
        reference = float(min(values))
    if plan is not None:
      cell = s.cells[plan.cell]
      baseline = cell.mde_abs / cell.mde if cell.mde else float('nan')
      lift = plan.expected / baseline if baseline else float('nan')
      worst_detail = (
          'could be zero or negative'
          if plan.worst_case <= 0
          else 'iCPA up to '
          + report_text.format_money(
              plan.spend / plan.worst_case, self._currency, 2
          )
      )
      target_ok = goals.target_cpic is None or (
          plan.max_cpic >= goals.target_cpic
      )
      blocks.append(formatter.Tiles(
          [
              formatter.Tile(
                  'Conclusive-result chance',
                  f'{plan.chance:.0%}',
                  detail=report_data.chance_label(plan.chance),
                  tone=_chance_tone(plan.chance),
              ),
              formatter.Tile(
                  'Expected extra conversions',
                  report_text.format_signed(plan.expected),
                  detail=(
                      f'Lift: {report_text.format_percent(lift, 2, True)}'
                  ),
              ),
              formatter.Tile(
                  f'Worst case ({1 - report_data.alpha_and_sides(design)[0]:.0%}'
                  ' conf.)',
                  report_text.format_signed(plan.worst_case),
                  detail=worst_detail,
                  tone='neutral' if plan.worst_case > 0 else 'warning',
              ),
              formatter.Tile(
                  f'Spend for a {plan.power:.0%} chance',
                  self._money(plan.spend_for_power),
                  detail=f'MDE {report_text.format_percent(s.max_mde)}',
              ),
              formatter.Tile(
                  'Max detectable iCPA',
                  report_text.format_money(plan.max_cpic, self._currency, 2),
                  detail=(
                      '≥ target iCPA ✓' if target_ok else 'below target iCPA ⚠'
                  ),
                  tone='success' if target_ok else 'warning',
              ),
          ],
          candidate=s.design_id,
      ))
    takeaway = None
    if plan is not None:
      takeaway = (
          f'With {self._money(plan.spend)} of'
          f' {context.terms.label.lower()} at a'
          f' {report_text.format_money(plan.cpic, self._currency, 2)} iCPA:'
          f' the chance of a conclusive result is {plan.chance:.0%}; a'
          f' {plan.power:.0%} chance needs {self._money(plan.spend_for_power)}.'
      )
    block = _chart(
        design_charts.power_simulator(
            s, design, goals, self._currency, context.terms.label, reference
        ),
        charts.Framing(
            'How fast does the chance drop as spend falls or iCPA rises?',
            reading_guide=(
                'Left: chance of a conclusive result at each spend level for'
                ' the chosen target iCPA. Right: chance at the chosen spend as'
                ' the target iCPA changes. A warning appears when the target'
                ' iCPA is far from the planning iCPA.'
            ),
            takeaway=takeaway,
        ),
        candidate=s.design_id,
        css_class=design_charts.SIMULATOR_CLASS,
    )
    if block is not None:
      blocks.append(block)
    return blocks

  # ------------------------------------------------------------------- launch

  def _how_it_works(
      self, s: report_data.CandidateSummary, design: api.Design
  ) -> str:
    """Plain-language explanation of why the split isolates ad impact."""
    n_treatment = sum(c.treatment_geo_count for c in s.cells.values())
    share = sum(c.treatment_share for c in s.cells.values())
    days = design.design_config.experiment_duration.days if (
        design.design_config
    ) else 0
    alpha, _ = report_data.alpha_and_sides(design)
    validation = (
        'Pre-test validation confirms that the control group closely tracks'
        ' treatment regions with no pre-existing bias'
        if s.min_aa_p_value >= alpha
        else 'Pre-test validation found a pre-existing difference between the'
        ' groups, so review the A/A check before launching'
    )
    return (
        '<h3>How this geo test isolates true ad impact</h3><p>This study'
        ' divides your market into matched <strong>treatment</strong>'
        f' ({n_treatment} regions, {share:.0%} of baseline volume) and'
        f' <strong>control</strong> ({s.control_geo_count} regions) groups'
        ' whose historical conversion trends move together.</p><p>'
        + formatter.escape(validation)
        + f', so comparing the two groups over the {days}-day test window'
        ' isolates true incremental lift from seasonality and market-wide'
        ' shifts.</p>'
    )

  def _delivery_steps(
      self,
      s: report_data.CandidateSummary,
      design: api.Design,
      context: '_DesignContext',
      unit: str,
  ) -> list[str]:
    """Where and how spend changes, one step per cell."""
    del context  # Unused; kept for a uniform signature.
    m = s.control_geo_count
    changes = self._changes(design)
    multi = len(changes) > 1
    steps = []
    for change, _ in changes:
      cell = s.cells.get(change.cell)
      n = (
          cell.treatment_geo_count
          if cell is not None
          else sum(c.treatment_geo_count for c in s.cells.values())
      )
      name = self._short_label(change.label)
      prefix = f'{name}: ' if multi else ''
      group = f'{name} {unit}' if multi else f'treatment {unit}'
      amount = self._short(abs(change.amount))
      pct = (
          report_text.format_percent(abs(change.planned_change), 0)
          if change.planned_change is not None
          else ''
      )
      control = (
          '' if multi
          else f' while keeping normal spend in the {m} control {unit}'
      )
      if change.experiment_type == api.ExperimentType.GO_DARK:
        share = f', {pct} of BAU' if pct else ''
        steps.append(
            f'{prefix}Pause campaign spend in the {n} {group} (about'
            f' {amount}{share}){control}.'
        )
      elif change.experiment_type == api.ExperimentType.HEAVY_UP:
        by = f' by {pct}' if pct else ''
        steps.append(
            f'{prefix}Increase campaign spend in the {n} {group}{by} (about'
            f' +{amount}){control}.'
        )
      elif unit == 'regions':
        steps.append(
            f'{prefix}Deliver the campaign only in the {n} {group} (about'
            f' {amount} planned spend) and hold back the {m} control {unit}.'
        )
      else:
        steps.append(
            f'{prefix}Run the campaign only in the {n} {group} (at least'
            f' {amount} of spend there) and keep it off in the {m} control'
            f' {unit}.'
        )
    if multi:
      # One shared control group: mention it once.
      steps = [
          step.replace(f' and hold back the {m} control {unit}', '')
          for step in steps
      ]
      steps.append(
          f'Keep normal spend in the {m} control {unit} throughout the test.'
      )
    return steps

  def _schedule_step(self) -> str:
    """Test window and cooldown, for the launch plan."""
    schedule = self._schedule
    if schedule.dated:
      text = (
          f'Run the test from'
          f' {_date_span(schedule.test_start, schedule.test_end)}'
          f' ({schedule.test_days} days'
          + (
              f', the first {schedule.ramp_up_days} a ramp-up'
              if schedule.ramp_up_days
              else ''
          )
          + ')'
      )
      if schedule.cooldown_days:
        text += (
            f', then a {schedule.cooldown_days}-day cooldown to'
            f' {_date(schedule.cooldown_end)}'
        )
    else:
      text = f'Run the test for {schedule.test_days} days'
      if schedule.cooldown_days:
        text += f', then a {schedule.cooldown_days}-day cooldown'
    if not schedule.cooldown_days:
      return (
          text + ', with no cooldown. Hold the regional split steady without'
          ' running other geo-specific promotions.'
      )
    return text + (
        '. Hold the regional split steady without running other geo-specific'
        ' promotions, then return all regions to business as usual for the'
        ' cooldown.'
    )

  def _next_steps_html(
      self,
      s: report_data.CandidateSummary,
      design: api.Design,
      context: '_DesignContext',
  ) -> str:
    """Analyst next steps, with the full design ID to copy."""
    schedule = self._schedule
    save = (
        f'Save the design ID of {formatter.escape(s.name)}'
        f' <code class="geox-id-chip">{formatter.escape(s.design_id)}</code>: '
        + _copy_button('Copy design ID', s.design_id)
        + ' Its geo lists are under <a href="#details">Candidate details and'
        ' export</a>.'
    )
    if schedule.dated:
      keep = (
          'Keep the split from'
          f' {_date_span(schedule.test_start, schedule.test_end)}'
          f' ({schedule.test_days} days)'
      )
      if schedule.cooldown_days:
        keep += (
            f', then a {schedule.cooldown_days}-day cooldown to'
            f' {_date(schedule.cooldown_end)}'
        )
    else:
      keep = f'Keep the split for {schedule.test_days} days'
      if schedule.cooldown_days:
        keep += f', then a {schedule.cooldown_days}-day cooldown'
    keep += ', and avoid other geo-specific changes.'
    if schedule.cooldown_days:
      keep += (
          ' During the cooldown, return every geo to business as usual'
          ' (pretest levels).'
      )
    start = (
        f' and the test start date ({schedule.test_start:%Y-%m-%d})'
        if schedule.dated and schedule.test_start is not None
        else ''
    )
    analyze = (
        f'Then run geox.analyze(...) with this design{start}, and'
        ' geox.AnalysisSummarizer for the readout.'
    )
    items = [
        save,
        *(
            formatter.escape(step)
            for step in self._delivery_steps(s, design, context, unit='geos')
        ),
        formatter.escape(keep),
        formatter.escape(analyze),
    ]
    return (
        '<ol class="geox-steps">'
        + ''.join(f'<li>{item}</li>' for item in items)
        + '</ol>'
    )

  def _launch_card(self, context: '_DesignContext') -> formatter.Card:
    schedule = self._schedule
    terms = context.terms
    blocks: list[formatter.Block] = []
    for s in context.scoped:
      design = context.designs[s.design_id]
      table = report_data.geo_table(design, self._names)
      plan = report_data.plan_numbers(s, design, context.goals)
      steps = [
          *self._delivery_steps(s, design, context, unit='regions'),
          self._schedule_step(),
          'After the test window finishes, review the post-test analysis'
          ' readout to confirm incremental conversions and efficiency.',
      ]
      if plan is not None and plan.chance < 0.8:
        steps.append(
            'To raise the chance of a conclusive result, plan at least'
            f' {self._short(plan.spend_for_power)} of {terms.label.lower()},'
            ' run the test longer, or accept a larger MDE.'
        )
      buttons = [
          formatter.Button(
              'Download geo list (CSV)', 'download', _geo_csv(table),
              f'geox_{s.short_id}_geos.csv',
          ),
      ]
      if self._targeting_map is not None:
        buttons.insert(
            1,
            formatter.Button(
                'Download targeting list (CSV)',
                'download',
                _targeting_csv(table, self._targeting_map),
                f'geox_{s.short_id}_targeting.csv',
            ),
        )
      blocks.append(formatter.Row(
          [
              formatter.Html(self._how_it_works(s, design)),
              formatter.Html(
                  '<h3>Launch plan</h3>' + _ordered_list(steps)
              ),
          ],
          candidate=s.design_id,
      ))
      blocks.append(formatter.Buttons(buttons, candidate=s.design_id))
      blocks.append(formatter.Html(
          '<h3>Next steps</h3>' + self._next_steps_html(s, design, context),
          candidate=s.design_id,
          audience='technical',
      ))
    first = context.designs[context.summaries[0].design_id]
    pretest = report_data.pretest_range(first)
    if schedule.dated and schedule.test_start is not None:
      takeaway = (
          f'Test: {report_text.short_date(schedule.test_start)} –'
          f' {report_text.short_date(schedule.test_end)}'
          f' ({schedule.test_days} days)'
      )
      if schedule.cooldown_days and schedule.test_end is not None:
        takeaway += (
            '; cooldown:'
            f' {report_text.short_date(schedule.test_end + pd.Timedelta(days=1))}'
            f' – {report_text.short_date(schedule.cooldown_end)}'
            f' ({schedule.cooldown_days} days)'
        )
      if pretest is not None:
        takeaway += f'. Pretest data: {report_text.date_range(*pretest)}.'
    else:
      takeaway = None
    calendar = _chart(
        design_charts.calendar_chart(schedule),
        charts.Framing(
            'When does the test run?',
            reading_guide=(
                'Blue days are test days; yellow days are cooldown. Light'
                ' yellow days at the start are ramp-up.'
                if schedule.ramp_up_days
                else 'Blue days are test days; yellow days are cooldown.'
            ),
            takeaway=takeaway,
        ),
    )
    if calendar is not None:
      blocks.append(calendar)
    else:
      blocks.append(
          formatter.Text(
              f'Phases: {schedule.test_days}-day test'
              + (
                  f', then a {schedule.cooldown_days}-day cooldown.'
                  if schedule.cooldown_days
                  else '.'
              )
              + ' Pass test_start to see calendar dates.'
          )
      )
    return formatter.Card(
        'launch',
        'Launch plan',
        blocks,
        subtitle='Next steps for the selected option.',
    )

  # ------------------------------------------------------------------ details

  def _roster_html(self, s: report_data.CandidateSummary,
                   table: pd.DataFrame) -> str:
    """Group-by-group geo roster of one candidate."""
    if s.rank == 1:
      showing = (
          f'Showing {s.name}. Pick another option in the comparison above to'
          ' see its geos.'
      )
    else:
      showing = (
          f'Showing {s.name}, selected in the comparison above. The'
          ' recommended design is Option 1.'
      )
    rows = []
    for group, frame in table.groupby('group', sort=False):
      names = sorted(str(n) for n in frame['geo_name'])
      share = float(frame['share'].sum())
      rows.append(
          f'<tr><td>{formatter.escape(self._group_title(str(group), s))}</td>'
          f'<td>{len(frame)}</td><td>{share:.1%}</td>'
          f'<td>{formatter.escape(", ".join(names))}</td></tr>'
      )
    return (
        f'<p class="geox-detail">{formatter.escape(showing)}</p>'
        f'<h3>Geo roster of {formatter.escape(s.name)}</h3>'
        '<div class="geox-table-wrap"><table class="geox-table"><thead><tr>'
        '<th scope="col">Group</th><th scope="col">Count</th>'
        '<th scope="col">Share of pretest conversions</th>'
        '<th scope="col">Geos (A–Z)</th></tr></thead><tbody>'
        + ''.join(rows)
        + '</tbody></table></div>'
    )

  def _group_title(
      self, group: str, s: report_data.CandidateSummary
  ) -> str:
    """Group label with what happens to its ads, e.g. 'Control · …'."""
    experiment_type = (
        next(iter(s.cells.values())).experiment_type
        if len(s.cells) == 1
        else None
    )
    treat_desc, control_desc = report_text.GROUP_DESCRIPTIONS.get(
        experiment_type, ('spend changed', 'business as usual')
    ) if experiment_type is not None else ('spend changed',
                                           'business as usual')
    if group == charts.CONTROL_LABEL:
      return f'Control · {control_desc}'
    if group == charts.EXCLUDED_LABEL:
      return 'Excluded · not in the test'
    if len(s.cells) == 1:
      return f'Treatment · {treat_desc}'
    return group

  def _candidates_table_html(self, context: '_DesignContext') -> str:
    header = ''.join(
        f'<th scope="col">{formatter.escape(c)}</th>'
        for c in ('Option', 'Design ID', 'Chance', 'MDE', context.terms.label,
                  'Treatment geos', 'Goals', 'Export')
    )
    rows = []
    for s in context.scoped:
      design = context.designs[s.design_id]
      plan = report_data.plan_numbers(s, design, context.goals)
      chance = (
          f'{report_data.chance_label(plan.chance)} · {plan.chance:.0%}'
          if plan is not None
          else '—'
      )
      missed = [
          name for name, value in (
              ('MDE', s.meets_mde), ('budget', s.meets_budget),
              ('iCPA', s.meets_cpic))
          if value is False
      ]
      goals_text = 'Meets goals' if not missed else (
          'Misses ' + ', '.join(missed)
      )
      table = report_data.geo_table(design, self._names)
      cells = [
          s.name, s.design_id, chance, report_text.format_percent(s.max_mde),
          self._short(s.total_spend),
          str(sum(c.treatment_geo_count for c in s.cells.values())),
          goals_text,
      ]
      export = (
          _copy_button('Copy ID', s.design_id)
          + ' '
          + _download_link(
              'Geos CSV', _geo_csv(table), f'geox_{s.short_id}_geos.csv'
          )
      )
      if self._targeting_map is not None:
        export += ' ' + _download_link(
            'Targeting CSV',
            _targeting_csv(table, self._targeting_map),
            f'geox_{s.short_id}_targeting.csv',
        )
      rows.append(
          '<tr>'
          + ''.join(f'<td>{formatter.escape(c)}</td>' for c in cells)
          + f'<td>{export}</td></tr>'
      )
    targeting_note = (
        ' Targeting CSV: your targeting map rows for each geo, with its group,'
        ' ready for ad-platform setup.'
        if self._targeting_map is not None
        else ''
    )
    return (
        '<h3>All candidate designs (ranked)</h3><div class="geox-table-wrap">'
        f'<table class="geox-table"><thead><tr>{header}</tr></thead><tbody>'
        + ''.join(rows)
        + '</tbody></table></div><p class="geox-detail">Chance: probability'
        ' of a conclusive result at your goals. Copy ID: the full design ID,'
        ' to pick this design in your notebook. Geos CSV: one row per geo with'
        ' its group and stratum.'
        + formatter.escape(targeting_note)
        + '</p>'
    )

  def _details_card(self, context: '_DesignContext') -> formatter.Card:
    blocks: list[formatter.Block] = []
    has_names = len(self._names) > 0
    for s in context.scoped:
      design = context.designs[s.design_id]
      table = report_data.geo_table(design, self._names)
      blocks.append(formatter.Html(
          self._roster_html(s, table), candidate=s.design_id
      ))
      scale = charts.group_scale(
          list(report_data.group_labels(design).values())
      )
      strata = _chart(
          design_charts.strata_chart(table, scale),
          charts.Framing(
              'How were geos split within each stratum?',
              reading_guide=(
                  'Stratified sampling groups similar geos into strata and'
                  ' splits each stratum between groups, so treatment and'
                  ' control look alike.'
              ),
              takeaway=(
                  f'{s.name}: every stratum contributes to both treatment and'
                  ' control.'
              ),
          ),
          candidate=s.design_id,
      )
      if strata is not None:
        blocks.append(strata)
      roster = table[table['cell'] != 'excluded']
      columns = (['Geo'] if has_names else []) + [
          'Location', 'Group', 'Stratum', 'Pretest share', 'Spend share'
      ]
      rows = []
      for r in roster.to_dict('records'):
        stratum = r['stratum']
        row = [
            str(r['location']),
            str(r['group']),
            '' if stratum is None or pd.isna(stratum) else str(int(stratum)),
            report_text.format_percent(float(r['share']), 2),
            report_text.format_percent(float(r['spend_share']), 2),
        ]
        if has_names:
          row.insert(0, str(r['geo_name']))
        rows.append(row)
      blocks.append(formatter.Table(
          columns=columns,
          rows=rows,
          caption=f'All geos · {s.name}',
          candidate=s.design_id,
      ))
    blocks.append(formatter.Html(self._candidates_table_html(context)))
    return formatter.Card(
        'details',
        'Candidate details and export',
        blocks,
        audience='technical',
        cue=SELECT_CUE,
    )

  # ------------------------------------------------------------------ quality

  def _quality_card(self, design: api.Design) -> formatter.Card:
    quality = design.quality_check_result
    flagged_geos = sorted(
        str(g) for g in (quality.outlier_geos if quality else ())
    )
    flagged_dates = sorted(
        pd.Timestamp(d) for d in (quality.outlier_dates if quality else ())
    )
    user_geos = sorted(
        str(g)
        for g in (design.constraints.excluded_geos if design.constraints
                  else ())
    )
    excluded_dates = sorted(
        pd.Timestamp(d) for d in (design.excluded_dates or ())
    )

    def names(geos: Sequence[str]) -> str:
      return ', '.join(self._names(g) for g in geos) if geos else 'none'

    def dates(values: Sequence[pd.Timestamp]) -> str:
      return ', '.join(f'{d:%Y-%m-%d}' for d in values) if values else 'none'

    blocks: list[Optional[formatter.Block]] = [
        formatter.Text(
            f'Geos flagged by data quality checks: {names(flagged_geos)}.'
            f' Outlier dates flagged by data quality checks:'
            f' {dates(flagged_dates)}. User-excluded geos: {names(user_geos)}.'
            f' Dates excluded from the design: {dates(excluded_dates)}.'
        ),
        _chart(
            design_charts.daily_spend_chart(design),
            charts.Framing('Daily spend in the pretest data'),
        ),
        _chart(
            design_charts.data_quality_chart(design),
            charts.Framing(
                'Were there unusual days in the pretest?',
                reading_guide='Marked dates were excluded from the design.',
                takeaway=(
                    f'{report_text.plural(len(excluded_dates), "pretest day")}'
                    f' flagged or excluded ({dates(excluded_dates)}); leaving'
                    ' them out keeps the model from learning one-off spikes.'
                    if excluded_dates
                    else 'No pretest days were flagged or excluded.'
                ),
            ),
        ),
    ]
    excluded_all = sorted({*flagged_geos, *user_geos})
    weekly = report_data.weekly_quality(design.data, excluded_all)
    if not weekly.empty:
      gap_geos = sorted(set(weekly.loc[weekly['zero'], 'location']))
      already = [g for g in gap_geos if g in excluded_all]
      gap_text = (
          f'{report_text.plural(len(gap_geos), "geo")} have weeks without any'
          ' conversion ('
          + report_text.name_list([self._names(g) for g in gap_geos], 4)
          + '), often a tracking gap'
          + (f'; {len(already)} already excluded' if already else '')
          + ', check the others before the test.'
          if gap_geos
          else 'No geo has a week without conversions.'
      )
      blocks.append(_chart(
          design_charts.tracking_gaps_chart(weekly, self._names),
          charts.Framing(
              'Are there tracking gaps?',
              reading_guide=(
                  'Weekly conversions of the largest geos plus any geo with an'
                  ' empty week. Red cells are weeks without conversions.'
              ),
              takeaway=gap_text,
          ),
      ))
      strong = weekly[weekly['z'].abs() > 3]
      if strong.empty:
        z_text = 'No geo-week deviates strongly (|z| > 3).'
      else:
        top = strong['location'].value_counts().index.tolist()
        z_text = (
            f'{report_text.plural(len(strong), "geo-week")} deviate strongly'
            ' (|z| > 3), most often in '
            + report_text.name_list([self._names(g) for g in top], 3)
            + '.'
        )
      blocks.append(_chart(
          design_charts.weekly_z_chart(weekly, self._names),
          charts.Framing(
              'Which geo-weeks behave unusually?',
              reading_guide=(
                  'Robust z-scores (median and MAD per geo) of weekly'
                  ' conversions. Dark cells are unusual weeks.'
              ),
              takeaway=z_text,
          ),
      ))
    volumes = report_data.pretest_volumes(design.data)
    no_conversion = sorted(
        str(g) for g, v in volumes.items()
        if v <= 0 and str(g) not in excluded_all
    )
    if no_conversion:
      geo_set = ', '.join(repr(g) for g in no_conversion)
      snippet = f'geox.Constraints(excluded_geos={{{geo_set}}})'
      blocks.append(formatter.Html(
          '<p>To exclude the suggested geos, keep your other settings and'
          ' use:</p><code class="geox-snippet">'
          + formatter.escape(snippet)
          + '</code>'
          + _copy_button('Copy geox.Constraints()', snippet)
      ))
    rows = (
        [['Geo', f'{self._names(g)} ({g})',
          'Flagged by the data quality checks; the geo cannot inform the'
          ' counterfactual.', 'Excluded automatically'] for g in flagged_geos]
        + [['Date', f'{d:%Y-%m-%d}',
            'Total conversions far from the pretest trend.',
            'Excluded automatically'] for d in flagged_dates]
        + [['Geo', f'{self._names(g)} ({g})',
            'No conversions at all in the pretest.', 'Suggested']
           for g in no_conversion]
    )
    if rows:
      blocks.append(formatter.Table(
          columns=['Type', 'Geo or date', 'Finding', 'Status'],
          rows=rows,
          caption='Flagged geos and dates',
      ))
    blocks.append(formatter.Html(
        '<p class="geox-detail">The data quality checks run before the design'
        ' search and, by default, exclude what they flag. '
        + _doc_link('Data quality checks', report_text.DOCS_DATA_QUALITY)
        + '</p>'
    ))
    return formatter.Card(
        'quality',
        'Pretest data quality',
        [b for b in blocks if b is not None],
        audience='technical',
    )


@dataclasses.dataclass(frozen=True)
class _DesignContext:
  designs: dict[str, api.Design]
  summaries: list[report_data.CandidateSummary]
  scoped: list[report_data.CandidateSummary]
  goals: DesignGoals
  terms: report_text.SpendTerms
  coverage: maps.BoundaryCoverage
  n_feasible: int


@dataclasses.dataclass(frozen=True)
class _CellReadout:
  cell: str
  label: str
  metrics: api.AnalysisMetrics
  verdict: report_text.Verdict
  efficiency: Optional[report_data.Efficiency]
  spend: Optional[float]
  split: Optional[report_data.EffectSplit]
  check: report_data.ExecutionCheck


class AnalysisSummarizer:
  """Builds the interactive analysis readout for an `AnalysisResult`."""

  def __init__(
      self,
      analysis_result: api.AnalysisResult,
      geo_metadata: Optional[pd.DataFrame] = None,
      map_config: maps.MapConfig = maps.MapConfig(),
      target_cpic: Optional[float] = None,
      data: Optional[pd.DataFrame] = None,
      response_is_value: bool = False,
      currency_symbol: str = '',
  ):
    """Initializes the summarizer.

    Args:
      analysis_result: Output of `geox.analyze(...)`.
      geo_metadata: Optional display-name table.
      map_config: Boundary polygon configuration for the regions map.
      target_cpic: Target iCPA compared against the estimate; defaults to the
        design's `cost_per_incremental_conversion`.
      data: Analysis input data. With spend columns, enables spend execution
        checks, spend pacing and geo pretest volumes.
      response_is_value: Treat conversions as monetary value and lead with
        iROAS instead of iCPA.
      currency_symbol: Symbol prefixed to monetary values.

    Raises:
      ValueError: On invalid inputs.
    """
    if not analysis_result.results:
      raise ValueError('analysis_result must contain non-empty results.')
    if target_cpic is not None and target_cpic <= 0:
      raise ValueError(f'target_cpic must be positive, got {target_cpic}.')
    self._result = analysis_result
    self._names = report_data.resolve_geo_metadata(geo_metadata)
    self._map_config = map_config
    self._data = data
    self._value = response_is_value
    self._currency = currency_symbol
    design = analysis_result.analysis_config.design
    if target_cpic is None:
      target_cpic = report_data.default_goals(
          design, DesignGoals()
      ).target_cpic
    self._target_cpic = target_cpic

  def generate_html(self) -> str:
    """Returns the self-contained HTML report."""
    return formatter.render_report(self._build_report())

  def to_iframe(self, height: int = 1100) -> str:
    """Returns the report wrapped in an iframe for notebook display.

    Use `display(HTML(summarizer.to_iframe()))` in Colab, Cider or Jupyter so
    charts and the Executive/Full toggle work.

    Args:
      height: Frame height in pixels.
    """
    return formatter.iframe(self.generate_html(), height)

  def _repr_html_(self) -> str:
    return self.to_iframe()

  def output_analysis_summary(self, filename: str, filepath: str) -> str:
    """Writes the report to `<filepath>/<filename>` and returns the path."""
    return formatter.write_html(self.generate_html(), filename, filepath)

  # ------------------------------------------------------------------------

  def _kpi(self) -> str:
    return 'conversion value' if self._value else 'conversions'

  def _format_efficiency(self, value: float) -> str:
    if self._value:
      return report_text.format_number(value, 2)
    return report_text.format_money(value, self._currency, 2)

  def _format_lift(self, value: float, signed: bool = True) -> str:
    if self._value:
      text = report_text.format_money(value, self._currency)
      return ('+' + text) if signed and value > 0 else text
    if signed:
      return report_text.format_signed(value)
    return report_text.format_number(value)

  def _readouts(self) -> list[_CellReadout]:
    result = self._result
    design = result.analysis_config.design
    window = report_data.analysis_window(result)
    alpha = 1 - report_data.confidence_level(result)
    labels = report_data.group_labels(design)
    checks = {
        c.cell: c
        for c in report_data.execution_checks(result, self._data, self._names)
    }
    readouts = []
    multi = len(result.results) > 1
    for cell in sorted(result.results, key=report_text.cell_index):
      metrics = result.results[cell]
      # Multi-cell readouts name cells 'Treatment 1', 'Treatment 2', ...
      label = labels.get(cell, cell)
      if multi:
        label = label.split(' · ')[0]
      check = checks.get(cell) or report_data.ExecutionCheck(
          cell, labels.get(cell, cell), api.ExperimentType.HOLDBACK, False
      )
      readouts.append(
          _CellReadout(
              cell=cell,
              label=label,
              metrics=metrics,
              verdict=report_data.verdict(metrics, window, alpha, check.ok),
              efficiency=report_data.efficiency(metrics, self._value),
              spend=report_data.implied_spend(metrics),
              split=report_data.effect_split(metrics, window),
              check=check,
          )
      )
    return readouts

  @staticmethod
  def _overall(readouts: Sequence[_CellReadout]) -> report_text.Verdict:
    verdicts = [r.verdict for r in readouts]
    for v in (report_text.Verdict.NEEDS_REVIEW, report_text.Verdict.TOO_EARLY):
      if v in verdicts:
        return v
    if all(v == report_text.Verdict.YES for v in verdicts):
      return report_text.Verdict.YES
    return report_text.Verdict.NO

  def _build_report(self) -> formatter.Report:
    result = self._result
    design = result.analysis_config.design
    window = report_data.analysis_window(result)
    level = report_data.confidence_level(result)
    readouts = self._readouts()
    experiment_type = report_data.single_experiment_type(design)
    cards = [
        self._executive_card(readouts, window, level, experiment_type),
        self._details_card(readouts, level, window),
        self._counterfactual_card(readouts, window),
        self._cumulative_card(readouts, window),
        self._regions_card(design, window),
        self._data_card(window),
    ]
    chips = self._chips(design, window, level, experiment_type)
    return formatter.Report(
        title='GeoX experiment analysis',
        kind='analysis',
        cards=cards,
        chips=chips,
    )

  def _chips(
      self,
      design: api.Design,
      window: report_data.AnalysisWindow,
      level: float,
      experiment_type: Optional[api.ExperimentType],
  ) -> list[formatter.Chip]:
    """Header chips: status, region, test window, split, type, alpha."""
    n_treatment = sum(len(d.treatment_geos) for d in design.designs.values())
    n_control = len(design.control_geos)
    n_excluded = len(design.excluded_geos or ())
    test_days = min(window.observed_days, window.planned_days)
    test_end = min(window.end, window.planned_end)
    status = (
        'Test window complete'
        if window.complete
        else f'In progress (day {test_days} of {window.planned_days})'
    )
    chips = [formatter.Chip('Status', status)]
    if self._map_config.region_name:
      total = n_treatment + n_control + n_excluded
      chips.append(formatter.Chip(
          'Region', f'{self._map_config.region_name} ({total} geos)'
      ))
    chips.append(formatter.Chip(
        'Test',
        f'{report_text.date_range(window.start, test_end)} ({test_days} days)',
    ))
    if window.cooldown_days:
      chips.append(formatter.Chip(
          'Cooldown',
          report_text.date_range(
              window.planned_end + pd.Timedelta(days=1), window.end
          ),
      ))
    chips.append(formatter.Chip(
        'Regions split',
        f'{n_treatment} treatment, {n_control} control, {n_excluded} excluded',
    ))
    if experiment_type is None:
      types = ', '.join(
          f'Treatment {i}'
          f' {report_text.EXPERIMENT_TYPE_LABELS[r.check.experiment_type].lower()}'
          for i, r in enumerate(self._readouts(), start=1)
      )
      chips.append(formatter.Chip('Types', types))
    else:
      chips.append(formatter.Chip(
          'Type',
          report_text.EXPERIMENT_TYPE_LABELS[experiment_type]
          + ' · '
          + report_text.EXPERIMENT_TYPE_EXPLANATIONS[experiment_type],
      ))
    chips.append(formatter.Chip('Alpha', f'{1 - level:g} · {level:.0%} CIs'))
    return chips

  def _count(self, value: float) -> str:
    """Compact count (or value), e.g. '34.0k'."""
    return report_text.short_money(
        value, self._currency if self._value else ''
    )

  @staticmethod
  def _significant(r: '_CellReadout', level: float) -> bool:
    """A significant positive effect, using the verdict's rule."""
    lift = r.metrics.lift
    return (
        report_data.is_significant(lift, 1 - level)
        and lift.point_estimate > 0
    )

  @staticmethod
  def _significant_decrease(r: '_CellReadout', level: float) -> bool:
    lift = r.metrics.lift
    return (
        report_data.is_significant(lift, 1 - level)
        and lift.point_estimate < 0
    )

  @staticmethod
  def _change_phrase(r: '_CellReadout') -> str:
    if r.check.experiment_type == api.ExperimentType.HEAVY_UP:
      return 'without the extra spend'
    return 'without the campaign'

  def _lift_sentence(self, r: '_CellReadout') -> str:
    """Effect in extra conversions (geo tests avoid relative-lift terms)."""
    point = r.metrics.lift.point_estimate or 0.0
    extra = self._count(abs(point))
    if r.check.experiment_type == api.ExperimentType.GO_DARK:
      # TBR reports go-dark effects with the sign flipped: positive means
      # conversions fell while ads were paused.
      word = 'fewer' if point >= 0 else 'more'
      return (
          f'Treatment geos had about {extra} {word} {self._kpi()} while ads'
          ' were paused than expected had ads kept running.'
      )
    word = 'more' if point >= 0 else 'fewer'
    return (
        f'Treatment geos had about {extra} {word} {self._kpi()} than'
        f' expected {self._change_phrase(r)}.'
    )

  def _target_phrase(self, value: float, short: bool = False) -> str:
    """'within your $1.00 target' (or '' without a target)."""
    if self._target_cpic is None or self._value:
      return ''
    target = self._format_efficiency(self._target_cpic)
    if short:
      return f', target {target}'
    better = value <= self._target_cpic
    return (
        f'{"below" if better else "above"} your {target} target'
    )

  def _cost_sentence(self, r: '_CellReadout') -> str:
    eff = r.efficiency
    if eff is None:
      return ''
    value = self._format_efficiency(eff.point)
    if self._value:
      return f'Each unit of spend returned about {value} ({eff.label}).'
    return (
        f'Each extra conversion cost about {value}'
        f' ({eff.label}{self._target_phrase(eff.point, short=True)}).'
    )

  def _no_lift_sentence(self, r: '_CellReadout', level: float) -> str:
    upper = self._count(r.metrics.lift.upper_bound)
    return (
        f'More than about {upper} extra {self._kpi()} is unlikely (the top of'
        f' the {level:.0%} CI); smaller effects cannot be ruled out.'
    )

  def _banner(
      self,
      readouts: Sequence['_CellReadout'],
      overall: report_text.Verdict,
      window: report_data.AnalysisWindow,
      level: float,
  ) -> formatter.Callout:
    """Verdict banner: title, plain-language result and a docs link."""
    multi = len(readouts) > 1
    # Headline leads with incremental conversions and iCPA (UX feedback).

    def impact(r: '_CellReadout') -> str:
      text = f'{self._format_lift(r.metrics.lift.point_estimate)} extra'
      text += f' {self._kpi()}'
      if r.efficiency is not None and self._significant(r, level):
        text += (
            f' at {self._format_efficiency(r.efficiency.point)}'
            f' {r.efficiency.label}'
        )
      return text

    if multi:
      title = '; '.join(
          f'{r.label}: '
          + (impact(r) if self._significant(r, level) else 'no lift detected')
          for r in readouts
      )
    else:
      r = readouts[0]
      if self._significant(r, level):
        title = f'Yes: {impact(r)}'
      elif self._significant_decrease(r, level):
        title = f'Needs review: significant decrease ({impact(r)})'
      else:
        title = f'No lift detected (estimate {impact(r)}, not significant)'
    if overall == report_text.Verdict.TOO_EARLY:
      title = title.replace('Yes: ', '', 1)
      title = f'Interim: {title}'
    sentences = []
    for r in readouts:
      prefix = f'{r.label}: ' if multi else ''
      if self._significant(r, level):
        text = self._cost_sentence(r)
        if not multi:
          text = (text + ' ' + self._lift_sentence(r)).strip()
      else:
        text = (
            'The estimated effect is not statistically significant, so the'
            ' data are consistent with no effect. '
            + self._no_lift_sentence(r, level)
        )
      sentences.append(prefix + text)
    if overall == report_text.Verdict.TOO_EARLY:
      sentences.append(
          f'Results cover {min(window.observed_days, window.planned_days)} of'
          f' {window.planned_days} planned test days and may change.'
      )
    issues = [r.label for r in readouts if not r.check.ok]
    if issues:
      sentences.append(
          f'⚠ Spend did not change as planned in {", ".join(issues)}; see'
          ' "Was the test run as planned?" below before acting on this.'
      )
    return formatter.Callout(
        ' '.join(sentences),
        title=title,
        tone=report_text.RESULT_WORDING[overall].tone,
        links=[('Learn more', report_text.DOCS_INTRO_TO_ANALYSIS)],
    )

  def _meaning(
      self,
      r: '_CellReadout',
      window: report_data.AnalysisWindow,
      level: float,
  ) -> list[str]:
    """'What this means' paragraphs for one cell."""
    verdict = r.verdict
    if verdict == report_text.Verdict.TOO_EARLY:
      return [
          'So far the campaign shows a significant lift, but interim results'
          ' can change.'
          if self._significant(r, level)
          else 'So far there is no significant lift; interim results can'
          ' change.'
      ]
    if not self._significant(r, level):
      return [
          'The test found no statistically significant effect. '
          + self._no_lift_sentence(r, level)
      ]
    text = self._lift_sentence(r).rstrip('.')
    text += f' over {window.observed_days} days'
    if r.efficiency is not None and not self._value:
      text += f', at about {self._format_efficiency(r.efficiency.point)} each'
      target = self._target_phrase(r.efficiency.point)
      if target:
        text += f' ({target})'
    text += '.'
    paragraphs = [text]
    if r.split is not None and window.cooldown_days:
      cooldown = report_text.date_range(
          window.planned_end + pd.Timedelta(days=1), window.end
      )
      paragraphs.append(
          f'Includes ~{self._count(r.split.test_effect)} extra {self._kpi()}'
          f' during the {window.planned_days}-day test and'
          f' ~{self._count(r.split.cooldown_effect)} delayed {self._kpi()}'
          f' captured in the {r.split.cooldown_days}-day cooldown'
          f' ({cooldown}).'
      )
    if verdict == report_text.Verdict.NEEDS_REVIEW:
      paragraphs.append(report_text.RESULT_WORDING[verdict].meaning)
    return paragraphs

  def _next_step_text(
      self,
      r: '_CellReadout',
      window: report_data.AnalysisWindow,
      level: float,
  ) -> str:
    verdict = r.verdict
    if verdict == report_text.Verdict.TOO_EARLY:
      return (
          f'Wait for the planned end ({window.planned_end:%Y-%m-%d}) and'
          ' re-run analyze() with analysis_end_date on or after it.'
      )
    if verdict == report_text.Verdict.NEEDS_REVIEW:
      return (
          'Check how spend changed in the flagged geos before using this'
          ' result; consider re-running the analysis without them.'
      )
    if not self._significant(r, level):
      return (
          'To detect a smaller effect next time, plan a larger spend change,'
          ' a longer test or more treatment geos.'
      )
    return (
        'Use this result to plan budgets and to calibrate your marketing mix'
        ' model; share the summary with the intervals below.'
    )

  def _execution_line(self, r: '_CellReadout') -> tuple[Optional[bool], str]:
    """Execution check for one cell: (ok, sentence)."""
    check = r.check
    if not check.has_spend:
      return None, report_text.execution_wording(
          None, None, None, has_spend=False
      ).sentence
    change = check.observed_change
    if change is None:
      return None, 'Treatment geos had no pretest spend to compare against.'
    pct = report_text.format_percent(abs(change), 0)
    signed = report_text.format_percent(change, 0, signed=True)
    moved = f'rose {signed}' if change >= 0 else f'fell {pct}'
    if check.issue_geos:
      names = report_text.name_list(list(check.issue_geos), 5)
      return False, (
          f'Treatment spend {moved} overall, but'
          f' {report_text.plural(len(check.issue_geos), "treatment geo")} did'
          f' not follow the plan: {names}.'
      )
    if check.experiment_type == api.ExperimentType.GO_DARK:
      ok = (
          check.verified
          if check.verified is not None
          else change <= -(1 - report_data.GO_DARK_MAX_SPEND_SHARE)
      )
      return ok, (
          f'Treatment spend fell {pct}, as a go-dark requires.'
          if ok
          else f'Treatment spend only changed {signed}; a go-dark needs ads'
          ' paused.'
      )
    if check.planned_change is not None and check.verified is not None:
      planned = report_text.format_percent(check.planned_change, 0, signed=True)
      return check.verified, (
          f'Treatment spend {moved} relative to control (planned {planned}).'
      )
    if change <= 0:
      return None, (
          f'Treatment spend {moved} relative to control while the campaign'
          ' ran. This is an observation; no planned spend target is known.'
      )
    # No planned amount to compare against: report the observation only.
    return None, (
        f'Treatment spend {moved} relative to control while the campaign ran.'
        ' This is an observation; it has not been verified against a planned'
        ' spend target.'
    )

  def _execution_callout(
      self, readouts: Sequence['_CellReadout']
  ) -> formatter.Callout:
    """'Was the test run as planned?' with one line per cell."""
    multi = len(readouts) > 1
    results = [(r, *self._execution_line(r)) for r in readouts]
    oks = [ok for _, ok, _ in results]
    tone: formatter.Tone
    title = 'Was the test run as planned?'
    if False in oks:
      icon, tone = '⚠', 'warning'
    elif all(ok is True for ok in oks):
      icon, tone = '✓', 'success'
    else:
      icon, tone = 'ℹ', 'neutral'
      if not any(r.check.has_spend for r in readouts):
        title = 'Spend execution not checked'
      elif not any(oks):
        title = 'Observed spend change'
    icons = {True: '✓', False: '⚠', None: 'ℹ'}
    lines = [
        (f'{r.label}: ' if multi else '') + f'{icons[ok]} {text}'
        for r, ok, text in results
    ]
    return formatter.Callout(
        ' · '.join(lines),
        title=f'{icon} {title}',
        tone=tone,
    )

  def _totals_takeaway(
      self,
      r: '_CellReadout',
      window: report_data.AnalysisWindow,
      level: float,
  ) -> str:
    """'Treatment geos converted 917.5k against 883.6k expected …'."""
    observed, counterfactual = self._window_totals(r, window)
    lift = r.metrics.lift
    expected = (
        'expected had ads kept running'
        if r.check.experiment_type == api.ExperimentType.GO_DARK
        else 'expected without the change'
    )
    return (
        f'Treatment geos converted {self._count(observed)} against'
        f' {self._count(counterfactual)} {expected}:'
        f' {self._format_lift(lift.point_estimate)} incremental'
        f' {self._kpi()} ({level:.0%} CI'
        f' {self._format_lift(lift.lower_bound)} to'
        f' {self._format_lift(lift.upper_bound)})'
        + (
            '; the interval includes zero, so the effect is not'
            ' statistically significant.'
            if not self._significant(r, level)
            else '.'
        )
    )

  def _tiles(
      self,
      readout: _CellReadout,
      level: float,
      multi: bool,
      window: Optional[report_data.AnalysisWindow] = None,
  ) -> formatter.Tiles:
    """Verdict, extra conversions, lift, iCPA and implied spend tiles."""
    metrics = readout.metrics
    window = window or report_data.analysis_window(self._result)
    prefix = f'{readout.label}: ' if multi else ''
    significant = self._significant(readout, level)
    wording = report_text.RESULT_WORDING[readout.verdict]
    verdict_detail = {
        report_text.Verdict.YES: 'A statistically significant lift in'
        f' {self._kpi()}.',
        report_text.Verdict.NO: 'No statistically significant effect.',
        report_text.Verdict.TOO_EARLY: (
            f'Day {min(window.observed_days, window.planned_days)} of'
            f' {window.planned_days}: '
            + ('a significant lift so far.' if significant
               else 'no significant lift so far.')
        ),
        report_text.Verdict.NEEDS_REVIEW: 'Spend did not change as planned'
        ' (see below).',
    }[readout.verdict]
    tiles = [
        formatter.Tile(
            f'{readout.label}: extra {self._kpi()}?'
            if multi
            else f'Did it drive extra {self._kpi()}?',
            readout.verdict.value,
            detail=verdict_detail,
            tone=wording.tone,
        )
    ]
    lift = metrics.lift
    tiles.append(
        formatter.Tile(
            prefix + f'Extra {self._kpi()}',
            self._count(lift.point_estimate),
            ci=report_text.format_interval(
                self._count(lift.lower_bound),
                self._count(lift.upper_bound),
                level,
            ),
            detail=(
                f'over {window.observed_days} analysed days'
                if significant
                else 'not statistically significant'
            ),
        )
    )
    eff = readout.efficiency
    if eff is not None:
      label = (
          'Return per unit of spend' if self._value
          else 'Cost per extra conversion'
      )
      if significant:
        target = self._target_phrase(eff.point)
        better = (
            self._target_cpic is None
            or self._value
            or eff.point <= self._target_cpic
        )
        tiles.append(
            formatter.Tile(
                prefix + label,
                self._format_efficiency(eff.point),
                ci=report_text.format_interval(
                    self._format_efficiency(eff.lower),
                    self._format_efficiency(eff.upper),
                    level,
                ),
                detail=(
                    target.replace('below your', 'within your')
                    if target else eff.label
                ),
                tone='success' if (target and better) else (
                    'warning' if target else 'neutral'
                ),
            )
        )
      else:
        tiles.append(
            formatter.Tile(prefix + label, '—', detail='needs a significant lift')
        )
    if readout.spend is not None:
      terms = report_text.spend_terms(readout.check.experiment_type)
      tiles.append(
          formatter.Tile(
              prefix + f'{terms.label} (implied)',
              report_text.short_money(abs(readout.spend), self._currency),
              detail='implied by the estimates, no interval',
          )
      )
    return formatter.Tiles(tiles)


  def _executive_card(
      self,
      readouts: Sequence[_CellReadout],
      window: report_data.AnalysisWindow,
      level: float,
      experiment_type: Optional[api.ExperimentType],
  ) -> formatter.Card:
    result = self._result
    design = result.analysis_config.design
    overall = self._overall(readouts)
    multi = len(readouts) > 1
    n_treatment = sum(len(d.treatment_geos) for d in design.designs.values())
    blocks: list[Optional[formatter.Block]] = [
        formatter.Text(
            report_text.test_objective(
                experiment_type,
                n_treatment,
                len(design.control_geos),
                window.planned_days,
                self._value,
            )
        ),
        self._banner(readouts, overall, window, level),
    ]
    for r in readouts:
      blocks.append(self._tiles(r, level, multi, window))
    if not window.complete:
      blocks.append(formatter.Callout(
          f'The analysis window covers'
          f' {min(window.observed_days, window.planned_days)} of the'
          f' {window.planned_days} planned test days. Re-run analyze() with'
          f' analysis_end_date on or after {window.planned_end:%Y-%m-%d} for'
          ' the full result.',
          tone='info',
      ))

    def per_cell(texts: Sequence[tuple[_CellReadout, str]]) -> str:
      return ''.join(
          '<p>'
          + (f'<strong>{formatter.escape(r.label)}</strong>: ' if multi else '')
          + formatter.escape(text)
          + '</p>'
          for r, text in texts
      )

    meaning = [(r, p) for r in readouts for p in self._meaning(r, window, level)]
    steps = [(r, self._next_step_text(r, window, level)) for r in readouts]
    formula = (
        'iROAS = conversion value ÷ campaign spend.'
        if self._value
        else 'iCPA = campaign spend ÷ extra conversions.'
    )
    blocks.append(formatter.Row([
        formatter.Html('<h3>What this means</h3>' + per_cell(meaning)),
        formatter.Html('<h3>Next step</h3>' + per_cell(steps)),
    ]))
    blocks.append(formatter.Callout(formula, tone='info'))
    blocks.append(self._execution_callout(readouts))
    labels = {r.cell: r.label for r in readouts}
    totals: list[formatter.Block] = []
    for r in readouts:
      chart = _chart(
          analysis_charts.effect_totals_chart(result, labels, cell=r.cell),
          charts.Framing(
              f'How many extra {self._kpi()} did the campaign drive?'
              + (f' · {r.label}' if multi else ''),
              reading_guide=(
                  f'Left: total {self._kpi()} in treatment geos during the'
                  ' test, as expected without the change (counterfactual) and'
                  ' as observed. Right: the difference, i.e. incremental'
                  f' {self._kpi()}, with its confidence interval. An interval'
                  ' that crosses zero means the effect is not statistically'
                  ' significant.'
              ),
              takeaway=self._totals_takeaway(r, window, level),
          ),
      )
      if chart is not None:
        totals.append(chart)
    if totals:
      blocks.append(formatter.Row(totals) if multi else totals[0])
    blocks.extend(self._estimates_blocks(readouts, level, window))
    return formatter.Card(
        'executive',
        'Executive summary',
        [b for b in blocks if b is not None],
    )


  def _estimates_blocks(
      self,
      readouts: Sequence[_CellReadout],
      level: float,
      window: report_data.AnalysisWindow,
  ) -> list[formatter.Block]:
    """Analyst estimates table (one row per metric) with a CSV download."""
    multi = len(readouts) > 1
    rows: list[list[str]] = []

    def interval(point: str, lower: str, upper: str) -> str:
      return f'{point} [{lower}, {upper}]'

    for r in readouts:
      metrics = r.metrics
      p_value = f'{metrics.lift.p_value:.3f}'
      significant = self._significant(r, level)
      eff = r.efficiency
      cell_rows: list[tuple[str, str, str]] = []
      if eff is not None:
        name = (
            'iROAS (conversion value per unit of spend)' if self._value
            else 'iCPA (cost per incremental conversion)'
        )
        cell_rows.append((
            name,
            interval(
                self._format_efficiency(eff.point),
                self._format_efficiency(eff.lower),
                self._format_efficiency(eff.upper),
            ) if significant else '— not estimable',
            p_value,
        ))
      lift = metrics.lift
      cell_rows.append((
          f'Incremental {self._kpi()}',
          interval(
              report_text.format_number(lift.point_estimate),
              report_text.format_number(lift.lower_bound),
              report_text.format_number(lift.upper_bound),
          ),
          p_value,
      ))
      pct = metrics.percent_lift
      cell_rows.append((
          'Lift (% vs. counterfactual)',
          interval(
              report_text.format_percent(pct.point_estimate, 2),
              report_text.format_percent(pct.lower_bound, 2),
              report_text.format_percent(pct.upper_bound, 2),
          ),
          f'{pct.p_value:.3f}',
      ))
      if metrics.icpd is not None:
        icpd = metrics.icpd
        cell_rows.append((
            'iCPD (extra conversions per unit of spend; iCPA = 1 / iCPD)',
            interval(
                report_text.format_number(icpd.point_estimate, 2),
                report_text.format_number(icpd.lower_bound, 2),
                report_text.format_number(icpd.upper_bound, 2),
            ),
            f'{icpd.p_value:.3f}',
        ))
      if r.spend is not None:
        terms = report_text.spend_terms(r.check.experiment_type)
        cell_rows.append((
            terms.label,
            report_text.short_money(abs(r.spend), self._currency),
            '—',
        ))
      observed, counterfactual = self._window_totals(r, window)
      cell_rows.append((
          f'Observed {self._kpi()} (treatment geos)',
          report_text.format_number(observed), '—',
      ))
      cell_rows.append((
          f'Counterfactual {self._kpi()}',
          report_text.format_number(counterfactual), '—',
      ))
      descriptive = getattr(metrics, 'descriptive_metrics', None)
      bau = getattr(descriptive, 'estimated_bau_spend', None)
      if bau is not None:
        cell_rows.append((
            'Estimated BAU spend',
            report_text.short_money(bau, self._currency), '—',
        ))
      for name, estimate, p in cell_rows:
        rows.append(([r.label] if multi else []) + [name, estimate, p])
    columns = (['Cell'] if multi else []) + ['Metric', 'Estimate', 'p-value']
    csv_text = formatter.to_csv(
        [dict(zip(columns, row)) for row in rows], columns
    )
    return [
        formatter.Table(
            columns=columns,
            rows=rows,
            caption=f'Estimates ({level:.0%} CI in brackets)',
            audience='technical',
        ),
        formatter.Buttons(
            [formatter.Button(
                'Download CSV', 'download', csv_text, 'geox_estimates.csv'
            )],
            audience='technical',
        ),
    ]

  def _window_totals(
      self, readout: _CellReadout, window: report_data.AnalysisWindow
  ) -> tuple[float, float]:
    frame = readout.metrics.counterfactual_conversions
    if frame is None or frame.empty:
      return float('nan'), float('nan')
    dates = pd.to_datetime(frame.index)
    in_window = frame[(dates >= window.start) & (dates <= window.end)]
    return (
        float(in_window['observed'].sum()),
        float(in_window['counterfactual'].sum()),
    )

  def _equation_text(
      self, readout: _CellReadout, window: report_data.AnalysisWindow
  ) -> str:
    observed, counterfactual = self._window_totals(readout, window)
    go_dark = readout.check.experiment_type == api.ExperimentType.GO_DARK
    effect = counterfactual - observed if go_dark else observed - counterfactual
    if go_dark:
      expr = (
          f'Counterfactual {report_text.format_number(counterfactual)} −'
          f' observed {report_text.format_number(observed)}'
      )
    else:
      expr = (
          f'Observed {report_text.format_number(observed)} − counterfactual'
          f' {report_text.format_number(counterfactual)}'
      )
    return (
        f'{readout.label}: {expr} = {self._format_lift(effect)} extra'
        f' {self._kpi()}.'
    )

  def _details_card(
      self,
      readouts: Sequence[_CellReadout],
      level: float,
      window: Optional[report_data.AnalysisWindow] = None,
  ) -> formatter.Card:
    blocks: list[formatter.Block] = []
    window = window or report_data.analysis_window(self._result)
    blocks.append(formatter.Text(
        'How the observed and counterfactual totals give the effect, and how'
        f' certain each estimate is ({level:.0%} CIs in brackets and in the'
        ' charts). The headline numbers, their intervals and the effect over'
        ' time are in the Executive summary.'
    ))
    for r in readouts:
      metrics = r.metrics
      blocks.append(formatter.Html(
          '<p class="geox-equation">'
          + formatter.escape(self._equation_strip(r, window, level))
          + '</p>'
      ))
      row: list[Optional[formatter.Block]] = []
      if r.efficiency is not None and metrics.icpd is not None:
        transform: Optional[Callable[[float], float]] = (
            None if self._value else _reciprocal
        )
        row.append(
            _chart(
                analysis_charts.confidence_curve(
                    metrics.icpd,
                    r.efficiency.label,
                    ',.2f',
                    transform=transform,
                    level=level,
                ),
                charts.Framing(
                    f'{r.label} · {r.efficiency.label} confidence',
                    reading_guide=(
                        'How confident we are that the true value is above each'
                        ' point on the x-axis, based on the reported placebo'
                        ' confidence interval.'
                    ),
                    takeaway=(
                        f'{r.efficiency.label}'
                        f' {self._format_efficiency(r.efficiency.point)}, with a'
                        f' {level:.0%} interval of'
                        f' {self._format_efficiency(r.efficiency.lower)} to'
                        f' {self._format_efficiency(r.efficiency.upper)}.'
                    ),
                ),
            )
        )
      blocks.append(formatter.Row([b for b in row if b is not None]))
    return formatter.Card(
        'detailed',
        'Detailed results',
        blocks,
        audience='technical',
    )


  def _equation_strip(
      self,
      r: _CellReadout,
      window: report_data.AnalysisWindow,
      level: float,
  ) -> str:
    """'917.5k Observed − 883.6k Counterfactual = 34.0k Incremental · CI'."""
    observed, counterfactual = self._window_totals(r, window)
    lift = r.metrics.lift
    if r.check.experiment_type == api.ExperimentType.GO_DARK:
      expr = (
          f'{self._count(counterfactual)} Counterfactual (expected had ads kept'
          f' running) − {self._count(observed)} Observed in treatment geos'
      )
    else:
      expr = (
          f'{self._count(observed)} Observed in treatment geos −'
          f' {self._count(counterfactual)} Counterfactual (expected)'
      )
    prefix = f'{r.label}: ' if len(self._result.results) > 1 else ''
    return (
        f'{prefix}{expr} = {self._count(lift.point_estimate)} Incremental ·'
        f' {level:.0%} CI [{self._count(lift.lower_bound)},'
        f' {self._count(lift.upper_bound)}]'
    )

  def _counterfactual_card(
      self,
      readouts: Sequence[_CellReadout],
      window: report_data.AnalysisWindow,
  ) -> formatter.Card:
    intro = (
        'Before the test start marker the counterfactual should track observed'
        f' {self._kpi()} closely; after it, the gap is the estimated effect.'
    )
    if window.cooldown_days:
      intro += (
          ' Shaded days after the planned test end (cooldown) capture delayed'
          f' {self._kpi()} and are included in the estimates.'
      )
    if self._data is not None:
      intro += (
          ' The spend pacing charts check that spend changed as planned in'
          ' treatment geos and stayed at business as usual in control geos.'
      )
    blocks: list[Optional[formatter.Block]] = [
        formatter.Callout(intro, tone='neutral')
    ]
    for r in readouts:
      blocks.append(
          _chart(
              analysis_charts.observed_counterfactual_chart(
                  r.metrics, window, r.label
              ),
              charts.Framing(
                  f'{r.label} · observed vs. counterfactual',
                  reading_guide=(
                      'The counterfactual is what treatment geos would have'
                      ' done without the change, predicted from control geos.'
                      ' Shading marks the test and cooldown periods.'
                  ),
                  takeaway=self._equation_text(r, window),
              ),
          )
      )
      row: list[Optional[formatter.Block]] = [
          _chart(
              analysis_charts.pointwise_chart(r.metrics, window),
              charts.Framing(
                  'Daily gap',
                  reading_guide='Observed minus counterfactual, per day.',
              ),
          )
      ]
      if self._data is not None:
        spend = report_data.daily_spend_by_group(
            self._result, self._data, r.cell
        )
        row.append(
            _chart(
                analysis_charts.spend_pacing_chart(
                    spend, window, [r.label]
                ),
                charts.Framing(
                    'Did spend change as planned?',
                    reading_guide=(
                        'Daily spend in treatment vs. control geos.'
                    ),
                    takeaway=report_text.execution_wording(
                        r.check.observed_change,
                        r.check.planned_change,
                        r.check.verified,
                        r.check.issue_geos,
                        has_spend=r.check.has_spend,
                    ).sentence,
                ),
            )
        )
      blocks.append(formatter.Row([b for b in row if b is not None]))
    return formatter.Card(
        'counterfactual',
        'Observed vs. counterfactual',
        [b for b in blocks if b is not None],
        audience='technical',
    )

  def _cumulative_card(
      self,
      readouts: Sequence[_CellReadout],
      window: report_data.AnalysisWindow,
  ) -> formatter.Card:
    blocks: list[formatter.Block] = []
    for r in readouts:
      row = [
          _chart(
              analysis_charts.cumulative_chart(r.metrics, window),
              charts.Framing(
                  f'{r.label} · cumulative extra {self._kpi()}',
                  reading_guide='The band is the confidence interval.',
                  takeaway=(
                      f'By {window.end:%b} {window.end.day}, cumulative extra'
                      f' {self._kpi()} reached'
                      f' {self._format_lift(r.metrics.lift.point_estimate)}'
                      f' ({report_text.format_interval(self._format_lift(r.metrics.lift.lower_bound), self._format_lift(r.metrics.lift.upper_bound), report_data.confidence_level(self._result))}).'
                  ),
              ),
          ),
          _chart(
              analysis_charts.cumulative_icpd_chart(r.metrics, window),
              charts.Framing(
                  f'{r.label} · cumulative iCPD',
                  reading_guide=(
                      'iCPD is extra conversions per unit of spend (the'
                      ' statistical estimand behind iCPA).'
                  ),
              ),
          ),
      ]
      blocks.append(formatter.Row([b for b in row if b is not None]))
    return formatter.Card(
        'cumulative',
        'Cumulative effect',
        blocks,
        audience='technical',
    )

  def _regions_card(
      self, design: api.Design, window: report_data.AnalysisWindow
  ) -> formatter.Card:
    volumes = None
    if design.data is None and self._data is not None:
      volumes = report_data.pretest_volumes(self._data, before=window.start)
    geos = sorted(
        {str(g) for g in design.control_geos}
        | {str(g) for d in design.designs.values() for g in d.treatment_geos}
    )
    coverage = maps.boundary_coverage(geos, self._map_config)
    blocks: list[Optional[formatter.Block]] = [
        _map_block(
            design,
            self._names,
            self._map_config,
            charts.Framing(
                'Where the test ran',
                reading_guide=(
                    'Hover a region for its group and pretest share.'
                ),
            ),
            volumes=volumes,
        ),
        formatter.Html(_map_legend(
            design, report_data.geo_table(design, self._names, volumes)
        )),
        _coverage_callout(coverage, self._names),
    ]
    blocks = [
        formatter.Callout(
            'TBR estimates one pooled effect across treatment geos, so there'
            ' is no per-geo lift. Hover a region for its group and pretest'
            ' volume.',
            tone='neutral',
        ),
        *self._data_notes(design),
        *blocks,
    ]
    return formatter.Card(
        'regions',
        'Regions split',
        [b for b in blocks if b is not None],
    )

  def _data_notes(self, design: api.Design) -> list[formatter.Block]:
    """Outlier dates and excluded geos and dates, as short callouts."""
    notes: list[formatter.Block] = []
    quality = self._result.quality_check_result
    outlier_dates = sorted(quality.outlier_dates) if quality else []
    if outlier_dates:
      notes.append(formatter.Callout(
          'Outlier dates flagged by analysis data quality checks: '
          + ', '.join(f'{d:%Y-%m-%d}' for d in outlier_dates),
          tone='warning',
      ))
    excluded = sorted(str(g) for g in (design.excluded_geos or ()))
    if excluded:
      notes.append(formatter.Callout(
          'Excluded geos: ' + ', '.join(self._names(g) for g in excluded),
          tone='info',
      ))
    dates = sorted(design.excluded_dates or ())
    if dates:
      notes.append(formatter.Callout(
          'Excluded dates: ' + ', '.join(f'{d:%Y-%m-%d}' for d in dates),
          tone='info',
      ))
    return notes

  def _data_card(self, window: report_data.AnalysisWindow) -> formatter.Card:
    blocks: list[Optional[formatter.Block]] = []
    if self._data is not None:
      z_scores = report_data.robust_z_scores(self._data)
      blocks.append(
          _chart(
              analysis_charts.anomaly_heatmap(z_scores, self._names, window),
              charts.Framing(
                  'Unusual days by geo',
                  reading_guide=(
                      'Robust z-scores (median and MAD per geo). Dark cells'
                      ' are unusually high or low days worth checking.'
                  ),
              ),
          )
      )
    excluded_dates = sorted(
        pd.Timestamp(d) for d in (self._result.excluded_dates or ())
    )
    blocks.append(
        formatter.Text(
            'Excluded dates: '
            + (
                ', '.join(d.strftime('%Y-%m-%d') for d in excluded_dates)
                if excluded_dates
                else 'none'
            )
        )
    )
    excluded_geos = sorted(str(g) for g in (self._result.excluded_geos or ()))
    blocks.append(
        formatter.Text(
            'Excluded geos: '
            + (
                ', '.join(self._names(g) for g in excluded_geos)
                if excluded_geos
                else 'none'
            )
        )
    )
    return formatter.Card(
        'data',
        'Data checks',
        [b for b in blocks if b is not None],
        audience='technical',
    )

