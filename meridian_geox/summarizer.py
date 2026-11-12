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

"""Interactive HTML design summaries for GeoX.

`DesignSummarizer` turns a `DesignSet` into a self-contained HTML design report.
"""

import dataclasses
import datetime
from typing import NamedTuple, Optional, Sequence, Union

import altair as alt
from meridian_geox import api
from meridian_geox import charts
from meridian_geox import design_charts
from meridian_geox import maps
from meridian_geox import report_data
from meridian_geox import report_text
from meridian_geox.templates import formatter
import pandas as pd

DateLike = Union[None, str, datetime.date, pd.Timestamp]

# Maximum number of candidates with their own detail panels.
MAX_SCOPED_CANDIDATES = 10
_RESERVED_TARGETING_COLUMNS = ('geo_name', 'group')

SELECT_CUE = 'Select a candidate to view its geo split and trade-offs.'


def _goal_tone(value: Optional[bool]) -> formatter.Tone:
  if value is None:
    return 'neutral'
  return 'success' if value else 'warning'


def _takeaway(
    text: str, summary: report_data.CandidateSummary
) -> formatter.TemplateSpec:
  """A '→ ...' line scoped to one candidate."""
  return _section('', [f'→ {text}'], 'geox-takeaway', summary.design_id)


def _section(
    title: str,
    paragraphs: Sequence[Union[str, tuple[str, str]]],
    css: str = '',
    candidate: Optional[str] = None,
) -> formatter.TemplateSpec:
  """A heading and paragraphs; a (label, text) pair gets a bold label."""
  items = [
      {'label': p[0], 'text': p[1]} if isinstance(p, tuple) else {'text': p}
      for p in paragraphs
  ]
  return formatter.TemplateSpec(
      'section.html.jinja',
      {'title': title, 'paragraphs': items, 'css': css},
      candidate=candidate,
  )


def _chart(
    chart: Optional[alt.TopLevelMixin],
    framing: charts.Framing,
    candidate: Optional[str] = None,
    audience: formatter.Audience = 'all',
    css_class: str = '',
) -> Optional[formatter.Spec]:
  if chart is None:
    return None
  return formatter.ChartSpec(
      id='',
      chart_json=charts.to_json(charts.framed(chart, framing)),
      description=framing.takeaway,
      reading_guide=framing.reading_guide,
      candidate=candidate,
      audience=audience,
      css_class=css_class,
  )


def _coverage_callout(
    coverage: maps.BoundaryCoverage, names: report_data.GeoNames
) -> Optional[formatter.Spec]:
  if not coverage.configured or not coverage.checked or not coverage.unmatched:
    return None
  missing = [names(g) for g in coverage.unmatched]
  return formatter.CalloutSpec(
      report_text.MAP_COVERAGE_NOTE.format(
          len(missing), report_text.name_list(missing, 10)
      ),
      title=report_text.MAP_COVERAGE_TITLE,
      tone='warning',
  )


def _map_block(
    table: pd.DataFrame,
    scale: alt.Scale,
    map_config: maps.MapConfig,
    framing: charts.Framing,
    candidate: Optional[str] = None,
    show_legend: bool = True,
) -> formatter.Spec:
  """Boundary map of a geo table, or a setup note without boundaries."""
  if not map_config.polygon_url:
    return formatter.CalloutSpec(
        maps.SETUP_NOTE, title='Map', tone='info', candidate=candidate
    )
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


def _group_title(
    group: str,
    experiment_type: Optional[api.ExperimentType],
    single_treatment: bool,
) -> str:
  """Group label with what happens to its ads, e.g. 'Control · …'."""
  treat_desc, control_desc = (
      report_text.GROUP_DESCRIPTIONS.get(experiment_type)
      if experiment_type is not None
      else None
  ) or ('spend changed', 'business as usual')
  if group == charts.CONTROL_LABEL:
    return f'Control · {control_desc}'
  if group == charts.EXCLUDED_LABEL:
    return 'Excluded · not in the test'
  return f'Treatment · {treat_desc}' if single_treatment else group


def _legend_spec(
    design: api.Design, table: pd.DataFrame
) -> formatter.TemplateSpec:
  """Legend listing each group's size, share and largest geos."""
  experiment_type = report_data.single_experiment_type(design)
  labels = list(report_data.group_labels(design).values())
  colors = charts.group_colors(labels)
  total = float(table['conversions'].sum()) or 1.0
  items = []
  for group, frame in table.groupby('group', sort=False):
    names = frame.sort_values('share', ascending=False)['geo_name']
    items.append({
        'color': colors.get(str(group), charts.GREY_500),
        'title': _group_title(str(group), experiment_type, len(labels) == 1),
        'count': len(frame),
        'share': f'{float(frame["conversions"].sum()) / total:.0%}',
        'names': report_text.name_list([str(n) for n in names], 4),
    })
  return formatter.TemplateSpec('legend.html.jinja', {'items': items})


# Tiles shown only in the Full report.
_ANALYST_TILES = ('Pretest fit', 'Pre-test AA check')

_CHANCE_TONES: dict[str, formatter.Tone] = {
    'High': 'success',
    'Medium': 'warning',
    'Low': 'error',
}


def _chance_text(plan: Optional[report_data.PlanNumbers]) -> str:
  """'High · 85%', or '—' without plan numbers."""
  if plan is None:
    return '—'
  return f'{report_data.chance_label(plan.chance)} · {plan.chance:.0%}'


def _chance_tone(chance: Optional[float]) -> formatter.Tone:
  return _CHANCE_TONES.get(report_data.chance_label(chance), 'neutral')


def _type_chips(
    experiment_type: Optional[api.ExperimentType],
    cells: Sequence[tuple[str, api.ExperimentType]],
) -> list[formatter.ChipSpec]:
  """'Type' chip of a one-type test, else 'Types' from (label, type) cells."""
  if experiment_type is None:
    return [formatter.ChipSpec('Types', ', '.join(
        f'{label} {report_text.spend_terms(et).type_label.lower()}'
        for label, et in cells
    ))]
  terms = report_text.spend_terms(experiment_type)
  return [
      formatter.ChipSpec('Type', f'{terms.type_label} · {terms.explanation}')
  ]


class _Summarizer:
  """Shared plumbing of the report summarizers."""

  def __init__(
      self, geo_metadata: Optional[pd.DataFrame], map_config: maps.MapConfig
  ):
    self._names = report_data.resolve_geo_metadata(geo_metadata)
    self._map_config = map_config

  def _build_report(self) -> str:
    raise NotImplementedError

  def generate_html(self) -> str:
    """Returns the self-contained HTML report."""
    self._env = formatter.create_template_env()
    return self._build_report()

  def _write(self, filename: str, filepath: str) -> str:
    """Writes the report to `<filepath>/<filename>` and returns the path."""
    return formatter.write_html(self.generate_html(), filename, filepath)

  def _page(self, title: str, cards: Sequence[str], **kwargs) -> str:
    return formatter.create_summary_html(self._env, title, cards, **kwargs)

  def _region_chips(
      self, n_geos: int, default: str = ''
  ) -> list[formatter.ChipSpec]:
    region = self._map_config.region_name or default
    if not region:
      return []
    return [formatter.ChipSpec('Region', f'{region} ({n_geos} geos)')]

  def to_iframe(self, height: int = 1100) -> str:
    """Returns the report in an iframe for display(HTML(...)) in notebooks."""
    return formatter.iframe(self.generate_html(), height)

  def _repr_html_(self) -> str:
    return self.to_iframe()


class _Candidate(NamedTuple):
  """Values derived once per shown candidate."""

  summary: report_data.CandidateSummary
  design: api.Design
  plan: Optional[report_data.PlanNumbers]
  table: pd.DataFrame
  scale: alt.Scale
  checks: report_data.DesignChecks
  # Budget scope tip of a holdback budget goal, if any.
  tip: Optional[str]


@dataclasses.dataclass(frozen=True)
class _DesignContext:
  designs: dict[str, api.Design]
  summaries: list[report_data.CandidateSummary]
  scoped: list[_Candidate]
  goals: report_data.DesignGoals
  terms: report_text.SpendTerms
  coverage: maps.BoundaryCoverage
  n_feasible: int
  experiment_type: Optional[api.ExperimentType]


class DesignSummarizer(_Summarizer):
  """Builds the interactive design 2-pager for a `DesignSet`."""

  def __init__(
      self,
      design_set: api.DesignSet,
      geo_metadata: Optional[pd.DataFrame] = None,
      map_config: maps.MapConfig = maps.MapConfig(),
      currency_symbol: str = '',
      target_mde: Optional[float] = None,
      targeting_map: Optional[pd.DataFrame] = None,
      test_start: DateLike = None,
      cooldown_days: int = 14,
      ramp_up_days: int = 0,
  ):
    """Initializes the summarizer.

    Args:
      design_set: Output of `geox.run_design(...)`.
      geo_metadata: Display names: `location` plus `geo_name` or `name`.
      map_config: Boundary polygons for the map panels.
      currency_symbol: Symbol prefixed to monetary values.
      target_mde: Largest acceptable relative MDE, for example 0.05.
      targeting_map: Optional `location` to ad-platform target rows (for
        example postal codes), adding a targeting CSV download.
      test_start: Planned first test day (after the pretest window).
      cooldown_days: Planned cooldown length in days.
      ramp_up_days: First test days used to ramp spend up; shaded as
        ramp-up in the calendar and named in the launch plan.

    Raises:
      ValueError: On invalid inputs.
    """
    if design_set.design_metrics is None or design_set.design_metrics.empty:
      raise ValueError('design_set must have non-empty design_metrics.')
    if not design_set.designs:
      raise ValueError('design_set must contain at least one design.')
    self._goals = report_data.DesignGoals(target_mde=target_mde)
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
    super().__init__(geo_metadata, map_config)
    self._currency = currency_symbol
    self._text = report_text.DesignText(currency_symbol)
    self._targeting_map = targeting_map
    first = next(iter(design_set.designs.values()))
    self._schedule = report_data.experiment_schedule(
        first, test_start, cooldown_days, ramp_up_days
    )

  output_design_summary = _Summarizer._write

  # ------------------------------------------------------------------------

  def _build_report(self) -> str:
    designs = self._design_set.designs
    goals = report_data.default_goals(next(iter(designs.values())), self._goals)
    summaries = report_data.design_summaries(self._design_set, goals)
    if not summaries:
      raise ValueError('No candidate designs could be summarized.')
    best = summaries[0]
    best_design = designs[best.design_id]
    experiment_type = report_data.single_experiment_type(best_design)
    # Every candidate splits the same geos into treatment, control and excluded.
    geos = {*best_design.control_geos, *(best_design.excluded_geos or ())}
    for c in best_design.designs.values():
      geos.update(c.treatment_geos)
    all_geos = sorted(geos)
    context = _DesignContext(
        designs=designs,
        summaries=summaries,
        scoped=[
            self._candidate(s, goals)
            for s in summaries[:MAX_SCOPED_CANDIDATES]
        ],
        goals=goals,
        terms=report_text.spend_terms(experiment_type),
        coverage=maps.boundary_coverage(all_geos, self._map_config),
        n_feasible=sum(s.feasible for s in summaries),
        experiment_type=experiment_type,
    )
    # UX order: recommendation, then the comparison and its evidence, then
    # the launch plan and next steps, then details.
    cards = [
        self._recommended_card(context),
        self._compare_card(context),
        self._visuals_card(context),
        self._launch_card(context),
        self._details_card(context),
    ]
    return self._page(
        'GeoX experiment design',
        cards,
        subtitle=report_text.DESIGN_REPORT_SUBTITLE,
        kind='design',
        chips=self._chips(best_design, experiment_type, goals, all_geos),
        default_candidate=best.design_id,
    )

  def _candidate(
      self, s: report_data.CandidateSummary, goals: report_data.DesignGoals
  ) -> _Candidate:
    """Values derived once per shown candidate, with its budget tip."""
    design = self._design_set.designs[s.design_id]
    plan = report_data.plan_numbers(s, design, goals)
    tip = None
    scope = report_data.budget_scope(s, design, goals.target_budget, plan)
    if scope is not None:
      p = scope.chance
      tip = report_text.BUDGET_TIP.format(
          budget=self._text.short(scope.budget),
          treatment=self._text.short(scope.treatment_budget),
          share=scope.share,
          chance='' if p is None else report_text.BUDGET_TIP_CHANCE.format(p),
      )
    return _Candidate(
        s,
        design,
        plan,
        report_data.geo_table(design, self._names),
        charts.group_scale(list(report_data.group_labels(design).values())),
        report_data.design_checks(s, design),
        tip,
    )

  def _chips(
      self,
      design: api.Design,
      experiment_type: Optional[api.ExperimentType],
      goals: report_data.DesignGoals,
      all_geos: Sequence[str],
  ) -> list[formatter.ChipSpec]:
    """Header chips: region, test type, pretest window and goals."""
    chips = self._region_chips(len(all_geos), 'Geos')
    pretest = report_data.pretest_range(design)
    if pretest is not None:
      chips.append(
          formatter.ChipSpec('Pretest', report_text.date_range(*pretest))
      )
    config = design.design_config
    if config is not None:
      chips.append(formatter.ChipSpec(
          'Duration', f'{config.experiment_duration.days} days'
      ))
    chips += _type_chips(experiment_type, [
        (report_text.short_label(c.label), c.experiment_type)
        for c in report_data.spend_changes(design)
    ])
    if experiment_type is None:
      chips.append(formatter.ChipSpec('Cells', str(len(design.designs))))
    if config is not None:
      method = config.methodology.name
      rule = config.geo_assignment_rule.name.replace('_', ' ').title()
      chips.append(formatter.ChipSpec('Method', f'{method} · {rule}'))
    goal_parts = []
    if goals.target_mde is not None:
      goal_parts.append(f'MDE ≤ {report_text.format_percent(goals.target_mde)}')
    if goals.target_budget is not None:
      goal_parts.append(f'budget ≤ {self._text.short(goals.target_budget)}')
    if goals.target_cpic is not None:
      goal_parts.append('target iCPA ' + self._text.money(goals.target_cpic, 2))
    if goal_parts:
      chips.append(formatter.ChipSpec('Goals', ', '.join(goal_parts)))
    return chips

  # ---------------------------------------------------------------- recommend

  def _narrative(
      self,
      s: report_data.CandidateSummary,
      design: api.Design,
      context: _DesignContext,
      plan: Optional[report_data.PlanNumbers],
  ) -> str:
    """Banner sentences of one candidate."""
    text = self._text
    goals = context.goals
    alpha, _, test_type = report_data.design_params(design)
    sides = 'two-sided' if test_type == api.TestType.TWO_SIDED else 'one-sided'
    rows = [
        (c.label, c.experiment_type, c.amount, c.planned_change)
        for c in report_data.spend_changes(design)
    ]
    parts = [
        report_text.DETECTION_SENTENCE.format(
            report_text.format_percent(s.max_mde), plan.power if plan else 0.8,
            alpha, sides,
        ),
        text.spend_sentence(
            rows, s.total_spend,
            context.terms.label, _test_days(design),
        ),
    ]
    if plan is not None:
      parts.append(text.chance_sentence(
          plan.chance, plan.cpic, goals.target_budget,
          any(c.experiment_type == api.ExperimentType.HOLDBACK
              for c in s.cells.values()),
      ))
    parts.append(text.goals_sentence(
        goals.any_set,
        report_data.missed_goals(s),
        context.n_feasible,
        len(context.summaries),
    ))
    return ' '.join(parts)

  def _design_tiles(
      self,
      s: report_data.CandidateSummary,
      design: api.Design,
      context: _DesignContext,
      plan: Optional[report_data.PlanNumbers],
      checks: report_data.DesignChecks,
  ) -> list[formatter.StatsSpec]:
    """Tiles of one candidate: chance, spend, MDE, split, fit, A/A."""
    text = self._text
    goals = context.goals
    days = _test_days(design)
    tiles = []
    if plan is not None:
      tiles.append(formatter.StatsSpec(
          'Feasibility',
          f'{report_data.chance_label(plan.chance)} chance',
          detail=text.chance_detail(
              plan.chance, goals.target_budget is not None,
              goals.target_cpic is not None,
          ),
          ci=report_text.CHANCE_LEGEND,
          tone=_chance_tone(plan.chance),
      ))
    changes = report_data.spend_changes(design)
    spend, spend_detail, spend_formula = text.spend_tile(
        [(c.label, c.experiment_type, c.amount, c.planned_change)
         for c in changes],
        s.total_spend,
        goals.target_budget if s.meets_budget is not None else None,
        report_data.bau_spend(changes[0]) if len(changes) == 1 else None,
        days,
        (plan.mde_abs, plan.cpic) if plan is not None else None,
    )
    tiles.append(formatter.StatsSpec(
        context.terms.label,
        spend,
        detail=spend_detail,
        ci=spend_formula,
        tone=_goal_tone(s.meets_budget),
    ))
    tiles.append(formatter.StatsSpec(
        'Minimum detectable effect',
        report_text.format_percent(s.max_mde),
        detail=text.mde_detail(goals.target_mde, s.meets_mde),
        ci=text.mde_formula(
            [(c.label, c.mde) for c in s.cells.values()], s.total_mde_abs, days
        ),
        tone=_goal_tone(s.meets_mde),
    ))
    fit, fit_detail = report_text.FIT_VALUE[checks.good_fit]
    tiles.append(formatter.StatsSpec(
        'Pretest fit',
        fit,
        detail=fit_detail,
        ci=report_text.FIT_DETAIL.format(checks.r2, checks.min_r2),
        tone='success' if checks.good_fit else 'warning',
    ))
    tiles.append(formatter.StatsSpec(
        'Pre-test AA check',
        report_text.AA_VALUE[checks.aa_passed],
        detail=text.aa_text(checks.aa_passed, checks.aa_window),
        ci=report_text.AA_DETAIL.format(checks.aa_p_value, checks.alpha),
        tone='success' if checks.aa_passed else 'error',
    ))
    n_excluded = len(design.excluded_geos or ())
    tiles.append(formatter.StatsSpec(
        'Regions split',
        f'{s.treatment_geo_count} / {s.control_geo_count} / {n_excluded}',
        detail='treatment / control / excluded',
        ci=text.split_detail(
            checks.treatment_share,
            [(c.label, c.treatment_geo_count) for c in s.cells.values()],
        ),
    ))
    return tiles

  def _checks(
      self, c: _Candidate, context: _DesignContext
  ) -> list[report_text.Check]:
    s, checks, goals, text = c.summary, c.checks, context.goals, self._text
    line, pct = report_text.Check, report_text.format_percent
    out = [
        line(
            'pass' if checks.good_fit else 'warn',
            report_text.FIT_VALUE[checks.good_fit][1],
            report_text.FIT_DETAIL.format(checks.r2, checks.min_r2),
        ),
        line(
            'pass' if checks.aa_passed else 'fail',
            text.aa_text(checks.aa_passed, checks.aa_window),
            report_text.AA_DETAIL.format(checks.aa_p_value, checks.alpha),
        ),
    ]
    if goals.target_mde is not None:
      out.append(line(
          'pass' if s.meets_mde else 'warn',
          report_text.CHECK_MDE.format(pct(s.max_mde), pct(goals.target_mde)),
      ))
    if goals.target_budget is not None and any(
        cell.experiment_type == api.ExperimentType.HOLDBACK
        for cell in s.cells.values()
    ):
      budget = report_text.CHECK_BUDGET.format(
          context.terms.label, text.short(s.total_spend),
          text.short(goals.target_budget),
      )
      out.append(line('pass' if s.meets_budget else 'warn', budget))
      if c.tip is not None:
        out.append(line('info', c.tip))
    share = report_text.CHECK_SHARE.format(checks.treatment_share)
    out.append(line('pass', share))
    coverage = context.coverage
    if not coverage.configured:
      out.append(line('info', report_text.CHECK_NO_MAP))
    elif coverage.checked:
      missing = [self._names(g) for g in coverage.unmatched]
      total = len(coverage.matched) + len(missing)
      matched = report_text.CHECK_BOUNDARIES.format(total - len(missing), total)
      if missing:
        missing_text = report_text.name_list(missing, 5)
        matched += report_text.CHECK_MISSING.format(missing_text)
      out.append(line('warn' if missing else 'pass', matched))
    quality = report_data.data_quality(c.design)
    flagged = quality.flagged_geos
    user = [self._names(g) for g in quality.user_geos if g not in flagged]
    if user:
      out.append(line(
          'info', report_text.CHECK_USER.format(report_text.name_list(user, 5))
      ))
    for template, items, noun in (
        (report_text.CHECK_FLAGGED, flagged, 'geo'),
        (report_text.CHECK_EXCLUDED_DATES, quality.excluded_dates,
         'pretest date'),
    ):
      if items:
        out.append(line(
            'warn', template.format(report_text.plural(len(items), noun))
        ))
    return out

  def _recommended_card(self, context: _DesignContext) -> str:
    """The Recommended design card."""
    blocks: list[formatter.Spec] = []
    for c in context.scoped:
      s, design, plan = c.summary, c.design, c.plan
      if s.rank != 1:
        best = context.summaries[0]
        blocks.append(formatter.CalloutSpec(
            report_text.INSPECTING_NOTE.format(s.name, best.name),
            title=f'Inspecting {s.name}',
            tone='warning',
            candidate=s.design_id,
        ))
      blocks.append(formatter.TemplateSpec(
          'option_banner.html.jinja',
          {
              'warn': plan is not None and plan.chance < 0.8,
              'name': s.name + (
                  '' if plan is None else report_text.CHANCE_TITLE.format(
                      report_data.chance_label(plan.chance).lower()
                  )
              ),
              'narrative': self._narrative(s, design, context, plan),
              'learn_more_url': report_text.DOCS_INTRO_TO_DESIGN,
          },
          candidate=s.design_id,
      ))
      tiles = self._design_tiles(s, design, context, plan, c.checks)
      blocks.append(formatter.StatsSectionSpec(
          [t for t in tiles if t.title not in _ANALYST_TILES],
          candidate=s.design_id,
      ))
      blocks.append(formatter.StatsSectionSpec(
          [t for t in tiles if t.title in _ANALYST_TILES],
          candidate=s.design_id,
          audience='technical',
      ))
    return formatter.create_card_html(
        self._env,
        formatter.CardSpec(
            'recommended',
            'Recommended design',
            subtitle=report_text.RECOMMENDED_SUBTITLE,
        ),
        blocks,
    )

  def _notes(self, context: _DesignContext) -> list[formatter.Spec]:
    """Per-candidate spend definition and budget notes."""
    notes: list[formatter.Spec] = []
    experiment_type = context.experiment_type
    for c in context.scoped:
      s = c.summary
      if experiment_type is not None:
        notes.append(formatter.CalloutSpec(
            report_text.SPEND_DEFINITIONS[experiment_type],
            title=context.terms.label,
            tone='info',
            candidate=s.design_id,
            links=[('Experiment types', report_text.DOCS_EXPERIMENT_TYPES)],
        ))
      if c.tip is not None:
        notes.append(formatter.CalloutSpec(
            c.tip, title=report_text.BUDGET_SCOPE_TITLE, tone='neutral',
            candidate=s.design_id,
        ))
      if report_data.looks_national(c.design, context.goals.target_budget):
        notes.append(formatter.CalloutSpec(
            report_text.NATIONAL_BUDGET_NOTE,
            title='Check the budget scope',
            tone='warning',
            candidate=s.design_id,
        ))
    return notes

  # ------------------------------------------------------------------ visuals

  def _regions_sentence(
      self, s: report_data.CandidateSummary, table: pd.DataFrame
  ) -> str:
    treatment = table[~table['cell'].isin(['control', 'excluded'])]
    share = float(treatment['share'].sum())
    largest = ''
    if share > 0:
      top = treatment.sort_values('share', ascending=False).iloc[0]
      largest = report_text.REGIONS_LARGEST.format(
          str(top['geo_name']), float(top['share']) / share
      )
    return report_text.REGIONS_SENTENCE.format(
        s.name, len(treatment), share, largest, s.control_geo_count
    )

  def _candidate_map(
      self, c: _Candidate, show_legend: bool
  ) -> formatter.Spec:
    """Boundary map of a candidate's split, with its regions sentence."""
    return _map_block(
        c.table,
        c.scale,
        self._map_config,
        charts.Framing(
            'Where would the test run?',
            reading_guide=report_text.MAP_GUIDE,
            takeaway=self._regions_sentence(c.summary, c.table),
        ),
        candidate=c.summary.design_id,
        show_legend=show_legend,
    )

  def _visuals_card(self, context: _DesignContext) -> str:
    text = self._text
    blocks: list[formatter.Spec] = [
        formatter.CalloutSpec(report_text.VISUALS_INTRO, tone='neutral')
    ]
    for c in context.scoped:
      s, design, table = c.summary, c.design, c.table
      # The group list next to the map doubles as its legend.
      map_block = self._candidate_map(c, show_legend=False)
      power = c.plan.power if c.plan else 0.8
      sensitivity = _chart(
          design_charts.sensitivity_curve(
              s, design, context.goals.target_mde
          ),
          charts.Framing(
              report_text.SENSITIVITY_TITLE,
              reading_guide=text.sensitivity_guide(s.max_mde, power),
              takeaway=text.sensitivity_takeaway(s.name, s.max_mde, power),
          ),
      )
      row: list[formatter.Spec] = [
          formatter.RowSpec([map_block, _legend_spec(design, table)])
      ]
      if sensitivity is not None:
        row.append(sensitivity)
      blocks.append(formatter.RowSpec(row, candidate=s.design_id))
    coverage = _coverage_callout(context.coverage, self._names)
    if coverage is not None:
      blocks.append(coverage)
    return formatter.create_card_html(
        self._env,
        formatter.CardSpec('visuals', report_text.VISUALS_TITLE),
        blocks,
    )

  # ------------------------------------------------------------------ compare

  def _compare_card(self, context: _DesignContext) -> str:
    goals, label = context.goals, context.terms.label
    best = context.summaries[0]
    rows, tones, takeaways, maps_, tradeoffs = [], [], [], [], []
    evidence: list[formatter.Spec] = []
    simulators: list[Optional[formatter.Spec]] = []
    for c in context.scoped:
      s, plan, checks = c.summary, c.plan, c.checks
      rows.append([
          ('★ ' if s.rank == 1 else '') + s.name,
          _chance_text(plan),
          report_text.format_percent(s.max_mde),
          self._text.short(s.total_spend),
          self._text.money(s.max_detectable_cpic, 2),
          f'{checks.treatment_share:.0%}',
          f'{checks.r2:.2f}',
          ('✓ ' if checks.aa_passed else '✗ ') + f'{s.min_aa_p_value:.2f}',
          'Meets all goals' if s.feasible else 'Misses a goal',
      ])
      tones.append('success' if s.feasible else 'warning')
      detects = report_text.DETECTS_WITH.format(
          report_text.format_percent(s.max_mde),
          self._text.short(s.total_spend),
          label.lower(),
      )
      takeaways.append(_takeaway(report_text.INSPECT_TEXT[s.rank == 1].format(
          name=s.name,
          best=best.name,
          chance='' if plan is None else report_text.INSPECT_CHANCE.format(
              report_data.chance_label(plan.chance), plan.chance
          ),
          detects=detects,
      ), s))
      if self._map_config.polygon_url:
        maps_.append(self._candidate_map(c, show_legend=True))
      if s.feasible:
        zone = report_text.ZONE_INSIDE
      elif s.meets_budget is False and goals.target_budget is not None:
        zone = report_text.ZONE_OVER_BUDGET.format(
            self._text.short(goals.target_budget)
        )
      elif goals.target_mde is not None and s.max_mde > goals.target_mde:
        zone = report_text.ZONE_OVER_MDE.format(
            report_text.format_percent(goals.target_mde)
        )
      else:
        zone = report_text.ZONE_MISSES
      tradeoffs.append(_takeaway(f'{s.name} {detects}, {zone}.', s))
      evidence.extend(self._evidence_blocks(c))
      simulators.append(self._simulator_block(c, context))
    table = formatter.TableSpec(
        id='',
        title='',
        column_headers=[
            'Option', 'Chance', 'MDE', label, 'Max iCPA', 'Share', 'Fit R²',
            'A/A', 'Goals',
        ],
        row_values=rows,
        row_ids=[c.summary.design_id for c in context.scoped],
        selectable=True,
        row_tones=tones,
        description=report_text.COMPARE_TABLE_HELP,
    )
    tradeoff = _chart(
        design_charts.tradeoff_toggle_chart(
            [c.summary for c in context.scoped], goals, self._currency, label
        ),
        charts.Framing(
            report_text.TRADEOFF_TITLE,
            reading_guide=report_text.TRADEOFF_GUIDE,
        ),
    )
    # Takeaways follow the selected candidate; the trade-off chart sits next
    # to the selected candidate's geo split, as in the v3 layout.
    blocks: list[formatter.Spec] = [table, *takeaways]
    if tradeoff is not None or maps_:
      blocks.append(formatter.RowSpec(
          [b for b in [tradeoff, *maps_] if b is not None]
      ))
    if tradeoff is not None:
      blocks.extend(tradeoffs)
    blocks.extend(evidence)
    blocks.append(_section(
        report_text.SIMULATOR_TITLE, [report_text.SIMULATOR_INTRO]
    ))
    blocks.extend(b for b in simulators if b is not None)
    blocks.append(formatter.CalloutSpec(
        report_text.ICPA_TIP,
        title='Setting the target iCPA',
        tone='info',
        links=[
            ('How to choose it', report_text.DOCS_TROUBLESHOOTING),
            ('Glossary: iCPA', report_text.DOCS_ICPA_GLOSSARY),
        ],
    ))
    blocks.extend(self._notes(context))
    # As in v3, the comparison is analyst detail: Full report only.
    return formatter.create_card_html(
        self._env,
        formatter.CardSpec(
            'compare',
            'Compare candidate designs',
            subtitle=report_text.COMPARE_SUBTITLE,
            audience='technical',
            cue=SELECT_CUE,
        ),
        blocks,
    )

  def _evidence_blocks(self, c: _Candidate) -> list[formatter.Spec]:
    text = self._text
    s, design, table, checks = c.summary, c.design, c.table, c.checks
    scale = c.scale
    largest = report_data.largest_geos(table)
    row: list[Optional[formatter.Spec]] = [
        _chart(
            design_charts.largest_geos_chart(table, scale),
            charts.Framing(
                report_text.LARGEST_GEOS_TITLE,
                reading_guide=report_text.LARGEST_GEOS_GUIDE,
                takeaway=text.largest_geos_takeaway(
                    s.name, largest.treated, largest.count,
                    largest.spend_tracks,
                ),
            ),
        ),
        _chart(
            design_charts.pretest_fit_chart(design),
            charts.Framing(
                report_text.PRETEST_FIT_TITLE,
                reading_guide=report_text.PRETEST_FIT_GUIDE,
                takeaway=report_text.FIT_TAKEAWAY[
                    checks.r2 >= checks.min_r2
                ].format(s.name, checks.r2, checks.min_r2),
            ),
        ),
    ]
    blocks: list[formatter.Spec] = [formatter.RowSpec(
        [b for b in row if b is not None], candidate=s.design_id
    )]
    # One drift chart per cell (multi-cell designs have one per treatment).
    multi = len(s.cells) > 1
    drift_blocks: list[formatter.Spec] = []
    for cell, summary in s.cells.items():
      drift = report_data.aa_drift(design, cell)
      if drift is None:
        continue
      block = _chart(
          design_charts.aa_drift_chart(drift, summary.mde_abs),
          charts.Framing(
              report_text.AA_DRIFT_TITLE
              + (f' · {summary.label}' if multi else ''),
              reading_guide=report_text.AA_DRIFT_GUIDE,
              takeaway=text.drift_takeaway(
                  s.name, drift.start, drift.end, drift.final,
                  report_data.drift_ratio(drift, summary.mde_abs),
                  summary.aa_p_value,
              ),
          ),
          candidate=s.design_id,
      )
      if block is not None:
        drift_blocks.append(block)
    if len(drift_blocks) > 1:
      blocks.append(formatter.RowSpec(drift_blocks, candidate=s.design_id))
    else:
      blocks.extend(drift_blocks)
    return blocks

  def _simulator_block(
      self, c: _Candidate, context: _DesignContext
  ) -> Optional[formatter.Spec]:
    s, design, plan, goals = c.summary, c.design, c.plan, context.goals
    # The guardrail compares the target iCPA with the iCPA the design was
    # planned at (DesignConfig.cost_per_incremental_conversion).
    reference = report_data.planned_cpic(design) or goals.target_cpic
    # The headline numbers live in the chart, so they follow the controls.
    takeaway = None
    if plan is not None:
      takeaway = report_text.SIMULATOR_TAKEAWAY.format(
          spend=self._text.money(plan.spend),
          label=context.terms.label.lower(),
          cpic=self._text.money(plan.cpic, 2),
          chance=plan.chance,
          power=plan.power,
          needed=self._text.money(plan.spend_for_power),
      )
    return _chart(
        design_charts.power_simulator(
            s, design, goals, self._currency, context.terms.label, reference
        ),
        charts.Framing(
            report_text.SIMULATOR_CHART_TITLE,
            reading_guide=report_text.SIMULATOR_GUIDE,
            takeaway=takeaway,
        ),
        candidate=s.design_id,
        css_class=design_charts.SIMULATOR_CLASS,
    )

  # ------------------------------------------------------------------- launch

  def _delivery_steps(
      self, s: report_data.CandidateSummary, design: api.Design, unit: str
  ) -> list[str]:
    """Where and how spend changes, one step per cell."""
    changes = report_data.spend_changes(design)
    multi, m = len(changes) > 1, s.control_geo_count
    steps = []
    for c in changes:
      name = report_text.short_label(c.label)
      et = c.experiment_type
      key = et if et in report_text.DELIVERY_STEPS else unit
      control = '' if multi else report_text.DELIVERY_CONTROL[key]
      pct = report_text.change_percent(c.planned_change)
      steps.append((report_text.DELIVERY_STEPS[key] + control + '.').format(
          prefix=f'{name}: ' if multi else '',
          n=s.cells.get(c.cell, s).treatment_geo_count,
          group=f'{name if multi else "treatment"} {unit}',
          value=self._text.short(abs(c.amount)),
          share=f', {pct} of BAU' if pct else '',
          by=f' by {pct}' if pct else '',
          m=m,
          unit=unit,
      ))
    if multi:
      steps.append(report_text.SHARED_CONTROL_STEP.format(m=m, unit=unit))
    return steps

  def _launch_card(self, context: _DesignContext) -> str:
    text = self._text
    schedule = self._schedule
    cooldown = bool(schedule.cooldown_days)
    span = report_text.schedule_span
    blocks: list[formatter.Spec] = []
    for s, design, plan, table, _, checks, _ in context.scoped:
      steps = [
          *self._delivery_steps(s, design, unit='regions'),
          report_text.RUN_STEP[cooldown].format(span=span(schedule, True)),
          report_text.REVIEW_STEP,
      ]
      if plan is not None and plan.chance < 0.8:
        steps.append(report_text.POWER_STEP.format(
            spend=text.short(plan.spend_for_power),
            label=context.terms.label.lower(),
        ))
      buttons = [formatter.Button(
          'Download geo list (CSV)', _geo_csv(table),
          f'geox_option{s.rank}_geos.csv',
      )]
      if self._targeting_map is not None:
        buttons.append(formatter.Button(
            'Download targeting list (CSV)',
            _targeting_csv(table, self._targeting_map),
            f'geox_option{s.rank}_targeting.csv',
        ))
      how_it_works = formatter.TemplateSpec('how_it_works.html.jinja', {
          'n_treatment': s.treatment_geo_count,
          'share': f'{checks.treatment_share:.0%}',
          'n_control': s.control_geo_count,
          'days': _test_days(design),
          'validation': report_text.VALIDATION_TEXT[checks.aa_passed],
      })
      blocks.append(formatter.RowSpec(
          [
              how_it_works,
              formatter.TemplateSpec(
                  'steps.html.jinja', {'title': 'Launch plan', 'steps': steps}
              ),
          ],
          candidate=s.design_id,
      ))
      blocks.append(formatter.ButtonsSpec(buttons, candidate=s.design_id))
      blocks.append(formatter.TemplateSpec(
          'steps.html.jinja',
          {
              'title': 'Next steps',
              'save': {'name': s.name},
              'steps': [
                  *self._delivery_steps(s, design, unit='geos'),
                  report_text.KEEP_STEP[cooldown].format(span=span(schedule)),
                  report_text.ANALYZE_STEP[schedule.dated].format(
                      start=schedule.test_start
                  ),
              ],
          },
          candidate=s.design_id,
          audience='technical',
      ))
    first = context.designs[context.summaries[0].design_id]
    calendar = _chart(
        design_charts.calendar_chart(schedule),
        charts.Framing(
            'When does the test run?',
            reading_guide=report_text.CALENDAR_GUIDE[
                bool(schedule.ramp_up_days)
            ],
            takeaway=report_text.calendar_takeaway(
                schedule, report_data.pretest_range(first)
            ),
        ),
    )
    blocks.append(calendar or formatter.TextSpec(
        report_text.PHASES_TEXT[cooldown].format(s=schedule)
    ))
    return formatter.create_card_html(
        self._env,
        formatter.CardSpec(
            'launch', 'Launch plan', subtitle=report_text.LAUNCH_SUBTITLE
        ),
        blocks,
    )

  # ------------------------------------------------------------------ details

  def _roster_specs(
      self, s: report_data.CandidateSummary, table: pd.DataFrame
  ) -> list[formatter.Spec]:
    """Group-by-group geo roster of one candidate, collapsed as in v3."""
    single = len(s.cells) == 1
    experiment_type = (
        next(iter(s.cells.values())).experiment_type if single else None
    )
    rows = [
        [
            _group_title(str(group), experiment_type, single),
            str(len(frame)),
            f'{float(frame["share"].sum()):.1%}',
            ', '.join(sorted(str(n) for n in frame['geo_name'])),
        ]
        for group, frame in table.groupby('group', sort=False)
    ]
    return [
        _section(
            '', [report_text.ROSTER_NOTE[s.rank == 1].format(s.name)],
            'geox-detail', s.design_id,
        ),
        formatter.TableSpec(
            id='',
            title=f'Geo roster of {s.name}',
            column_headers=[
                'Group', 'Count', 'Share of pretest conversions', 'Geos (A–Z)'
            ],
            row_values=rows,
            collapsed_summary='Show all geos',
            candidate=s.design_id,
        ),
    ]

  def _candidates_table(
      self, context: _DesignContext
  ) -> formatter.TemplateSpec:
    """Ranked table of all candidates with CSV downloads."""
    rows = []
    for c in context.scoped:
      s, plan, table = c.summary, c.plan, c.table
      missed = report_data.missed_goals(s)
      downloads = [formatter.Button(
          'Geos CSV', _geo_csv(table), f'geox_option{s.rank}_geos.csv'
      )]
      if self._targeting_map is not None:
        downloads.append(formatter.Button(
            'Targeting CSV', _targeting_csv(table, self._targeting_map),
            f'geox_option{s.rank}_targeting.csv',
        ))
      rows.append({
          'cells': [
              s.name,
              _chance_text(plan),
              report_text.format_percent(s.max_mde),
              self._text.short(s.total_spend),
              str(s.treatment_geo_count),
              'Misses ' + ', '.join(missed) if missed else 'Meets goals',
          ],
          'downloads': downloads,
      })
    return formatter.TemplateSpec('candidates_table.html.jinja', {
        'columns': [
            'Option', 'Chance', 'MDE', context.terms.label,
            'Treatment geos', 'Goals', 'Export',
        ],
        'rows': rows,
        'note': report_text.CANDIDATES_NOTE[self._targeting_map is not None],
    })

  def _details_card(self, context: _DesignContext) -> str:
    blocks: list[formatter.Spec] = [
        formatter.TemplateSpec(
            'section.html.jinja',
            {'title': 'Design checks', 'checks': self._checks(c, context)},
            candidate=c.summary.design_id,
            audience='technical',
        )
        for c in context.scoped
    ]
    for c in context.scoped:
      s, table = c.summary, c.table
      blocks.extend(self._roster_specs(s, table))
      strata = _chart(
          design_charts.strata_chart(table, c.scale),
          charts.Framing(
              report_text.STRATA_TITLE,
              reading_guide=report_text.STRATA_GUIDE,
              takeaway=report_text.STRATA_TAKEAWAY.format(name=s.name),
          ),
          candidate=s.design_id,
      )
      if strata is not None:
        blocks.append(strata)
    blocks.append(self._candidates_table(context))
    return formatter.create_card_html(
        self._env,
        formatter.CardSpec(
            'details',
            'Candidate details and export',
            audience='technical',
            cue=SELECT_CUE,
        ),
        blocks,
    )


def _test_days(design: api.Design) -> int:
  config = design.design_config
  return config.experiment_duration.days if config else 0
