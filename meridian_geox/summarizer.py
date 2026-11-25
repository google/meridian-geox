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

import collections
import dataclasses
from typing import Callable, NamedTuple, Optional, Sequence, Union

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

# Maximum number of candidates with their own detail panels.
MAX_SCOPED_CANDIDATES = 10
_STATUS_COLORS = {
    report_text.STATUS_OK: charts.DARK_BLUE,
    report_text.STATUS_FLAGGED: '#F29900',
    report_text.STATUS_NO_CONVERSIONS: charts.RED,
    report_text.STATUS_USER: charts.GREY_700,
}

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


def _reciprocal(value: float) -> float:
  return 1.0 / value if value > 0 else float('inf')


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
  """Targeting rows of the test geos, matched on normalized geo ID."""

  def keyed(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.assign(
        _key=[maps.normalize_geo_id(v) for v in frame['location']]
    )

  columns = ['location', 'geo_name', 'group']
  geos = table.loc[table['cell'] != 'excluded', columns]
  merged = keyed(geos).drop(columns='location').merge(
      keyed(targeting_map), on='_key', how='inner'
  )
  columns += [c for c in targeting_map.columns if c != 'location']
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
      test_start: report_data.DateLike = None,
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
      reserved = [c for c in columns if c in ('geo_name', 'group')]
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
        self._quality_card(best_design),
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
      self, s: report_data.CandidateSummary,
      changes: Sequence[report_data.SpendChange], unit: str,
  ) -> list[str]:
    """Where and how spend changes, one step per cell."""
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
    for c in context.scoped:
      s, plan, checks = c.summary, c.plan, c.checks
      changes = report_data.spend_changes(c.design)
      steps = [
          *self._delivery_steps(s, changes, unit='regions'),
          report_text.RUN_STEP[cooldown].format(span=span(schedule, True)),
          report_text.REVIEW_STEP,
      ]
      if plan is not None and plan.chance < 0.8:
        steps.append(report_text.POWER_STEP.format(
            spend=text.short(plan.spend_for_power),
            label=context.terms.label.lower(),
        ))
      buttons = [formatter.Button(
          'Download geo list (CSV)', _geo_csv(c.table),
          f'geox_option{s.rank}_geos.csv',
      )]
      if self._targeting_map is not None:
        buttons.append(formatter.Button(
            'Download targeting list (CSV)',
            _targeting_csv(c.table, self._targeting_map),
            f'geox_option{s.rank}_targeting.csv',
        ))
      how_it_works = formatter.TemplateSpec('how_it_works.html.jinja', {
          'n_treatment': s.treatment_geo_count,
          'share': f'{checks.treatment_share:.0%}',
          'n_control': s.control_geo_count,
          'days': _test_days(c.design),
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
                  *self._delivery_steps(s, changes, unit='geos'),
                  report_text.KEEP_STEP[cooldown].format(span=span(schedule)),
                  report_text.ANALYZE_STEP[schedule.dated].format(
                      start=schedule.test_start
                  ),
              ],
          },
          candidate=s.design_id,
          audience='technical',
      ))
    calendar = _chart(
        design_charts.calendar_chart(schedule),
        charts.Framing(
            'When does the test run?',
            reading_guide=report_text.CALENDAR_GUIDE[
                bool(schedule.ramp_up_days)
            ],
            takeaway=report_text.calendar_takeaway(
                schedule, report_data.pretest_range(context.scoped[0].design)
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

  # ------------------------------------------------------------------ quality

  def _usable_geos_block(
      self, status: dict[str, str], findings: tuple[list[str], list[str]]
  ) -> list[formatter.Spec]:
    """The usable-data map with the findings (or text without boundaries)."""
    warnings, infos = findings
    text = self._text.usable_geos_text(
        collections.Counter(status.values()), len(status)
    )
    title = 'Which geos have usable data?'
    if status and self._map_config.polygon_url:
      table = pd.DataFrame({
          'location': list(status),
          'geo_name': [self._names(g) for g in status],
          'group': list(status.values()),
      })
      scale = alt.Scale(
          domain=[*_STATUS_COLORS, charts.NOT_IN_TEST_LABEL],
          range=[*_STATUS_COLORS.values(), charts.GREY_100],
      )
      block = _chart(
          maps.plot_geo_assignment(table, self._map_config, scale, title=''),
          charts.Framing(
              title, reading_guide=report_text.USABLE_GEOS_GUIDE,
              takeaway=text,
          ),
      )
      if isinstance(block, formatter.ChartSpec):
        # Meridian's warnings and infos, shown above the usable-data map.
        return [dataclasses.replace(block, warnings=warnings, infos=infos)]
    return [
        *(formatter.CalloutSpec(t, tone='warning') for t in warnings),
        *(formatter.CalloutSpec(t, tone='info') for t in infos),
        _section(title, [f'→ {text}'], 'geox-takeaway'),
    ]

  def _quality_card(self, design: api.Design) -> str:
    text, quality, names = (
        self._text, report_data.data_quality(design), self._names
    )

    def display(geos: Sequence[str]) -> list[str]:
      return [names(g) for g in geos]

    findings = text.quality_findings(
        display(quality.flagged_geos), quality.flagged_dates,
        display(quality.user_geos), quality.excluded_dates,
    )
    blocks: list[Optional[formatter.Spec]] = [
        *self._usable_geos_block(quality.geo_status, findings),
        _chart(
            design_charts.daily_spend_chart(design),
            charts.Framing(report_text.DAILY_SPEND_TITLE),
        ),
        _chart(
            design_charts.data_quality_chart(design),
            charts.Framing(
                report_text.UNUSUAL_DAYS_TITLE,
                reading_guide=report_text.UNUSUAL_DAYS_GUIDE,
                takeaway=text.unusual_days_takeaway(quality.excluded_dates),
            ),
        ),
    ]
    excluded = quality.excluded_geos
    weekly = report_data.weekly_quality(design.data, excluded)
    if not weekly.empty:
      gap_geos = sorted(set(weekly.loc[weekly['zero'], 'location']))
      strong = weekly[weekly['z'].abs() > 3]
      gaps = design_charts.tracking_gaps_chart(weekly, names)
      blocks += [
          _chart(gaps, charts.Framing(
              'Are there tracking gaps?',
              reading_guide=report_text.TRACKING_GAPS_GUIDE,
              takeaway=text.tracking_gaps_takeaway(
                  display(gap_geos), sum(g in excluded for g in gap_geos)
              ),
          )),
          _chart(design_charts.weekly_z_chart(weekly, names), charts.Framing(
              report_text.UNUSUAL_WEEKS_TITLE,
              reading_guide=report_text.UNUSUAL_WEEKS_GUIDE,
              takeaway=text.unusual_weeks_takeaway(
                  len(strong),
                  display(strong['location'].value_counts().index.tolist()),
              ),
          )),
      ]
    if quality.no_conversion_geos:
      # Rerunning with this keeps the user's exclusions and adds the
      # suggested geos; flagged dates are dropped automatically.
      blocks.append(formatter.TemplateSpec('snippet.html.jinja', {
          'intro': report_text.SNIPPET_INTRO,
          'code': text.constraints_snippet(
              sorted({*quality.user_geos, *quality.no_conversion_geos}),
              quality.user_dates,
          ),
          'copy_label': 'Copy geox.Constraints()',
      }))
    rows = text.quality_rows(
        [(f'{names(g)} ({g})', g in quality.spend_no_conversion)
         for g in quality.flagged_geos],
        quality.flagged_dates,
        [f'{names(g)} ({g})' for g in quality.no_conversion_geos],
    )
    if rows:
      blocks.append(formatter.TableSpec(
          id='',
          title='Flagged geos and dates',
          column_headers=['Type', 'Geo or date', 'Finding', 'Status'],
          row_values=rows,
      ))
    blocks.append(formatter.TextSpec(
        report_text.QUALITY_NOTE,
        detail=True,
        links=(('Data quality checks', report_text.DOCS_DATA_QUALITY),),
    ))
    return formatter.create_card_html(
        self._env,
        formatter.CardSpec(
            'quality', 'Pretest data quality', audience='technical'
        ),
        [b for b in blocks if b is not None],
    )


def _test_days(design: api.Design) -> int:
  config = design.design_config
  return config.experiment_duration.days if config else 0


class AnalysisSummarizer(_Summarizer):
  """Builds the interactive analysis readout for an `AnalysisResult`."""

  def __init__(
      self,
      analysis_result: api.AnalysisResult,
      geo_metadata: Optional[pd.DataFrame] = None,
      map_config: maps.MapConfig = maps.MapConfig(),
      data: Optional[pd.DataFrame] = None,
      response_is_value: bool = False,
      currency_symbol: str = '',
  ):
    """Initializes the summarizer.

    Args:
      analysis_result: Output of `geox.analyze(...)`.
      geo_metadata: Optional display-name table.
      map_config: Boundary polygon configuration for the regions map.
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
    super().__init__(geo_metadata, map_config)
    self._result = analysis_result
    self._data = data
    self._value = response_is_value
    # The target iCPA is the design's `cost_per_incremental_conversion`.
    self._text = report_text.AnalysisText(
        response_is_value,
        currency_symbol,
        report_data.default_goals(
            analysis_result.analysis_config.design, report_data.DesignGoals()
        ).target_cpic,
    )

  output_analysis_summary = _Summarizer._write

  # ------------------------------------------------------------------------

  def _build_report(self) -> str:
    result = self._result
    design = result.analysis_config.design
    window = report_data.analysis_window(result)
    level = report_data.confidence_level(result)
    readouts = report_data.cell_readouts(
        result, self._data, self._names, self._value
    )
    experiment_type = report_data.single_experiment_type(design)
    n_treatment = sum(len(d.treatment_geos) for d in design.designs.values())
    cards = [
        self._executive_card(readouts, window, level),
        self._details_card(readouts, level),
    ]
    return self._page(
        'GeoX experiment analysis',
        cards,
        kind='analysis',
        chips=self._chips(
            readouts, design, window, level, experiment_type, n_treatment
        ),
        subtitle=self._objective(readouts, design, window),
    )

  def _objective(
      self,
      readouts: Sequence[report_data.CellReadout],
      design: api.Design,
      window: report_data.AnalysisWindow,
  ) -> str:
    """Test objective shown above the Executive summary (UX report 1a)."""
    return report_text.test_objective(
        [
            (r.label, r.check.experiment_type,
             len(design.designs[r.cell].treatment_geos))
            for r in readouts
        ],
        len(design.control_geos),
        window.planned_days,
        self._value,
    )

  def _chips(
      self,
      readouts: Sequence[report_data.CellReadout],
      design: api.Design,
      window: report_data.AnalysisWindow,
      level: float,
      experiment_type: Optional[api.ExperimentType],
      n_treatment: int,
  ) -> list[formatter.ChipSpec]:
    """Header chips: status, region, test window, split, type, alpha."""
    n_control = len(design.control_geos)
    n_excluded = len(design.excluded_geos or ())
    status = (
        'Test window complete'
        if window.complete
        else f'In progress (day {window.test_days} of {window.planned_days})'
    )
    test = report_text.date_range(window.start, window.test_end)
    chips = [
        formatter.ChipSpec('Status', status),
        *self._region_chips(n_treatment + n_control + n_excluded),
        formatter.ChipSpec('Test', f'{test} ({window.test_days} days)'),
    ]
    if window.cooldown_days:
      chips.append(formatter.ChipSpec(
          'Cooldown', report_text.date_range(window.cooldown_start, window.end)
      ))
    chips.append(formatter.ChipSpec(
        'Regions split',
        f'{n_treatment} treatment, {n_control} control, {n_excluded} excluded',
    ))
    chips += _type_chips(experiment_type, [
        (f'Treatment {i}', r.check.experiment_type)
        for i, r in enumerate(readouts, start=1)
    ])
    chips.append(
        formatter.ChipSpec('Alpha', f'{1 - level:g} · {level:.0%} CIs')
    )
    return chips

  def _banner(
      self,
      readouts: Sequence[report_data.CellReadout],
      overall: report_text.Verdict,
      window: report_data.AnalysisWindow,
      level: float,
  ) -> formatter.CalloutSpec:
    """Verdict banner: business headline, one-line result and next step."""
    text = self._text
    lead = ''
    if len(readouts) > 1:
      title = report_text.MULTI_TITLE_PREFIX.get(overall, '') + ' · '.join(
          f'{r.label}: ' + text.short_result(
              r.metrics.lift.point_estimate, r.efficiency_value,
              r.significant, r.significant_decrease, r.verdict,
          )
          for r in readouts
      )
    else:
      r = readouts[0]
      title = text.headline(
          overall, r.metrics.lift.point_estimate, r.efficiency_value,
          r.significant_decrease,
      )
      if r.significant:
        lead = (
            text.cost_sentence(r.efficiency.point if r.efficiency else None)
            + ' '
            + text.lift_sentence(
                r.check.experiment_type, r.metrics.lift.point_estimate
            )
        ).strip()
      elif not r.significant_decrease:
        lead = text.no_lift_sentence(r.metrics.lift.upper_bound, level)
    body = text.banner_text(
        overall,
        lead,
        window.test_days,
        window.planned_days,
        [r.label for r in readouts if not r.check.ok],
    )
    return formatter.CalloutSpec(
        body,
        title=title,
        tone=report_text.RESULT_TONE[overall],
        links=[('Learn more', report_text.DOCS_INTRO_TO_ANALYSIS)],
    )

  def _meaning(
      self, r: report_data.CellReadout, window: report_data.AnalysisWindow,
      level: float,
  ) -> list[str]:
    cooldown = None
    if r.split is not None and window.cooldown_days:
      cooldown = (
          r.split.test_effect,
          r.split.cooldown_effect,
          window.planned_days,
          r.split.cooldown_days,
          report_text.date_range(window.cooldown_start, window.end),
      )
    return self._text.meaning(
        r.verdict, r.check.experiment_type, r.metrics.lift, r.significant,
        r.significant_decrease, level, window.observed_days,
        r.efficiency.point if r.efficiency else None, cooldown,
    )

  def _execution_sentence(self, r: report_data.CellReadout) -> str:
    check = r.check
    return self._text.execution_sentence(
        check.experiment_type, check.has_spend, check.observed_change,
        check.planned_change, check.verified, check.issue_geos, check.status,
    )

  def _execution_callout(
      self, readouts: Sequence[report_data.CellReadout]
  ) -> formatter.CalloutSpec:
    """'Was the test run as planned?' with one line per cell."""
    multi = len(readouts) > 1
    icons = {True: '✓', False: '⚠', None: 'ℹ'}
    lines = [
        (f'{r.label}: ' if multi else '')
        + f'{icons[r.check.status]} {self._execution_sentence(r)}'
        for r in readouts
    ]
    title, tone = self._text.execution_title(
        [r.check.status for r in readouts],
        any(r.check.has_spend for r in readouts),
    )
    return formatter.CalloutSpec(' · '.join(lines), title=title, tone=tone)

  def _tiles(
      self,
      r: report_data.CellReadout,
      level: float,
      multi: bool,
      window: report_data.AnalysisWindow,
  ) -> formatter.StatsSectionSpec:
    """Verdict, extra conversions, lift, iCPA and implied spend tiles."""
    text = self._text
    prefix = f'{r.label}: ' if multi else ''
    lift = r.metrics.lift
    tiles = [
        formatter.StatsSpec(
            report_text.VERDICT_QUESTION[multi].format(
                label=r.label, kpi=text.kpi
            ),
            r.verdict.value,
            detail=text.verdict_detail(
                r.verdict, r.significant, r.significant_decrease, r.check.ok,
                window.test_days,
                window.planned_days,
            ),
            tone=report_text.RESULT_TONE[r.verdict],
        ),
        formatter.StatsSpec(
            prefix + f'Extra {text.kpi}',
            text.count(lift.point_estimate),
            ci=report_text.CI_LINE.format(
                level=level, lower=text.count(lift.lower_bound),
                upper=text.count(lift.upper_bound),
            ),
            detail=text.lift_detail(
                r.significant, r.significant_decrease, window.observed_days
            ),
        ),
    ]
    eff = r.efficiency
    if eff is not None and r.significant:
      tiles.append(formatter.StatsSpec(
          prefix + text.efficiency_tile,
          text.efficiency(eff.point),
          ci=report_text.CI_LINE.format(
              level=level, lower=text.efficiency(eff.lower),
              upper=text.efficiency(eff.upper),
          ),
          detail=text.efficiency_detail(eff.point),
          tone=_goal_tone(text.meets_target(eff.point)),
      ))
    elif eff is not None:
      tiles.append(formatter.StatsSpec(
          prefix + text.efficiency_tile, '—',
          detail='needs a significant lift',
      ))
    if r.spend is not None:
      terms = report_text.spend_terms(r.check.experiment_type)
      tiles.append(formatter.StatsSpec(
          prefix + f'{terms.label} (implied)',
          text.short(abs(r.spend)),
          detail=report_text.IMPLIED_SPEND_DETAIL,
      ))
    return formatter.StatsSectionSpec(tiles)

  def _executive_card(
      self,
      readouts: Sequence[report_data.CellReadout],
      window: report_data.AnalysisWindow,
      level: float,
  ) -> str:
    text = self._text
    multi = len(readouts) > 1
    blocks: list[Optional[formatter.Spec]] = [
        self._banner(
            readouts, report_data.overall_verdict(readouts), window, level
        ),
        *(self._tiles(r, level, multi, window) for r in readouts),
    ]
    if not window.complete:
      blocks.append(formatter.CalloutSpec(
          report_text.INCOMPLETE_NOTE.format(
              window.test_days, window.planned_days, window.planned_end
          ),
          tone='info',
      ))

    def per_cell(
        texts: Sequence[tuple[report_data.CellReadout, str]]
    ) -> list[Union[str, tuple[str, str]]]:
      return [(r.label, t) if multi else t for r, t in texts]

    meaning = [
        (r, p) for r in readouts for p in self._meaning(r, window, level)
    ]
    steps = [
        (r, text.next_step_text(r.verdict, r.significant, window.planned_end))
        for r in readouts
    ]
    blocks.append(formatter.CalloutSpec(text.formula, tone='info'))
    blocks.append(self._execution_callout(readouts))
    # Analyst detail: the narrative and next steps per cell (the banner
    # already carries the headline and the suggested next step).
    blocks.append(formatter.RowSpec(
        [
            _section('What this means', per_cell(meaning)),
            _section('Next steps', per_cell(steps)),
        ],
        audience='technical',
    ))
    labels = {r.cell: r.label for r in readouts}
    # One cell per row: side by side, two cells' totals did not fit.
    for r in readouts:
      blocks.append(_chart(
          analysis_charts.effect_totals_chart(
              self._result, labels, cell=r.cell
          ),
          charts.Framing(
              report_text.TOTALS_TITLE.format(text.kpi)
              + (f' · {r.label}' if multi else ''),
              reading_guide=text.totals_guide,
              takeaway=text.totals_takeaway(
                  r.check.experiment_type, r.observed_total,
                  r.counterfactual_total, r.metrics.lift, level,
                  r.significant, r.significant_decrease,
              ),
          ),
          audience='technical',
      ))
    columns, rows = report_data.estimate_rows(readouts, text)
    blocks.append(formatter.TableSpec(
        id='',
        title=f'Estimates ({level:.0%} CI in brackets)',
        column_headers=columns,
        row_values=rows,
        audience='technical',
    ))
    blocks.append(formatter.ButtonsSpec(
        [formatter.Button(
            'Download CSV',
            formatter.to_csv([dict(zip(columns, row)) for row in rows],
                             columns),
            'geox_estimates.csv',
        )],
        audience='technical',
    ))
    return formatter.create_card_html(
        self._env,
        formatter.CardSpec('executive', 'Executive summary'),
        [b for b in blocks if b is not None],
    )

  def _details_card(
      self, readouts: Sequence[report_data.CellReadout], level: float
  ) -> str:
    text = self._text
    multi = len(readouts) > 1
    blocks: list[formatter.Spec] = [
        formatter.TextSpec(text.details_intro(level))
    ]
    for r in readouts:
      metrics = r.metrics
      strip = text.equation_strip(
          f'{r.label}: ' if multi else '', r.check.experiment_type,
          r.observed_total, r.counterfactual_total, metrics.lift, level,
      )
      blocks.append(_section('', [strip], 'geox-equation'))
      row: list[Optional[formatter.Spec]] = []
      eff = r.efficiency
      if eff is not None and r.significant and metrics.icpd is not None:
        transform: Optional[Callable[[float], float]] = (
            None if self._value else _reciprocal
        )
        row.append(_chart(
            analysis_charts.confidence_curve(
                metrics.icpd,
                eff.label,
                ',.2f',
                transform=transform,
                level=level,
                decreasing=transform is not None,
            ),
            charts.Framing(
                f'{r.label} · {eff.label} confidence',
                reading_guide=report_text.CONFIDENCE_GUIDE,
                takeaway=text.efficiency_confidence_takeaway(
                    eff.label, eff.point, eff.lower, eff.upper, level
                ),
            ),
        ))
      row.append(_chart(
          analysis_charts.confidence_curve(
              metrics.lift, f'Incremental {text.kpi}', ',.0f', level=level
          ),
          charts.Framing(
              f'{r.label} · incremental {text.kpi} confidence',
              reading_guide=text.lift_confidence_guide(level),
              takeaway=text.lift_confidence_takeaway(metrics.lift, level),
          ),
      ))
      blocks.append(formatter.RowSpec([b for b in row if b is not None]))
    return formatter.create_card_html(
        self._env,
        formatter.CardSpec('detailed', 'Detailed results', audience='technical'),
        blocks,
    )
