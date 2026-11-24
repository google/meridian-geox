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
from typing import NamedTuple, Optional, Sequence

import altair as alt
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


def _goal_tone(value: Optional[bool]) -> formatter.Tone:
  if value is None:
    return 'neutral'
  return 'success' if value else 'warning'


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
    design: api.Design,
    table: pd.DataFrame,
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
  scale = charts.group_scale(list(report_data.group_labels(design).values()))
  chart = maps.plot_geo_assignment(
      table, map_config, scale, title='', show_legend=show_legend
  )
  block = _chart(chart, framing, candidate=candidate)
  assert block is not None
  return block


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


@dataclasses.dataclass(frozen=True)
class _DesignContext:
  designs: dict[str, api.Design]
  summaries: list[report_data.CandidateSummary]
  scoped: list[_Candidate]
  goals: report_data.DesignGoals
  terms: report_text.SpendTerms
  coverage: maps.BoundaryCoverage
  n_feasible: int


class DesignSummarizer(_Summarizer):
  """Builds the interactive design 2-pager for a `DesignSet`."""

  def __init__(
      self,
      design_set: api.DesignSet,
      geo_metadata: Optional[pd.DataFrame] = None,
      map_config: maps.MapConfig = maps.MapConfig(),
      currency_symbol: str = '',
      target_mde: Optional[float] = None,
  ):
    """Initializes the summarizer.

    Args:
      design_set: Output of `geox.run_design(...)`.
      geo_metadata: Display names: `location` plus `geo_name` or `name`.
      map_config: Boundary polygons for the map panels.
      currency_symbol: Symbol prefixed to monetary values.
      target_mde: Largest acceptable relative MDE, for example 0.05.

    Raises:
      ValueError: On invalid inputs.
    """
    if design_set.design_metrics is None or design_set.design_metrics.empty:
      raise ValueError('design_set must have non-empty design_metrics.')
    if not design_set.designs:
      raise ValueError('design_set must contain at least one design.')
    self._goals = report_data.DesignGoals(target_mde=target_mde)
    self._design_set = design_set
    super().__init__(geo_metadata, map_config)
    self._text = report_text.DesignText(currency_symbol)

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
    )
    # UX order: recommendation, then the comparison and its evidence, then
    # the launch plan and next steps, then details.
    cards = [
        self._recommended_card(context),
        self._visuals_card(context),
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
    design = self._design_set.designs[s.design_id]
    return _Candidate(
        s,
        design,
        report_data.plan_numbers(s, design, goals),
        report_data.geo_table(design, self._names),
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

  def _recommended_card(self, context: _DesignContext) -> str:
    """The Recommended design card."""
    s, design, plan, _ = context.scoped[0]
    tiles = self._design_tiles(
        s, design, context, plan, report_data.design_checks(s, design)
    )
    blocks: list[formatter.Spec] = [
        formatter.TemplateSpec(
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
        ),
        formatter.StatsSectionSpec(
            [t for t in tiles if t.title not in _ANALYST_TILES]
        ),
        formatter.StatsSectionSpec(
            [t for t in tiles if t.title in _ANALYST_TILES],
            audience='technical',
        ),
    ]
    return formatter.create_card_html(
        self._env,
        formatter.CardSpec(
            'recommended',
            'Recommended design',
            subtitle=report_text.RECOMMENDED_SUBTITLE,
        ),
        blocks,
    )

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

  def _visuals_card(self, context: _DesignContext) -> str:
    text = self._text
    blocks: list[formatter.Spec] = [
        formatter.CalloutSpec(report_text.VISUALS_INTRO, tone='neutral')
    ]
    for c in context.scoped:
      s, design, table = c.summary, c.design, c.table
      map_block = _map_block(
          design,
          table,
          self._map_config,
          charts.Framing(
              'Where would the test run?',
              reading_guide=report_text.MAP_GUIDE,
              takeaway=self._regions_sentence(s, table),
          ),
          candidate=s.design_id,
          # The group list next to the map doubles as its legend.
          show_legend=False,
      )
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


def _test_days(design: api.Design) -> int:
  config = design.design_config
  return config.experiment_duration.days if config else 0
