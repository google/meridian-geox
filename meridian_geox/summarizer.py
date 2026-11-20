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
import math
from typing import Optional, Sequence

from meridian_geox import api
from meridian_geox import report_data
from meridian_geox import report_text
from meridian_geox.templates import formatter

# Maximum number of candidates with their own detail panels.
MAX_SCOPED_CANDIDATES = 10


def _goal_tone(value: Optional[bool]) -> formatter.Tone:
  if value is None:
    return 'neutral'
  return 'success' if value else 'warning'


# Tiles shown only in the Full report.
_ANALYST_TILES = ('Pretest fit', 'Pre-test AA check')

_CHANCE_TONES: dict[str, formatter.Tone] = {
    'High': 'success',
    'Medium': 'warning',
    'Low': 'error',
}


def _chance_tone(chance: Optional[float]) -> formatter.Tone:
  return _CHANCE_TONES.get(report_data.chance_label(chance), 'neutral')


class _Summarizer:
  """Shared plumbing of the report summarizers."""

  _currency: str = ''

  def _build_report(self) -> str:
    raise NotImplementedError

  def _money(self, value: Optional[float], digits: int = 0) -> str:
    return report_text.format_money(value, self._currency, digits)

  def _short(self, value: Optional[float]) -> str:
    return report_text.short_money(value, self._currency)

  def generate_html(self) -> str:
    """Returns the self-contained HTML report."""
    return self._build_report()

  def to_iframe(self, height: int = 1100) -> str:
    """Returns the report in an iframe for display(HTML(...)) in notebooks."""
    return formatter.iframe(self.generate_html(), height)

  def _repr_html_(self) -> str:
    return self.to_iframe()


@dataclasses.dataclass(frozen=True)
class _DesignContext:
  designs: dict[str, api.Design]
  summaries: list[report_data.CandidateSummary]
  scoped: list[report_data.CandidateSummary]
  goals: report_data.DesignGoals
  terms: report_text.SpendTerms
  n_feasible: int


class DesignSummarizer(_Summarizer):
  """Builds the interactive design 2-pager for a `DesignSet`."""

  def __init__(
      self,
      design_set: api.DesignSet,
      currency_symbol: str = '',
      target_mde: Optional[float] = None,
  ):
    """Initializes the summarizer.

    Args:
      design_set: Output of `geox.run_design(...)`.
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
    self._currency = currency_symbol
    self._text = report_text.DesignText(currency_symbol)

  def output_design_summary(self, filename: str, filepath: str) -> str:
    """Writes the report to `<filepath>/<filename>` and returns the path."""
    return formatter.write_html(self.generate_html(), filename, filepath)

  # ------------------------------------------------------------------------

  def _build_report(self) -> str:
    self._env = formatter.create_template_env()
    designs = {str(k): v for k, v in self._design_set.designs.items()}
    first = next(iter(designs.values()))
    goals = report_data.default_goals(first, self._goals)
    summaries = report_data.design_summaries(self._design_set, goals)
    if not summaries:
      raise ValueError('No candidate designs could be summarized.')
    best = summaries[0]
    best_design = designs[best.design_id]
    experiment_type = report_data.single_experiment_type(best_design)
    geo_set = set()
    for d in designs.values():
      geo_set.update(str(g) for g in d.control_geos)
      geo_set.update(str(g) for g in d.excluded_geos or ())
      for c in d.designs.values():
        geo_set.update(str(g) for g in c.treatment_geos)
    all_geos = sorted(geo_set)
    context = _DesignContext(
        designs=designs,
        summaries=summaries,
        scoped=summaries[:MAX_SCOPED_CANDIDATES],
        goals=goals,
        terms=report_text.spend_terms(experiment_type),
        n_feasible=sum(s.feasible for s in summaries),
    )
    cards = [
        self._recommended_card(context),
    ]
    return formatter.create_summary_html(
        self._env,
        'GeoX experiment design',
        cards,
        subtitle=report_text.DESIGN_REPORT_SUBTITLE,
        kind='design',
        chips=self._chips(best_design, experiment_type, goals, all_geos),
        default_candidate=best.design_id,
    )

  def _chips(
      self,
      design: api.Design,
      experiment_type: Optional[api.ExperimentType],
      goals: report_data.DesignGoals,
      all_geos: Sequence[str],
  ) -> list[formatter.ChipSpec]:
    """Header chips: region, test type, pretest window and goals."""
    region = 'Geos'
    chips = [formatter.ChipSpec('Region', f'{region} ({len(all_geos)} geos)')]
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
    if experiment_type is None:
      types = ', '.join(
          f'{report_text.short_label(c.label)}'
          f' {report_text.EXPERIMENT_TYPE_LABELS[c.experiment_type].lower()}'
          for c in report_data.spend_changes(design)
      )
      chips.append(formatter.ChipSpec('Types', types))
      chips.append(formatter.ChipSpec('Cells', str(len(design.designs))))
    else:
      chips.append(formatter.ChipSpec(
          'Type',
          report_text.EXPERIMENT_TYPE_LABELS[experiment_type]
          + ' · '
          + report_text.EXPERIMENT_TYPE_EXPLANATIONS[experiment_type],
      ))
    if config is not None:
      method = config.methodology.name
      rule = config.geo_assignment_rule.name.replace('_', ' ').title()
      chips.append(formatter.ChipSpec('Method', f'{method} · {rule}'))
    goal_parts = []
    if goals.target_mde is not None:
      goal_parts.append(f'MDE ≤ {report_text.format_percent(goals.target_mde)}')
    if goals.target_budget is not None:
      goal_parts.append(f'budget ≤ {self._short(goals.target_budget)}')
    if goals.target_cpic is not None:
      goal_parts.append('target iCPA ' + self._money(goals.target_cpic, 2))
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
    alpha, sides = report_data.alpha_and_sides(design)
    rows = [
        (c.label, c.experiment_type, c.amount, c.planned_change)
        for c in report_data.spend_changes(design)
    ]
    parts = [
        text.detection_sentence(
            s.max_mde, plan.power if plan else 0.8, alpha, sides
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
    spend_value = self._short(s.total_spend)
    spend_detail = None
    if s.meets_budget is not None and goals.target_budget is not None:
      spend_detail = text.budget_detail(s.total_spend, goals.target_budget)
    spend_formula = None
    changes = report_data.spend_changes(design)
    if len(changes) > 1:
      spend_formula = text.cell_changes([
          (c.label, c.experiment_type, c.amount, c.planned_change)
          for c in changes
      ])
    elif changes and changes[0].experiment_type in (
        api.ExperimentType.HEAVY_UP, api.ExperimentType.GO_DARK
    ):
      change = changes[0]
      pct = report_text.format_percent(abs(change.planned_change or 0), 0)
      if change.experiment_type == api.ExperimentType.HEAVY_UP:
        spend_value = '+' + self._short(abs(change.amount))
      spend_detail = f'{pct} of treatment BAU spend'
      bau = report_data.bau_spend(change)
      if bau is not None:
        spend_formula = text.bau_formula(bau, days, pct)
    elif plan is not None and len(s.cells) == 1:
      spend_formula = text.mde_spend_formula(plan.mde_abs, plan.cpic)
    tiles.append(formatter.StatsSpec(
        context.terms.label,
        spend_value,
        detail=spend_detail,
        ci=spend_formula,
        tone=_goal_tone(s.meets_budget),
    ))
    mde_abs = sum(
        c.mde_abs for c in s.cells.values() if not math.isnan(c.mde_abs)
    )
    tiles.append(formatter.StatsSpec(
        'Minimum detectable effect',
        report_text.format_percent(s.max_mde),
        detail=text.mde_detail(goals.target_mde, s.meets_mde),
        ci=text.mde_formula(
            [(c.label, c.mde) for c in s.cells.values()], mde_abs, days
        ),
        tone=_goal_tone(s.meets_mde),
    ))
    tiles.append(formatter.StatsSpec(
        'Pretest fit',
        '✓ Good fit' if checks.good_fit else '⚠ Weak fit',
        detail=(
            report_text.CHECK_GOOD_FIT
            if checks.good_fit
            else report_text.CHECK_WEAK_FIT
        ),
        ci=text.fit_detail(checks.r2, checks.min_r2),
        tone='success' if checks.good_fit else 'warning',
    ))
    tiles.append(formatter.StatsSpec(
        'Pre-test AA check',
        '✓ Passed' if checks.aa_passed else '✗ Failed',
        detail=text.aa_text(checks.aa_passed, checks.aa_window),
        ci=text.aa_detail(checks.aa_p_value, checks.alpha),
        tone='success' if checks.aa_passed else 'error',
    ))
    n_treatment = sum(c.treatment_geo_count for c in s.cells.values())
    n_excluded = len(design.excluded_geos or ())
    tiles.append(formatter.StatsSpec(
        'Regions split',
        f'{n_treatment} / {s.control_geo_count} / {n_excluded}',
        detail='treatment / control / excluded',
        ci=text.split_detail(
            checks.treatment_share,
            [(c.label, c.treatment_geo_count) for c in s.cells.values()],
        ),
    ))
    return tiles

  def _recommended_card(self, context: _DesignContext) -> str:
    """The Recommended design card."""
    text = self._text
    blocks: list[formatter.Spec] = []
    for s in context.scoped:
      design = context.designs[s.design_id]
      plan = report_data.plan_numbers(s, design, context.goals)
      if s.rank != 1:
        best = context.summaries[0]
        blocks.append(formatter.CalloutSpec(
            text.inspecting_note(s.name, best.name),
            title=f'Inspecting {s.name}',
            tone='warning',
            candidate=s.design_id,
        ))
      blocks.append(formatter.TemplateSpec(
          'option_banner.html.jinja',
          {
              'warn': plan is not None and plan.chance < 0.8,
              'name': s.name + text.chance_title(
                  report_data.chance_label(plan.chance) if plan else None
              ),
              'narrative': self._narrative(s, design, context, plan),
              'learn_more_url': report_text.DOCS_INTRO_TO_DESIGN,
          },
          candidate=s.design_id,
      ))
      tiles = self._design_tiles(
          s, design, context, plan, report_data.design_checks(s, design)
      )
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


def _test_days(design: api.Design) -> int:
  config = design.design_config
  return config.experiment_duration.days if config else 0
