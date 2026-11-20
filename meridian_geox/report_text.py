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

"""Narrative copy shared by the GeoX design and analysis reports.

Keeping all user-facing wording here makes the reports consistent and lets the
wording be reviewed and tested in one place.
"""

import dataclasses
import datetime
import math
from typing import Optional, Sequence

from meridian_geox import api


def option_name(rank: int) -> str:
  """Returns the display name of the candidate at the given 1-based rank."""
  if rank < 1:
    raise ValueError(f'rank must be >= 1, got {rank}.')
  return f'Option {rank}' + (' (Recommended)' if rank == 1 else '')


@dataclasses.dataclass(frozen=True)
class SpendTerms:
  """Spend vocabulary for an experiment type.

  Attributes:
    label: Name of the spend amount (for example "Campaign spend").
    treatment_action: Short description of what happens in treatment geos.
    type_label: Name of the experiment type (for example "Holdback").
    explanation: Plain-language description of the treatment geos.
  """

  label: str
  treatment_action: str
  type_label: str = ''
  explanation: str = ''


_SPEND_TERMS = {
    api.ExperimentType.HOLDBACK: SpendTerms(
        label='Campaign spend',
        treatment_action='campaign on',
        type_label='Holdback',
        explanation='ads show only in treatment geos',
    ),
    api.ExperimentType.HEAVY_UP: SpendTerms(
        label='Extra spend',
        treatment_action='spend increased',
        type_label='Heavy-up',
        explanation='extra spend in treatment geos',
    ),
    api.ExperimentType.GO_DARK: SpendTerms(
        label='Paused spend',
        treatment_action='ads paused',
        type_label='Go dark',
        explanation='ads paused in treatment geos',
    ),
}
_MULTI_CELL_TERMS = SpendTerms(
    label='Incremental spend',
    treatment_action='spend changed',
)


def spend_terms(experiment_type: Optional[api.ExperimentType]) -> SpendTerms:
  """Returns spend vocabulary; `None` means a multi-cell experiment."""
  if experiment_type is None:
    return _MULTI_CELL_TERMS
  return _SPEND_TERMS[experiment_type]


# Public GeoX documentation linked from the reports.
_DOCS = 'https://developers.google.com/meridian/geox'
DOCS_INTRO_TO_DESIGN = f'{_DOCS}/intro-to-design'

CHANCE_LEGEND = 'High ≥ 80% · Medium 50–80% · Low < 50%'


def _non_finite(value: Optional[float]) -> str:
  """'n/a' for a missing or NaN value, '∞' or '-∞' for an infinite one."""
  if value is None or math.isnan(value):
    return 'n/a'
  return '∞' if value > 0 else '-∞'


def short_money(value: Optional[float], currency_symbol: str = '') -> str:
  """Formats money compactly: `$8.7k`, `$1.2M`, `$950`."""
  if value is None or not math.isfinite(value):
    return _non_finite(value)
  sign = '-' if value < 0 else ''
  amount = abs(value)
  if amount >= 1e6:
    text = f'{amount / 1e6:.1f}M'
  elif amount >= 1e3:
    text = f'{amount / 1e3:.1f}k'
  else:
    text = f'{amount:,.0f}'
  return f'{sign}{currency_symbol}{text}'


def date_range(start, end) -> str:
  """Formats a date range like `Jan 1 – Mar 31, 2020`."""
  if start.year == end.year:
    return f'{start:%b} {start.day} – {end:%b} {end.day}, {end.year}'
  return (
      f'{start:%b} {start.day}, {start.year} – {end:%b} {end.day}, {end.year}'
  )


def cell_index(cell: str) -> int:
  """Returns the 1-based index of a cell name such as `cell_2`."""
  try:
    return int(str(cell).rsplit('_', 1)[-1])
  except ValueError:
    return 1


def treatment_label(
    cell: str, experiment_type: api.ExperimentType, multi_cell: bool
) -> str:
  """Returns the legend label of a treatment group."""
  if not multi_cell:
    return 'Treatment'
  action = spend_terms(experiment_type).treatment_action
  return f'Treatment {cell_index(cell)} · {action}'


def format_number(value: Optional[float], digits: int = 0) -> str:
  """Formats a number with thousands separators."""
  if value is None or not math.isfinite(value):
    return _non_finite(value)
  return f'{value:,.{digits}f}'


def format_money(
    value: Optional[float], currency_symbol: str = '', digits: int = 0
) -> str:
  """Formats a monetary value with an optional currency symbol."""
  if value is None or not math.isfinite(value):
    return _non_finite(value)
  sign = '-' if value < 0 else ''
  return f'{sign}{currency_symbol}{abs(value):,.{digits}f}'


def format_percent(
    value: Optional[float], digits: int = 1, signed: bool = False
) -> str:
  """Formats a fraction (0.05) as a percentage (5.0%)."""
  if value is None or not math.isfinite(value):
    return _non_finite(value)
  text = f'{value * 100:,.{digits}f}%'
  if signed and value > 0:
    text = '+' + text
  return text


def change_percent(
    planned_change: Optional[float], signed: bool = False
) -> str:
  """Planned spend change as a whole percent ('34%', or '+34%'), or ''."""
  if planned_change is None:
    return ''
  value = planned_change if signed else abs(planned_change)
  return format_percent(value, 0, signed)


def short_label(label: str) -> str:
  """'Treatment 1 · ads paused' -> 'Treatment 1'."""
  return label.split(' · ')[0]


@dataclasses.dataclass(frozen=True)
class DesignText:
  """Number formats and sentences of the design report.

  Methods take plain values, so the summarizer only wires report_data
  numbers into them.

  Attributes:
    currency_symbol: Symbol prefixed to monetary values.
  """

  currency_symbol: str = ''

  def money(self, value: Optional[float], digits: int = 0) -> str:
    """Formats a monetary value, e.g. '$1,200'."""
    return format_money(value, self.currency_symbol, digits)

  def short(self, value: Optional[float]) -> str:
    """Compact monetary value, e.g. '$1.2k'."""
    return short_money(value, self.currency_symbol)

  # ------------------------------------------------------------- recommend

  def change_clause(
      self,
      experiment_type: api.ExperimentType,
      amount: float,
      planned_change: Optional[float],
      where: str,
  ) -> str:
    """'adds +€14.1k of extra spend in the treatment geos (+34% …)'."""
    value = self.short(abs(amount))
    pct = change_percent(planned_change)
    if experiment_type == api.ExperimentType.HEAVY_UP:
      share = f' (+{pct} of their BAU spend)' if pct else ''
      return f'adds +{value} of extra spend in {where}{share}'
    if experiment_type == api.ExperimentType.GO_DARK:
      share = f' ({pct} of their BAU spend)' if pct else ''
      return f'pauses {value} of spend in {where}{share}'
    return f'needs about {value} of campaign spend in {where}'

  @staticmethod
  def detection_sentence(
      mde: float, power: float, alpha: float, sides: str
  ) -> str:
    """'It detects a lift of 5.0% or more … 80% of the time (alpha …).'."""
    return (
        f'It detects a lift of {format_percent(mde)} or more in treatment-geo'
        f' conversions {power:.0%} of the time (alpha {alpha:g}, {sides}).'
    )

  def spend_sentence(
      self,
      changes: Sequence[tuple[str, api.ExperimentType, float, Optional[float]]],
      total_spend: float,
      spend_label: str,
      days: int,
  ) -> str:
    """How much spend the candidate needs or changes over the test.

    Args:
      changes: (cell label, experiment type, amount, planned change) per cell.
      total_spend: Total spend of the candidate.
      spend_label: Spend term, e.g. 'Campaign spend'.
      days: Test length in days.

    Returns:
      One sentence.
    """
    if all(et == api.ExperimentType.HOLDBACK for _, et, _, _ in changes):
      return (
          f'It needs about {self.short(total_spend)} of'
          f' {spend_label.lower()} in the treatment geos over the {days}-day'
          ' test.'
      )
    if len(changes) == 1:
      _, et, amount, planned = changes[0]
      clause = self.change_clause(et, amount, planned, 'the treatment geos')
      return f'It {clause} over the {days}-day test.'
    clauses = '; '.join(
        f'{short_label(label)} {self.change_clause(et, amount, planned, "its geos")}'
        for label, et, amount, planned in changes
    )
    return f'Spend over the {days}-day test: {clauses}.'

  def chance_sentence(
      self,
      chance: float,
      cpic: float,
      budget: Optional[float],
      has_holdback: bool,
  ) -> str:
    """Chance of a significant result at the budget goal or target iCPA."""
    value = self.money(cpic, 2)
    if budget is not None and has_holdback:
      return (
          f'Spending your {self.short(budget)} budget goal at an iCPA of'
          f' {value} gives a {chance:.0%} chance of a statistically'
          ' significant result.'
      )
    return (
        f'If each incremental conversion costs your target iCPA of {value},'
        f' the chance of a statistically significant result is {chance:.0%}.'
    )

  @staticmethod
  def goals_sentence(
      any_goal: bool, missed: Sequence[str], n_feasible: int, n: int
  ) -> str:
    """Whether the candidate meets the goals, and how many do."""
    if not any_goal:
      return 'Set goals to rank candidates against them.'
    if missed:
      names = ' and '.join(f'{m} goal' for m in missed)
      return (
          f'This candidate misses your {names}; {n_feasible} of {n}'
          ' candidates meet all of them.'
      )
    return (
        f'This candidate meets your goals; {n_feasible} of {n} candidates meet'
        ' all of them.'
    )

  @staticmethod
  def chance_title(chance_label: Optional[str]) -> str:
    """': high chance of a conclusive result' (or '')."""
    if chance_label is None:
      return ''
    return f': {chance_label.lower()} chance of a conclusive result'

  # ----------------------------------------------------------------- tiles

  @staticmethod
  def chance_detail(
      chance: float, has_budget: bool, has_target_cpic: bool
  ) -> str:
    """Detail line of the Feasibility tile."""
    if has_budget:
      return f'{chance:.0%} chance at your budget goal'
    if has_target_cpic:
      return f'{chance:.0%} chance at your target iCPA'
    return f'{chance:.0%} chance at the planned spend'

  def budget_detail(self, total_spend: float, budget: float) -> str:
    """'within your $10k goal' or '$2k over your $10k goal'."""
    gap = total_spend - budget
    if gap <= 0:
      return f'within your {self.short(budget)} goal'
    return f'{self.short(gap)} over your {self.short(budget)} goal'

  def cell_changes(
      self,
      changes: Sequence[tuple[str, api.ExperimentType, float, Optional[float]]],
  ) -> str:
    """'Treatment 1: −$4k (−100%) · Treatment 2: +$2k (+30%)'."""
    parts = []
    for label, et, amount, planned in changes:
      sign = '−' if amount < 0 or et == api.ExperimentType.GO_DARK else '+'
      pct = change_percent(planned, signed=True)
      pct = f' ({pct})' if pct else ''
      parts.append(f'{short_label(label)}: {sign}{self.short(abs(amount))}{pct}')
    return ' · '.join(parts)

  def bau_formula(self, bau: float, days: int, pct: str) -> str:
    """'= $40k BAU spend in treatment geos over 28 days × 35%'."""
    return f'= {self.short(bau)} BAU spend in treatment geos over {days} days × {pct}'

  def mde_spend_formula(self, mde_abs: float, cpic: float) -> str:
    """'= 1,200 incremental conversions (MDE) × $2.00 iCPA'."""
    return (
        f'= {format_number(mde_abs)} incremental conversions (MDE) ×'
        f' {self.money(cpic, 2)} iCPA'
    )

  def spend_tile(
      self,
      changes: Sequence[tuple[str, api.ExperimentType, float, Optional[float]]],
      total_spend: float,
      budget: Optional[float],
      bau: Optional[float],
      days: int,
      mde_spend: Optional[tuple[float, float]],
  ) -> tuple[str, Optional[str], Optional[str]]:
    """Value, detail and formula lines of the spend tile.

    Args:
      changes: (cell label, experiment type, amount, planned change) per cell.
      total_spend: Total spend of the candidate.
      budget: Budget goal to compare the spend with, if any.
      bau: Treatment BAU spend of a single go-dark or heavy-up cell, if known.
      days: Test length in days.
      mde_spend: (MDE in incremental conversions, iCPA) of a holdback cell.

    Returns:
      The tile value, its detail line and its formula line.
    """
    value, detail, formula = self.short(total_spend), None, None
    if budget is not None:
      detail = self.budget_detail(total_spend, budget)
    if len(changes) > 1:
      formula = self.cell_changes(changes)
    elif changes and changes[0][1] in (
        api.ExperimentType.HEAVY_UP, api.ExperimentType.GO_DARK
    ):
      _, experiment_type, amount, planned = changes[0]
      pct = change_percent(planned)
      if experiment_type == api.ExperimentType.HEAVY_UP:
        value = '+' + self.short(abs(amount))
      detail = f'{pct} of treatment BAU spend'
      if bau is not None:
        formula = self.bau_formula(bau, days, pct)
    elif mde_spend is not None:
      formula = self.mde_spend_formula(*mde_spend)
    return value, detail, formula

  @staticmethod
  def fit_value(good_fit: bool) -> tuple[str, str]:
    """Value and detail line of the Pretest fit tile."""
    if good_fit:
      return '✓ Good fit', CHECK_GOOD_FIT
    return '⚠ Weak fit', CHECK_WEAK_FIT

  @staticmethod
  def aa_value(passed: bool) -> str:
    """Value of the Pre-test AA check tile."""
    return '✓ Passed' if passed else '✗ Failed'

  @staticmethod
  def mde_detail(mde_goal: Optional[float], meets: Optional[bool]) -> str:
    """Detail line of the MDE tile."""
    if mde_goal is None:
      return 'Smallest lift the test reliably detects'
    verb = 'meets' if meets else 'misses'
    return f'{verb} your {format_percent(mde_goal)} goal'

  @staticmethod
  def mde_formula(
      cell_mdes: Sequence[tuple[str, float]], mde_abs: float, days: int
  ) -> str:
    """Per-cell MDEs, or the MDE in incremental conversions."""
    if len(cell_mdes) > 1:
      return ' · '.join(
          f'{short_label(label)}: {format_percent(mde)}'
          for label, mde in cell_mdes
      )
    return f'≈ {format_number(mde_abs)} incremental conversions over {days} days'

  @staticmethod
  def fit_detail(r2: float, min_r2: float) -> str:
    """'R² 0.97; minimum 0.8'."""
    return f'R² {r2:.2f}; minimum {min_r2:g}'

  @staticmethod
  def aa_text(
      passed: bool,
      window: Optional[tuple[datetime.date, datetime.date]],
  ) -> str:
    """A/A check sentence."""
    span = f' over {date_range(*window)}' if window is not None else ''
    if passed:
      return (
          'No pre-existing difference between treatment and control'
          f' geos{span}.'
      )
    return f'Treatment and control geos already differed{span}.'

  @staticmethod
  def aa_detail(p_value: float, alpha: float) -> str:
    """'AA p-value 0.62; passes at ≥ 0.1'."""
    return f'AA p-value {p_value:.2f}; passes at ≥ {alpha:g}'

  @staticmethod
  def split_detail(share: float, cell_counts: Sequence[tuple[str, int]]) -> str:
    """'Treatment geos hold 25.0% of conversions (Treatment 1: 10, …)'."""
    text = f'Treatment geos hold {share:.1%} of conversions'
    if len(cell_counts) > 1:
      text += ' (' + ', '.join(
          f'{short_label(label)}: {n}' for label, n in cell_counts
      ) + ')'
    return text

DESIGN_REPORT_SUBTITLE = (
    'Compare candidate geo splits against your goals, then pick one to '
    'launch.'
)

CHECK_GOOD_FIT = (
    'Control geos predict treatment-geo conversions closely before the test.'
)

CHECK_WEAK_FIT = (
    'Control geos predict treatment geos less closely than required.'
)

RECOMMENDED_SUBTITLE = (
    'Which design to run, why, and whether it meets your goals.'
)
