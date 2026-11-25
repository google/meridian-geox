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
from typing import Literal, Optional, Sequence

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


# What the spend amount means for each experiment type.
SPEND_DEFINITIONS = {
    api.ExperimentType.HOLDBACK: (
        'Holdback: new campaign spend in the treatment geos during the test,'
        ' while ads stay off in control geos. The minimum is the spend that'
        ' buys the minimum detectable effect at your expected cost per'
        ' incremental conversion (MDE × iCPA).'
    ),
    api.ExperimentType.HEAVY_UP: (
        'Heavy-up: spend added on top of business as usual in the treatment'
        ' geos during the test; control geos stay at business as usual.'
    ),
    api.ExperimentType.GO_DARK: (
        'Go dark: business-as-usual spend switched off in the treatment geos'
        ' during the test; control geos keep spending as usual.'
    ),
}

GROUP_DESCRIPTIONS = {
    api.ExperimentType.HOLDBACK: ('ads show', "ads don't show"),
    api.ExperimentType.HEAVY_UP: ('spend increased', 'business as usual'),
    api.ExperimentType.GO_DARK: ('ads paused', 'business as usual'),
}

# Public GeoX documentation linked from the reports.
_DOCS = 'https://developers.google.com/meridian/geox'
DOCS_INTRO_TO_DESIGN = f'{_DOCS}/intro-to-design'
DOCS_EXPERIMENT_TYPES = f'{_DOCS}/types-of-experiments'
DOCS_ICPA_GLOSSARY = f'{_DOCS}/glossary#cost_per_incremental_conversion_cpic'
DOCS_TROUBLESHOOTING = f'{_DOCS}/troubleshooting'
DOCS_DATA_QUALITY = f'{_DOCS}/data-validation-and-quality-checks'

ICPA_TIP = (
    'Setting the target iCPA. Anchor the target iCPA (cost per incremental'
    ' conversion) to evidence: your historical iCPA, prior lift tests (GeoX'
    ' or Conversion Lift) or a marketing mix model (MMM), not an aspirational'
    ' goal. A lower iCPA makes the same spend look more powerful on paper'
    ' only.'
)

CHANCE_LEGEND = 'High ≥ 80% · Medium 50–80% · Low < 50%'


def plural(count: int, noun: str) -> str:
  """Returns `'1 geo'` / `'3 geos'`."""
  word = noun if count == 1 else noun + 's'
  return f'{count:,} {word}'


def name_list(names: list[str], limit: int = 3) -> str:
  """Returns `'A, B, C and 4 more'`."""
  if not names:
    return ''
  shown = ', '.join(names[:limit])
  rest = len(names) - limit
  return f'{shown} and {rest} more' if rest > 0 else shown


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


def short_date(value, with_year: bool = True) -> str:
  """Formats a timestamp like `Wed, Apr 1, 2020` (or `Wed, Apr 1`)."""
  text = f'{value:%a, %b} {value.day}'
  return f'{text}, {value.year}' if with_year else text


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


def format_signed(value: Optional[float], digits: int = 0) -> str:
  """Formats a number with an explicit sign."""
  text = format_number(value, digits)
  if value is not None and not math.isnan(value) and value > 0:
    return '+' + text
  return text


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


def date_list(dates: Sequence[datetime.date], empty: str = '') -> str:
  """'2024-01-01, 2024-01-08' (or `empty`)."""
  return ', '.join(f'{d:%Y-%m-%d}' for d in dates) or empty


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


def schedule_span(schedule, ramp_up: bool = False) -> str:
  """Test window and cooldown of a `report_data.ExperimentSchedule`."""
  s = schedule
  if s.dated:
    ramp = ramp_up and s.ramp_up_days
    start, end = s.test_start, s.test_end
    first = short_date(start, with_year=start.year != end.year)
    text = f'from {first} to {short_date(end)} ({s.test_days} days'
    text += f', the first {ramp} a ramp-up)' if ramp else ')'
  else:
    text = f'for {s.test_days} days'
  if s.cooldown_days:
    text += f', then a {s.cooldown_days}-day cooldown'
    text += f' to {short_date(s.cooldown_end)}' if s.dated else ''
  return text


def calendar_takeaway(
    schedule, pretest: Optional[tuple[datetime.date, datetime.date]]
) -> Optional[str]:
  """'Test: … (28 days); cooldown: … (7 days). Pretest data: …', if dated."""
  s = schedule
  if not s.dated:
    return None
  text = (
      f'Test: {short_date(s.test_start)} – {short_date(s.test_end)}'
      f' ({s.test_days} days)'
  )
  if s.cooldown_days:
    text += (
        f'; cooldown: {short_date(s.cooldown_start)} –'
        f' {short_date(s.cooldown_end)} ({s.cooldown_days} days)'
    )
  return text + (f'. Pretest data: {date_range(*pretest)}.' if pretest else '')


@dataclasses.dataclass(frozen=True)
class Check:
  """One line of the design checks list."""

  status: Literal['pass', 'warn', 'fail', 'info']
  text: str
  detail: Optional[str] = None


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

  def spend_tile(
      self,
      changes: Sequence[tuple[str, api.ExperimentType, float, Optional[float]]],
      total_spend: float,
      budget: Optional[float],
      bau: Optional[float],
      days: int,
      mde_spend: Optional[tuple[float, float]],
  ) -> tuple[str, Optional[str], Optional[str]]:
    """Value, detail and formula lines of the spend tile."""
    value, detail, formula = self.short(total_spend), None, None
    if budget is not None:
      gap = total_spend - budget
      detail = f'{self.short(gap)} over your' if gap > 0 else 'within your'
      detail += f' {self.short(budget)} goal'
    if len(changes) > 1:
      parts = []
      for label, et, amount, planned in changes:
        sign = '−' if amount < 0 or et == api.ExperimentType.GO_DARK else '+'
        pct = change_percent(planned, signed=True)
        pct = f' ({pct})' if pct else ''
        spend = self.short(abs(amount))
        parts.append(f'{short_label(label)}: {sign}{spend}{pct}')
      formula = ' · '.join(parts)
    elif changes and changes[0][1] in (
        api.ExperimentType.HEAVY_UP, api.ExperimentType.GO_DARK
    ):
      _, experiment_type, amount, planned = changes[0]
      pct = change_percent(planned)
      if experiment_type == api.ExperimentType.HEAVY_UP:
        value = '+' + self.short(abs(amount))
      detail = f'{pct} of treatment BAU spend'
      if bau is not None:
        formula = BAU_FORMULA.format(self.short(bau), days, pct)
    elif mde_spend is not None:
      mde_abs, cpic = mde_spend
      formula = MDE_SPEND_FORMULA.format(
          format_number(mde_abs), self.money(cpic, 2)
      )
    return value, detail, formula

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
  def split_detail(share: float, cell_counts: Sequence[tuple[str, int]]) -> str:
    """'Treatment geos hold 25.0% of conversions (Treatment 1: 10, …)'."""
    text = f'Treatment geos hold {share:.1%} of conversions'
    if len(cell_counts) > 1:
      text += ' (' + ', '.join(
          f'{short_label(label)}: {n}' for label, n in cell_counts
      ) + ')'
    return text

  # ---------------------------------------------------------------- visuals

  @staticmethod
  def sensitivity_guide(mde: float, power: float) -> str:
    """Reading guide of the sensitivity curve."""
    return (
        'The blue curve shows how likely this geo test is to confirm a true'
        ' campaign lift of each size. Once the lift reaches'
        f' {format_percent(mde)} or higher (inside the green ≥ {power:.0%}'
        ' zone), the test is well-powered to separate ad impact from everyday'
        ' market variation.'
    )

  @staticmethod
  def sensitivity_takeaway(name: str, mde: float, power: float) -> str:
    """Takeaway of the sensitivity curve."""
    return (
        f'{name} detects lifts of {format_percent(mde)} or more {power:.0%} of'
        ' the time; smaller lifts may be missed.'
    )

  # ---------------------------------------------------------------- compare

  @staticmethod
  def largest_geos_takeaway(
      name: str, treated: int, count: int, spend_tracks: Optional[int]
  ) -> str:
    """'Option 1 puts 9 of the 22 largest geos in treatment. …'."""
    text = f'{name} puts {treated} of the {count} largest geos in treatment.'
    if spend_tracks is None:
      return text
    which = 'all' if spend_tracks == count else spend_tracks
    return f'{text} Spend tracks conversions in {which} of them.'

  @staticmethod
  def drift_takeaway(
      name: str,
      start: datetime.date,
      end: datetime.date,
      final: float,
      ratio: Optional[float],
      p_value: float,
  ) -> str:
    """Takeaway of the A/A drift chart."""
    ratio_text = f' ({ratio:.0%} of the MDE; ' if ratio is not None else ' ('
    verdict = (
        'Noise stays well inside what the test can detect.'
        if ratio is not None and ratio < 0.5
        else 'Noise alone comes close to the detectable effect.'
    )
    return (
        f'{name}: over {date_range(start, end)}, treatment geos drifted'
        f' {format_signed(final)} conversions from the'
        f' counterfactual{ratio_text}AA p = {p_value:.2f}). {verdict}'
    )

  # ---------------------------------------------------------------- quality

  @staticmethod
  def usable_geos_text(counts: dict[str, int], total: int) -> str:
    """'202 of 206 geos have usable data; 2 flagged by the quality checks'."""
    if not total:
      return 'No pretest data was attached to the design.'
    parts = [
        f'{counts[label]} {words}'
        for label, words in (
            (STATUS_FLAGGED, 'flagged by the quality checks'),
            (STATUS_NO_CONVERSIONS, 'without conversions'),
            (STATUS_USER, 'excluded by you'),
        )
        if counts.get(label)
    ]
    text = f'{counts.get(STATUS_OK, 0)} of {total} geos have usable data'
    return text + ('; ' + ', '.join(parts) if parts else '') + '.'

  @staticmethod
  def quality_findings(
      flagged_geos: Sequence[str],
      flagged_dates: Sequence[datetime.date],
      user_geos: Sequence[str],
      excluded_dates: Sequence[datetime.date],
  ) -> tuple[list[str], list[str]]:
    """(warnings, infos) of the data quality card, warnings first."""
    warnings = [prefix + text for prefix, items, text in (
        ('Geos flagged by data quality checks: ', flagged_geos,
         ', '.join(flagged_geos)),
        ('Outlier dates flagged by data quality checks: ', flagged_dates,
         date_list(flagged_dates)),
    ) if items]
    infos = [prefix + text for prefix, items, text in (
        ('User-excluded geos: ', user_geos, ', '.join(user_geos)),
        ('Dates excluded from the design: ', excluded_dates,
         date_list(excluded_dates)),
    ) if items]
    if not infos and not warnings:
      infos.append('No geos or dates were flagged or excluded.')
    return warnings, infos

  @staticmethod
  def unusual_days_takeaway(excluded_dates: Sequence[datetime.date]) -> str:
    """Takeaway of the unusual days chart."""
    if not excluded_dates:
      return 'No pretest days were flagged or excluded.'
    return (
        f'{plural(len(excluded_dates), "pretest day")} flagged or excluded'
        f' ({date_list(excluded_dates)}); leaving them out keeps the model'
        ' from learning one-off spikes.'
    )

  @staticmethod
  def tracking_gaps_takeaway(gap_names: Sequence[str], already: int) -> str:
    """Takeaway of the tracking gaps chart."""
    if not gap_names:
      return 'No geo has a week without conversions.'
    excluded = f'; {already} already excluded' if already else ''
    return (
        f'{plural(len(gap_names), "geo")} have weeks without any conversion'
        f' ({name_list(list(gap_names), 4)}), often a tracking gap{excluded},'
        ' check the others before the test.'
    )

  @staticmethod
  def unusual_weeks_takeaway(strong: int, top_names: Sequence[str]) -> str:
    """Takeaway of the unusual geo-weeks chart."""
    if not strong:
      return 'No geo-week deviates strongly (|z| > 3).'
    return (
        f'{plural(strong, "geo-week")} deviate strongly (|z| > 3), most often'
        f' in {name_list(list(top_names), 3)}.'
    )

  @staticmethod
  def constraints_snippet(
      geos: Sequence[str], dates: Sequence[datetime.date]
  ) -> str:
    """`geox.Constraints(...)` call that excludes the suggested items."""
    geo_set = ', '.join(repr(g) for g in sorted(geos))
    snippet = f'geox.Constraints(excluded_geos={{{geo_set}}}'
    if dates:
      date_set = ', '.join(f"pd.Timestamp('{d:%Y-%m-%d}')" for d in dates)
      snippet += f', excluded_dates={{{date_set}}}'
    return snippet + ')'

  @staticmethod
  def quality_rows(
      flagged_geos: Sequence[tuple[str, bool]],
      flagged_dates: Sequence[datetime.date],
      suggested_geos: Sequence[str],
  ) -> list[list[str]]:
    """Rows of the flagged geos and dates table.

    Args:
      flagged_geos: (display name, has spend but no conversions) per flagged
        geo.
      flagged_dates: Flagged dates.
      suggested_geos: Display names of geos suggested for exclusion.

    Returns:
      Table rows.
    """
    auto = 'Excluded automatically'
    reasons = {
        True: (
            'Spend but no conversions in the pretest, so the geo cannot'
            ' inform the counterfactual.'
        ),
        False: (
            'Flagged by the data quality checks; the geo cannot inform the'
            ' counterfactual.'
        ),
    }
    rows = [
        ['Geo', name, reasons[spend_only], auto]
        for name, spend_only in flagged_geos
    ]
    rows += [
        ['Date', f'{d:%Y-%m-%d}', (
            'Total conversions far from the pretest trend (leave-one-out gap'
            ' above Q3 + 3 × IQR of all days).'
        ), auto]
        for d in flagged_dates
    ]
    rows += [
        ['Geo', name, 'No conversions at all in the pretest.', 'Suggested']
        for name in suggested_geos
    ]
    return rows


QUALITY_NOTE = (
    'The data quality checks run before the design search and, by default, '
    'exclude what they flag.'
)

SNIPPET_INTRO = 'To exclude the suggested items, keep your other settings and use'

# Launch plan wording; {span} is a schedule_span(...), keys say whether the
# schedule has a cooldown (RUN_STEP, KEEP_STEP, PHASES_TEXT), a start date
# (ANALYZE_STEP), ramp-up days (CALENDAR_GUIDE) or a passed A/A check.
RUN_STEP = {
    False: (
        'Run the test {span}, with no cooldown. Hold the regional split'
        ' steady without running other geo-specific promotions.'
    ),
    True: (
        'Run the test {span}. Hold the regional split steady without running'
        ' other geo-specific promotions, then return all regions to business'
        ' as usual for the cooldown.'
    ),
}
_KEEP = 'Keep the split {span}, and avoid other geo-specific changes.'
KEEP_STEP = {False: _KEEP, True: _KEEP + (
    ' During the cooldown, return every geo to business as usual (pretest'
    ' levels).'
)}
_ANALYZE = 'Then run geox.analyze(...) with this design'
_READOUT = ', and geox.AnalysisSummarizer for the readout.'
ANALYZE_STEP = {False: _ANALYZE + _READOUT, True: _ANALYZE + (
    ' and the test start date ({start:%Y-%m-%d})'
) + _READOUT}
_PASS_START = ' Pass test_start to see calendar dates.'
PHASES_TEXT = {False: 'Phases: {s.test_days}-day test.' + _PASS_START, True: (
    'Phases: {s.test_days}-day test, then a {s.cooldown_days}-day cooldown.'
) + _PASS_START}
CALENDAR_GUIDE = {
    False: 'Blue days are test days; yellow days are cooldown.',
    True: (
        'Blue days are test days; yellow days at the start are ramp-up and'
        ' yellow days after the end are cooldown.'
    ),
}
VALIDATION_TEXT = {
    False: (
        'Pre-test validation found a pre-existing difference between the'
        ' groups, so review the A/A check before launching'
    ),
    True: (
        'Pre-test validation confirms that the control group closely tracks'
        ' treatment regions with no pre-existing bias'
    ),
}
POWER_STEP = (
    'To raise the chance of a conclusive result, plan at least {spend} of'
    ' {label}, run the test longer, or accept a larger MDE.'
)

# Delivery step per experiment type (per unit for a holdback); with one cell it
# ends with the DELIVERY_CONTROL clause, with several SHARED_CONTROL_STEP
# follows.
DELIVERY_STEPS = {
    api.ExperimentType.GO_DARK: (
        '{prefix}Pause campaign spend in the {n} {group} (about {value}{share})'
    ),
    api.ExperimentType.HEAVY_UP: (
        '{prefix}Increase campaign spend in the {n} {group}{by} (about'
        ' +{value})'
    ),
    'regions': (
        '{prefix}Deliver the campaign only in the {n} {group} (about {value}'
        ' planned spend)'
    ),
    'geos': (
        '{prefix}Run the campaign only in the {n} {group} (at least {value} of'
        ' spend there)'
    ),
}
_KEEP_NORMAL = ' while keeping normal spend in the {m} control {unit}'
DELIVERY_CONTROL = {
    api.ExperimentType.GO_DARK: _KEEP_NORMAL,
    api.ExperimentType.HEAVY_UP: _KEEP_NORMAL,
    'regions': ' and hold back the {m} control {unit}',
    'geos': ' and keep it off in the {m} control {unit}',
}
SHARED_CONTROL_STEP = (
    'Keep normal spend in the {m} control {unit} throughout the test.'
)

REVIEW_STEP = (
    'After the test window finishes, review the post-test analysis readout '
    'to confirm incremental conversions and efficiency.'
)

# Design check lines (DesignSummarizer._checks).
CHECK_MDE = 'MDE {} vs. goal {}'
CHECK_BUDGET = '{} {} vs. budget goal {}'
CHECK_SHARE = (
    'Treatment geos hold {:.1%} of conversions in the test geos (excluded geos'
    ' aside)'
)
CHECK_BOUNDARIES = '{} of {} geos matched a boundary polygon'
CHECK_MISSING = '; missing: {}'
CHECK_USER = 'Excluded by you: {}'
CHECK_FLAGGED = '{} flagged by data quality checks (see Pretest data quality)'
CHECK_EXCLUDED_DATES = '{} excluded (see Pretest data quality)'
BUDGET_TIP = (
    'Enter the spend planned for the treatment geos only. If {budget} were a'
    ' national budget, only about {treatment} would run in the treatment geos'
    ' ({share:.0%} of conversions){chance}.'
)
BUDGET_TIP_CHANCE = ', for a {:.0%} chance of a conclusive result'

# Compare card takeaways; INSPECT_TEXT is keyed by "is the recommended
# design" and FIT_TAKEAWAY by "good fit".
DETECTS_WITH = 'detects lifts of {} or more with {} of {}'
INSPECT_TEXT = {
    True: 'Inspecting {name}.{chance} {detects}.',
    False: (
        'Inspecting {name}; the recommended design is {best}.{chance}'
        ' {detects}.'
    ),
}
INSPECT_CHANCE = ' {} chance ({:.0%}):'
ZONE_INSIDE = 'inside the ideal zone (meets your goals)'
ZONE_OVER_BUDGET = (
    'outside the ideal zone: it needs more than your {} spend goal'
)
ZONE_OVER_MDE = (
    'outside the ideal zone: its detectable lift is above your {} goal'
)
ZONE_MISSES = 'outside the ideal zone: it misses at least one goal'
FIT_TAKEAWAY = {
    True: (
        '{}: control geos predict treatment-geo conversions closely before'
        ' the test (R² {:.2f}; minimum {:g}).'
    ),
    False: (
        '{}: the pretest fit is weaker than required (R² {:.2f}; minimum'
        ' {:g}).'
    ),
}
AA_DRIFT_TITLE = 'Would noise alone have looked like a lift before the test?'
SIMULATOR_TAKEAWAY = (
    'At your plan ({spend} of {label} at a {cpic} iCPA), the chance of a'
    ' conclusive result is {chance:.0%}; a {power:.0%} chance needs {needed}.'
    ' Move the controls to see how the numbers change.'
)

INSPECTING_NOTE = (
    'You are inspecting {}. The recommended design is {}; select it in the'
    ' comparison to go back.'
)

SIMULATOR_INTRO = (
    'Use the controls first; the chance of a conclusive result and the spend '
    'needed update with them.'
)

STRATA_TAKEAWAY = '{name}: every stratum contributes to both treatment and control.'

# Pretest data status of each geo, for the usable-data map.
STATUS_OK = 'Has data'
STATUS_FLAGGED = 'Flagged by quality checks'
STATUS_NO_CONVERSIONS = 'No conversions'
STATUS_USER = 'Excluded by you'


# Fixed report copy: card subtitles, chart titles, reading guides and
# callouts. Kept here, as in Meridian's summary_text, so the summarizer
# only wires data into them.

# TODO: Use the Methodology enum instead of assuming TBR
# once the enum refactor is submitted.
MAP_GUIDE = (
    'Colours show which geos get the change (treatment), stay as they are '
    '(control) or are left out (excluded). TBR estimates one pooled effect '
    'across all treatment geos, so there is no per-geo lift. Hover a geo for '
    'its pretest volume and stratum.'
)

MAP_COVERAGE_TITLE = 'Some geos are missing from the map'
MAP_COVERAGE_NOTE = (
    '{} geos have no matching boundary polygon and are not drawn on the map:'
    ' {}.'
)
REGIONS_SENTENCE = (
    '{}: treatment has {} geos with {:.0%} of pretest conversions{}; {}'
    ' control geos.'
)
REGIONS_LARGEST = ' (largest, {}, is {:.0%} of the group)'

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

# Recommended card wording; FIT_VALUE is (tile value, check line).
DETECTION_SENTENCE = (
    'It detects a lift of {} or more in treatment-geo conversions {:.0%} of'
    ' the time (alpha {:g}, {}).'
)
CHANCE_TITLE = ': {} chance of a conclusive result'
FIT_VALUE = {
    True: ('✓ Good fit', CHECK_GOOD_FIT),
    False: ('⚠ Weak fit', CHECK_WEAK_FIT),
}
FIT_DETAIL = 'R² {:.2f}; minimum {:g}'
AA_VALUE = {True: '✓ Passed', False: '✗ Failed'}
AA_DETAIL = 'AA p-value {:.2f}; passes at ≥ {:g}'
BAU_FORMULA = '= {} BAU spend in treatment geos over {} days × {}'
MDE_SPEND_FORMULA = '= {} incremental conversions (MDE) × {} iCPA'

CHECK_NO_MAP = 'No boundary file was passed, so maps are omitted.'

RECOMMENDED_SUBTITLE = (
    'Which design to run, why, and whether it meets your goals.'
)

BUDGET_SCOPE_TITLE = 'Budget goal = treatment-geo spend'

NATIONAL_BUDGET_NOTE = (
    'The budget goal is at least 90% of all spend across the market over the '
    'test window. It looks like a national budget; a holdback test only needs'
    ' the spend for the treatment geos.'
)

VISUALS_INTRO = (
    'Matched treatment and control regions run side by side while the '
    'sensitivity curve shows the smallest campaign lift this design can '
    'reliably confirm.'
)

SENSITIVITY_TITLE = 'What lift size can this test reliably detect?'

VISUALS_TITLE = 'Where the test runs and what it can detect'

COMPARE_TABLE_HELP = (
    'Click a row of the ranked table or a bubble to inspect that candidate. '
    'The map, largest geos, pretest fit, AA check and simulator update, and '
    'so do the design summary above and the export list below. ★ marks the '
    'recommended design.'
)

TRADEOFF_TITLE = 'Which design detects the smallest lift for the least spend?'

TRADEOFF_GUIDE = (
    'Each dot is a candidate; the selected one is outlined and labelled. Use '
    'the toggle to switch between detectable lift (lower is better) and max '
    'detectable iCPA (higher is better). Green dots meet every goal. Click a '
    'dot to select it.'
)

SIMULATOR_TITLE = 'What if the spend or target iCPA changes?'

COMPARE_SUBTITLE = (
    'Why the recommendation, and how it compares with the other options.'
)

LARGEST_GEOS_TITLE = (
    'Do the biggest geos split sensibly, and does spend follow?'
)

LARGEST_GEOS_GUIDE = (
    "Solid bars are each geo's share of pretest conversions, light bars its "
    'share of spend. A few very large geos can dominate the result.'
)

PRETEST_FIT_TITLE = 'Does the model track treatment geos before the test?'

PRETEST_FIT_GUIDE = (
    'Before the test, control geos should predict treatment closely. Red '
    'lines mark excluded dates.'
)

AA_DRIFT_GUIDE = (
    'Cumulative gap between observed and predicted treatment-geo conversions '
    'over the last test-length of pretest, when nothing changed. Dashed red '
    'lines mark ± the MDE.'
)

SIMULATOR_CHART_TITLE = (
    'How fast does the chance drop as spend falls or iCPA rises?'
)

SIMULATOR_GUIDE = (
    'The noise level is fixed by the geo split; spend and target iCPA set the'
    ' expected lift (spend ÷ iCPA incremental conversions). Top left: grey is'
    ' what noise alone produces, blue is the expected lift, and the shaded '
    'part right of the red line is the chance of a statistically significant '
    'result. Top right: how that chance falls as the iCPA rises. Bottom: the '
    'chance at each spend level. A warning appears when the target iCPA is '
    'far from the planned iCPA.'
)

LAUNCH_SUBTITLE = 'Next steps for the selected option.'

ROSTER_NOTE = {
    True: (
        'Showing {}. Pick another option in the comparison above to see its'
        ' geos.'
    ),
    False: (
        'Showing {}, selected in the comparison above. The recommended design'
        ' is Option 1.'
    ),
}
_CANDIDATES_NOTE = (
    'Chance: probability of a conclusive result at your goals. Geos CSV: one'
    ' row per geo with its group and stratum.'
)
# Keyed by "has a targeting map".
CANDIDATES_NOTE = {
    False: _CANDIDATES_NOTE,
    True: _CANDIDATES_NOTE + (
        ' Targeting CSV: your targeting map rows for each geo, with its group,'
        ' ready for ad-platform setup.'
    ),
}

STRATA_TITLE = 'How were geos split within each stratum?'

STRATA_GUIDE = (
    'Stratified sampling groups similar geos into strata and splits each '
    'stratum between groups, so treatment and control look alike.'
)

USABLE_GEOS_GUIDE = (
    'Each geo is coloured by its pretest data: usable, flagged by the quality'
    ' checks, without any conversions, or excluded by you. Only usable geos '
    'enter the design.'
)

DAILY_SPEND_TITLE = 'Daily spend in the pretest data'

UNUSUAL_DAYS_TITLE = 'Were there unusual days in the pretest?'

UNUSUAL_DAYS_GUIDE = 'Marked dates were excluded from the design.'

TRACKING_GAPS_GUIDE = (
    'Weekly conversions of the largest geos plus any geo with an empty week. '
    'Red cells are weeks without conversions.'
)

UNUSUAL_WEEKS_TITLE = 'Which geo-weeks behave unusually?'

UNUSUAL_WEEKS_GUIDE = (
    'Robust z-scores (median and MAD per geo) of weekly conversions. Dark '
    'cells are unusual weeks.'
)
