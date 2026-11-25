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
import enum
import math
from typing import Literal, Optional, Sequence

from meridian_geox import api

Tone = Literal['neutral', 'info', 'success', 'warning', 'error']


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
DOCS_INTRO_TO_ANALYSIS = f'{_DOCS}/analysis-outputs'

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


class Verdict(enum.Enum):
  YES = 'YES'
  NO = 'NO'
  TOO_EARLY = 'TOO EARLY'
  NEEDS_REVIEW = 'NEEDS REVIEW'


RESULT_TONE: dict[Verdict, Tone] = {
    Verdict.YES: 'success', Verdict.NO: 'warning', Verdict.TOO_EARLY: 'info',
    Verdict.NEEDS_REVIEW: 'error',
}
NEEDS_REVIEW_MEANING = (
    'Something in the readout needs an analyst to look at it before the'
    ' result is used (see the checks below).'
)


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


# The response wording, keyed by response_is_value.
KPI = {False: 'conversions', True: 'conversion value'}
EXTRA_KPI = {False: 'extra conversions', True: 'in extra conversion value'}
CI_LINE = '{level:.0%} CI: {lower} to {upper}'
# AnalysisText wording, keyed by response_is_value.
_VALUE_TEXT = {
    False: dict(
        kpi=KPI[False], label='iCPA',
        efficiency_metric='iCPA (cost per incremental conversion)',
        efficiency_tile='Cost per extra conversion',
        formula='iCPA = campaign spend ÷ extra conversions.',
    ),
    True: dict(
        kpi=KPI[True], label='iROAS',
        efficiency_metric='iROAS (conversion value per unit of spend)',
        efficiency_tile='Return per unit of spend',
        formula='iROAS = conversion value ÷ campaign spend.',
    ),
}

_OBJECTIVE_CHANGES = {
    api.ExperimentType.HOLDBACK: 'running the campaign',
    api.ExperimentType.HEAVY_UP: 'increasing spend',
    api.ExperimentType.GO_DARK: 'pausing ads',
}


def test_objective(
    cells: Sequence[tuple[str, api.ExperimentType, int]],
    n_control_geos: int,
    test_days: int,
    response_is_value: bool = False,
) -> str:
  """What the test measured, from (label, type, treatment geos) per cell."""
  kpi = KPI[response_is_value]
  if len(cells) == 1:
    _, et, n = cells[0]
    caused, shared = f'{_OBJECTIVE_CHANGES[et]} in {n} treatment geos', ''
  else:
    parts = [
        f'{_OBJECTIVE_CHANGES[et]} in {label} ({n} geos)'
        for label, et, n in cells
    ]
    caused, shared = ', '.join(parts[:-1]) + ' and ' + parts[-1], 'shared '
  return (
      f'Objective: measure the extra {kpi} caused by {caused}, compared with'
      f' {n_control_geos} {shared}control geos, over a {test_days}-day test.'
  )


# Suggested next step after the banner; YES takes the efficiency label.
NEXT_STEP = {
    Verdict.YES: (
        'Suggested next step: use the measured {} and its confidence interval'
        ' as an input to budget planning, and as a calibration prior for your'
        ' marketing mix model (MMM).'
    ),
    Verdict.NO: (
        'Suggested next step: do not treat this as proof of zero effect. Review'
        ' the detectable effect size, and consider a longer or larger test'
        ' before shifting budget. The confidence interval can still bound the'
        ' effect when calibrating your MMM.'
    ),
    Verdict.TOO_EARLY: (
        'Suggested next step: keep the test running as planned and avoid'
        ' changing spend in test geos until the planned end date.'
    ),
    Verdict.NEEDS_REVIEW: (
        'Suggested next step: ask an analyst to review the execution and data'
        ' checks before using this result for budget planning or MMM'
        ' calibration.'
    ),
}
# Verdict tile title (keyed by multi-cell) and effect totals chart title.
VERDICT_QUESTION = {
    True: '{label}: extra {kpi}?', False: 'Did it drive extra {kpi}?'
}
TOTALS_TITLE = 'How many extra {} did the campaign drive?'


ICPD_METRIC = 'iCPD (extra conversions per unit of spend; iCPA = 1 / iCPD)'
INCOMPLETE_NOTE = (
    'The analysis window covers {} of the {} planned test days. Re-run'
    ' analyze() with analysis_end_date on or after {:%Y-%m-%d} for the full'
    ' result.'
)
# Prefix of a multi-cell banner title.
MULTI_TITLE_PREFIX = {
    Verdict.TOO_EARLY: 'Interim: ', Verdict.NEEDS_REVIEW: 'Needs review: '
}
# (Effect wording while the change ran, what the counterfactual assumes).
_LIFT_CONTEXT = {
    api.ExperimentType.GO_DARK: (
        ' while ads were paused', 'had ads kept running'
    ),
    api.ExperimentType.HEAVY_UP: ('', 'without the extra spend'),
}
# What went wrong in the flagged geos, per experiment type.
_EXECUTION_PROBLEMS = {
    api.ExperimentType.GO_DARK: (
        'kept spending while ads should have been paused'
    ),
    api.ExperimentType.HEAVY_UP: 'did not increase spend',
}


@dataclasses.dataclass(frozen=True)
class AnalysisText:
  """Number formats and sentences of the analysis readout.

  Methods take plain numbers, so the summarizer only wires report_data
  values into them.

  Attributes:
    response_is_value: Whether conversions are monetary value (iROAS).
    currency_symbol: Symbol prefixed to monetary values.
    target_cpic: Target iCPA of the design, if any.
    kpi: 'conversions' or 'conversion value'.
    label: Efficiency label, 'iCPA' or 'iROAS'.
    efficiency_metric: Name of the efficiency row of the estimates table.
    efficiency_tile: Title of the efficiency tile.
    formula: How the efficiency metric is defined.
  """

  response_is_value: bool = False
  currency_symbol: str = ''
  target_cpic: Optional[float] = None
  kpi: str = dataclasses.field(init=False)
  label: str = dataclasses.field(init=False)
  efficiency_metric: str = dataclasses.field(init=False)
  efficiency_tile: str = dataclasses.field(init=False)
  formula: str = dataclasses.field(init=False)

  def __post_init__(self):
    for name, value in _VALUE_TEXT[self.response_is_value].items():
      object.__setattr__(self, name, value)

  def short(self, value: Optional[float]) -> str:
    """Compact monetary value, e.g. '$1.2k'."""
    return short_money(value, self.currency_symbol)

  def count(self, value: float) -> str:
    """Compact count (or value), e.g. '34.0k'."""
    return short_money(
        value, self.currency_symbol if self.response_is_value else ''
    )

  def efficiency(self, value: float) -> str:
    """Formats an iCPA (money) or iROAS (ratio)."""
    if self.response_is_value:
      return format_number(value, 2)
    return format_money(value, self.currency_symbol, 2)

  def lift(self, value: float, signed: bool = True) -> str:
    """Formats incremental conversions (or value)."""
    if self.response_is_value:
      text = format_money(value, self.currency_symbol)
      return ('+' + text) if signed and value > 0 else text
    return format_signed(value) if signed else format_number(value)

  def meets_target(self, value: float) -> Optional[bool]:
    """Whether an iCPA is at or below the target; None without a target."""
    if self.target_cpic is None or self.response_is_value:
      return None
    return value <= self.target_cpic

  def target_phrase(self, value: float, short: bool = False) -> str:
    """'below your $1.00 target' (or '' without a target)."""
    met = self.meets_target(value)
    if met is None:
      return ''
    assert self.target_cpic is not None
    target = self.efficiency(self.target_cpic)
    if short:
      return f', target {target}'
    return f'{"below" if met else "above"} your {target} target'

  def lift_sentence(
      self, experiment_type: api.ExperimentType, point: Optional[float]
  ) -> str:
    """Effect in extra conversions (geo tests avoid relative-lift terms)."""
    point = point or 0.0
    # TODO: Use the Methodology enum instead of assuming TBR
    # once the enum refactor is submitted.
    # TBR reports go-dark effects with the sign flipped: positive means
    # conversions fell while ads were paused.
    go_dark = experiment_type == api.ExperimentType.GO_DARK
    word = 'fewer' if (point >= 0) == go_dark else 'more'
    during, change = _LIFT_CONTEXT.get(
        experiment_type, ('', 'without the campaign')
    )
    return (
        f'Treatment geos had about {self.count(abs(point))} {word} {self.kpi}'
        f'{during} than expected {change}.'
    )

  def cost_sentence(self, point: Optional[float]) -> str:
    """'Each extra conversion cost about $0.40 (iCPA, target $1.00).'."""
    if point is None:
      return ''
    value = self.efficiency(point)
    if self.response_is_value:
      return f'Each unit of spend returned about {value} ({self.label}).'
    return (
        f'Each extra conversion cost about {value}'
        f' ({self.label}{self.target_phrase(point, short=True)}).'
    )

  def no_lift_sentence(self, upper: float, level: float) -> str:
    """What a non-significant result still rules out."""
    return (
        f'More than about {self.count(upper)} extra {self.kpi} is unlikely'
        f' (the top of the {level:.0%} CI); smaller effects cannot be ruled'
        ' out.'
    )

  def _at(self, efficiency: Optional[float]) -> str:
    """' at $0.40 iCPA' (or '' without an efficiency)."""
    if efficiency is None:
      return ''
    return f' at {self.efficiency(efficiency)} {self.label}'

  def headline(
      self, verdict: Verdict, lift: float, efficiency: Optional[float],
      significant_decrease: bool,
  ) -> str:
    """Business headline of a single-cell readout."""
    estimate = f'{self.lift(lift)} {EXTRA_KPI[self.response_is_value]}'
    at = self._at(efficiency)
    if verdict == Verdict.TOO_EARLY:
      return f'Interim estimate: {estimate} so far{at} (not final).'
    if significant_decrease:
      return (
          f'Estimated {estimate}: a statistically significant'
          ' decrease — review the checks before using this result.'
      )
    if verdict == Verdict.NO:
      return (
          f'Estimated {estimate}{at}, but the effect is not'
          ' statistically significant.'
      )
    if verdict == Verdict.NEEDS_REVIEW:
      return (
          f'Estimated {estimate}{at} — review the checks before using this'
          ' result.'
      )
    return f'The ads drove an estimated {estimate}{at}.'

  def short_result(
      self, lift: float, efficiency: Optional[float], significant: bool,
      significant_decrease: bool, verdict: Verdict,
  ) -> str:
    """'+12k extra conversions at $0.40 iCPA' or '… (not significant)'."""
    text = f'{self.lift(lift)} extra {self.kpi}{self._at(efficiency)}'
    if significant_decrease:
      return text + ' (significant decrease)'
    if not significant:
      return text + ' (not significant)'
    if verdict == Verdict.NEEDS_REVIEW:
      return text + ' (needs review)'
    return text

  def banner_text(
      self, overall: Verdict, lead: str, test_days: int, planned_days: int,
      issue_labels: Sequence[str],
  ) -> str:
    """Banner body: the result, caveats and the suggested next step."""
    sentences = [lead]
    if overall == Verdict.TOO_EARLY:
      sentences.append(
          f'Results cover {test_days} of {planned_days} planned test days and'
          ' may change.'
      )
    if issue_labels:
      sentences.append(
          f'⚠ Spend did not change as planned in {", ".join(issue_labels)};'
          ' see "Was the test run as planned?" below before acting on this.'
      )
    sentences.append(NEXT_STEP[overall].format(self.label))
    return ' '.join(s for s in sentences if s)

  def meaning(
      self, verdict: Verdict, experiment_type: api.ExperimentType,
      lift: api.Estimate, significant: bool, significant_decrease: bool,
      level: float, observed_days: int, efficiency: Optional[float],
      cooldown: Optional[tuple[float, float, int, int, str]] = None,
  ) -> list[str]:
    """'What this means' paragraphs for one cell (cooldown: split, if any)."""
    if verdict == Verdict.TOO_EARLY:
      return [
          'So far the campaign shows a significant lift, but interim results'
          ' can change.'
          if significant
          else 'So far there is no significant lift; interim results can'
          ' change.'
      ]
    lift_text = self.lift_sentence(experiment_type, lift.point_estimate)
    if significant_decrease:
      return [
          'The test found a statistically significant decrease. '
          + lift_text
          + ' Check the setup and execution before using this result.'
      ]
    if not significant:
      return [
          'The test found no statistically significant effect. '
          + self.no_lift_sentence(lift.upper_bound, level)
      ]
    text = lift_text.rstrip('.') + f' over {observed_days} days'
    if efficiency is not None and not self.response_is_value:
      text += f', at about {self.efficiency(efficiency)} each'
      target = self.target_phrase(efficiency)
      if target:
        text += f' ({target})'
    paragraphs = [text + '.']
    if cooldown is not None:
      test_effect, cooldown_effect, test_days, cooldown_days, dates = cooldown
      paragraphs.append(
          f'Includes ~{self.count(test_effect)} extra {self.kpi} during the'
          f' {test_days}-day test and ~{self.count(cooldown_effect)} delayed'
          f' {self.kpi} captured in the {cooldown_days}-day cooldown'
          f' ({dates}).'
      )
    if verdict == Verdict.NEEDS_REVIEW:
      paragraphs.append(NEEDS_REVIEW_MEANING)
    return paragraphs

  def next_step_text(
      self, verdict: Verdict, significant: bool, planned_end: datetime.date
  ) -> str:
    """Analyst next step for one cell."""
    if verdict == Verdict.TOO_EARLY:
      return (
          f'Wait for the planned end ({planned_end:%Y-%m-%d}) and re-run'
          ' analyze() with analysis_end_date on or after it.'
      )
    if verdict == Verdict.NEEDS_REVIEW:
      return (
          'Check how spend changed in the flagged geos before using this'
          ' result; consider re-running the analysis without them.'
      )
    if not significant:
      return (
          'To detect a smaller effect next time, plan a larger spend change,'
          ' a longer test or more treatment geos.'
      )
    return (
        'Use this result to plan budgets and to calibrate your marketing mix'
        ' model; share the summary with the intervals below.'
    )

  def execution_sentence(
      self, experiment_type: api.ExperimentType, has_spend: bool,
      observed_change: Optional[float], planned_change: Optional[float],
      verified: Optional[bool], issue_geos: Sequence[str], ok: Optional[bool],
  ) -> str:
    """Execution check sentence, separating observation from verification."""
    if not has_spend:
      return (
          'No spend data was passed, so the report cannot show how spend'
          ' changed during the test.'
      )
    if observed_change is None:
      return 'Treatment geos had no pretest spend to compare against.'
    change = format_percent(observed_change, 0, signed=True)
    if issue_geos:
      problem = _EXECUTION_PROBLEMS.get(experiment_type) or (
          'had no campaign spend during the test' if planned_change is None
          else 'did not follow the plan'
      )
      return (
          f'Treatment spend changed by {change} compared with control geos,'
          f' but {plural(len(issue_geos), "treatment geo")} {problem}:'
          f' {name_list(list(issue_geos), 5)}.'
      )
    if experiment_type == api.ExperimentType.GO_DARK and verified is None:
      if ok:
        return (
            f'Treatment spend fell {format_percent(abs(observed_change), 0)},'
            ' as a go-dark requires.'
        )
      return f'Treatment spend only changed {change}; a go-dark needs ads paused.'
    if planned_change is None or verified is None:
      return (
          f'Treatment spend changed by {change} compared with control geos.'
          ' This is an observation only; it has not been verified against a'
          ' planned spend target.'
      )
    planned = format_percent(planned_change, 0, signed=True)
    verb = 'is in line with' if verified else 'differs from'
    return (
        f'Observed treatment spend change ({change}) {verb} the planned'
        f' change ({planned}).'
    )

  def execution_title(
      self, statuses: Sequence[Optional[bool]], any_spend: bool
  ) -> tuple[str, Tone]:
    """Title and tone of 'Was the test run as planned?'."""
    if False in statuses:
      return '⚠ Was the test run as planned?', 'warning'
    if all(ok is True for ok in statuses):
      return '✓ Was the test run as planned?', 'success'
    if not any_spend:
      return 'ℹ Spend execution not checked', 'neutral'
    if not any(statuses):
      return 'ℹ Observed spend change', 'neutral'
    return 'ℹ Was the test run as planned?', 'neutral'

  def verdict_detail(
      self, verdict: Verdict, significant: bool, significant_decrease: bool,
      execution_ok: bool, test_days: int, planned_days: int,
  ) -> str:
    """Detail line of the verdict tile."""
    if verdict == Verdict.YES:
      return f'A statistically significant lift in {self.kpi}.'
    if verdict == Verdict.NO:
      return 'No statistically significant effect.'
    if verdict == Verdict.TOO_EARLY:
      so_far = 'a' if significant else 'no'
      return (
          f'Day {test_days} of {planned_days}: {so_far} significant lift so'
          ' far.'
      )
    if significant_decrease:
      return 'A statistically significant decrease.'
    if not execution_ok:
      return 'Spend did not change as planned (see below).'
    return 'The estimate needs an analyst review.'

  @staticmethod
  def lift_detail(
      significant: bool, significant_decrease: bool, observed_days: int
  ) -> str:
    """Detail line of the extra conversions tile."""
    if significant:
      return f'over {observed_days} analysed days'
    if significant_decrease:
      return 'significant decrease'
    return 'not statistically significant'

  def efficiency_detail(self, value: float) -> str:
    """Detail line of the efficiency tile."""
    target = self.target_phrase(value)
    return target.replace('below your', 'within your') if target else self.label

  @property
  def totals_guide(self) -> str:
    """Reading guide of the effect totals chart."""
    return (
        f'Left: total {self.kpi} in treatment geos during the test, as'
        ' expected without the change (counterfactual) and as observed.'
        ' Right: the difference, i.e. incremental'
        f' {self.kpi}, with its confidence interval. An interval that crosses'
        ' zero means the effect is not statistically significant.'
    )

  def totals_takeaway(
      self, experiment_type: api.ExperimentType, observed: float,
      counterfactual: float, lift: api.Estimate, level: float,
      significant: bool, significant_decrease: bool,
  ) -> str:
    """'Treatment geos converted 917.5k against 883.6k expected …'."""
    expected = (
        'expected had ads kept running'
        if experiment_type == api.ExperimentType.GO_DARK
        else 'expected without the change'
    )
    if significant_decrease:
      ending = '; the whole interval is below zero (a significant decrease).'
    elif not significant:
      ending = (
          '; the interval includes zero, so the effect is not statistically'
          ' significant.'
      )
    else:
      ending = '.'
    return (
        f'Treatment geos converted {self.count(observed)} against'
        f' {self.count(counterfactual)} {expected}:'
        f' {self.lift(lift.point_estimate)} incremental {self.kpi}'
        f' ({level:.0%} CI {self.lift(lift.lower_bound)} to'
        f' {self.lift(lift.upper_bound)}){ending}'
    )

  @staticmethod
  def details_intro(level: float) -> str:
    """Intro of the Detailed results card."""
    return (
        'How the observed and counterfactual totals give the effect, and how'
        f' certain each estimate is ({level:.0%} CIs in brackets and in the'
        ' charts). The headline numbers, their intervals and the effect over'
        ' time are in the Executive summary.'
    )

  def equation_strip(
      self,
      prefix: str,
      experiment_type: api.ExperimentType,
      observed: float,
      counterfactual: float,
      lift: api.Estimate,
      level: float,
  ) -> str:
    """'917.5k Observed − 883.6k Counterfactual = 34.0k Incremental · CI'."""
    if experiment_type == api.ExperimentType.GO_DARK:
      expr = (
          f'{self.count(counterfactual)} Counterfactual (expected had ads kept'
          f' running) − {self.count(observed)} Observed in treatment geos'
      )
    else:
      expr = (
          f'{self.count(observed)} Observed in treatment geos −'
          f' {self.count(counterfactual)} Counterfactual (expected)'
      )
    return (
        f'{prefix}{expr} = {self.count(lift.point_estimate)} Incremental ·'
        f' {level:.0%} CI [{self.count(lift.lower_bound)},'
        f' {self.count(lift.upper_bound)}]'
    )

  def efficiency_confidence_takeaway(
      self, label: str, point: float, lower: float, upper: float, level: float
  ) -> str:
    """Takeaway of the iCPA (iROAS) confidence curve."""
    return (
        f'{label} {self.efficiency(point)}, with a {level:.0%} interval of'
        f' {self.efficiency(lower)} to {self.efficiency(upper)}.'
    )

  @staticmethod
  def lift_confidence_guide(level: float) -> str:
    """Reading guide of the incremental conversions confidence curve."""
    return (
        'How confident we are that the true effect is above each value on the'
        f' x-axis; the dashed lines mark the {level:.0%} interval.'
    )

  def lift_confidence_takeaway(self, lift: api.Estimate, level: float) -> str:
    """Takeaway of the incremental conversions confidence curve."""
    return (
        f'{self.lift(lift.point_estimate)} extra {self.kpi}, {level:.0%}'
        f' interval {self.lift(lift.lower_bound)} to'
        f' {self.lift(lift.upper_bound)}.'
    )


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
    warnings = [prefix + text for prefix, text in (
        ('Geos flagged by data quality checks: ', ', '.join(flagged_geos)),
        ('Outlier dates flagged by data quality checks: ',
         date_list(flagged_dates)),
    ) if text]
    infos = [prefix + text for prefix, text in (
        ('User-excluded geos: ', ', '.join(user_geos)),
        ('Dates excluded from the design: ', date_list(excluded_dates)),
    ) if text]
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
    """Flagged geos and dates table rows; geos are (name, spend only)."""
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

IMPLIED_SPEND_DETAIL = 'implied by the estimates, no interval'

CONFIDENCE_GUIDE = (
    'How confident we are that the true value is above each point on the '
    'x-axis, based on the reported placebo confidence interval.'
)
