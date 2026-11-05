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
import enum
import math
from typing import Literal, Optional

from meridian_geox import api

Tone = Literal['neutral', 'info', 'success', 'warning', 'error']

RECOMMENDED_SUFFIX = ' (Recommended)'
SHORT_ID_LENGTH = 8


def option_name(rank: int) -> str:
  """Returns the display name of the candidate at the given 1-based rank."""
  if rank < 1:
    raise ValueError(f'rank must be >= 1, got {rank}.')
  return f'Option {rank}' + (RECOMMENDED_SUFFIX if rank == 1 else '')


def short_id(design_id: str) -> str:
  """Returns a short, human-friendly prefix of a design ID."""
  return str(design_id)[:SHORT_ID_LENGTH]


@dataclasses.dataclass(frozen=True)
class SpendTerms:
  """Spend vocabulary for an experiment type.

  Attributes:
    label: Name of the spend amount (for example "Campaign spend").
    change: How spend changes in treatment geos.
    treatment_action: Short description of what happens in treatment geos.
  """

  label: str
  change: str
  treatment_action: str


_SPEND_TERMS = {
    api.ExperimentType.HOLDBACK: SpendTerms(
        label='Campaign spend',
        change='Campaign runs only in treatment geos',
        treatment_action='campaign on',
    ),
    api.ExperimentType.HEAVY_UP: SpendTerms(
        label='Extra spend',
        change='Spend increases in treatment geos',
        treatment_action='spend increased',
    ),
    api.ExperimentType.GO_DARK: SpendTerms(
        label='Paused spend',
        change='Ads are paused in treatment geos',
        treatment_action='ads paused',
    ),
}
_MULTI_CELL_TERMS = SpendTerms(
    label='Incremental spend',
    change='Spend changes differ by treatment cell',
    treatment_action='spend changed',
)


def spend_terms(experiment_type: Optional[api.ExperimentType]) -> SpendTerms:
  """Returns spend vocabulary; `None` means a multi-cell experiment."""
  if experiment_type is None:
    return _MULTI_CELL_TERMS
  return _SPEND_TERMS[experiment_type]


EXPERIMENT_TYPE_LABELS = {
    api.ExperimentType.HOLDBACK: 'Holdback',
    api.ExperimentType.HEAVY_UP: 'Heavy-up',
    api.ExperimentType.GO_DARK: 'Go dark',
}


# Plain-language description of what happens in treatment and control geos.
EXPERIMENT_TYPE_EXPLANATIONS = {
    api.ExperimentType.HOLDBACK: 'ads show only in treatment geos',
    api.ExperimentType.HEAVY_UP: 'extra spend in treatment geos',
    api.ExperimentType.GO_DARK: 'ads paused in treatment geos',
}

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


def plural(count: int, noun: str, plural_noun: Optional[str] = None) -> str:
  """Returns `'1 geo'` / `'3 geos'`."""
  word = noun if count == 1 else (plural_noun or noun + 's')
  return f'{count:,} {word}'


def name_list(names: list[str], limit: int = 3) -> str:
  """Returns `'A, B, C and 4 more'`."""
  if not names:
    return ''
  shown = ', '.join(names[:limit])
  rest = len(names) - limit
  return f'{shown} and {rest} more' if rest > 0 else shown


def short_money(value: Optional[float], currency_symbol: str = '') -> str:
  """Formats money compactly: `$8.7k`, `$1.2M`, `$950`."""
  if value is None or (isinstance(value, float) and math.isnan(value)):
    return 'n/a'
  if math.isinf(value):
    return '∞' if value > 0 else '-∞'
  sign = '-' if value < 0 else ''
  amount = abs(value)
  if amount >= 1e6:
    text = f'{amount / 1e6:.1f}M'
  elif amount >= 1e3:
    text = f'{amount / 1e3:.1f}k'
  else:
    text = f'{amount:,.0f}'
  return f'{sign}{currency_symbol}{text}'


def short_date(value) -> str:
  """Formats a timestamp like `Wed, Apr 1, 2020`."""
  return f'{value:%a, %b} {value.day}, {value.year}'


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


@dataclasses.dataclass(frozen=True)
class ResultWording:
  """Wording for an analysis verdict."""

  question: str
  meaning: str
  tone: Tone


RESULT_WORDING = {
    Verdict.YES: ResultWording(
        question='Did the ads drive extra conversions?',
        meaning=(
            'The test found a statistically significant effect: treatment geos'
            ' performed differently from what the control geos predict.'
        ),
        tone='success',
    ),
    Verdict.NO: ResultWording(
        question='Did the ads drive extra conversions?',
        meaning=(
            'The test did not find a statistically significant effect. The true'
            ' effect may be zero or smaller than the test could detect.'
        ),
        tone='warning',
    ),
    Verdict.TOO_EARLY: ResultWording(
        question='Is there a result yet?',
        meaning=(
            'The test is still running. Interim numbers can move a lot; wait'
            ' for the planned end date before making decisions.'
        ),
        tone='info',
    ),
    Verdict.NEEDS_REVIEW: ResultWording(
        question='Can we trust this result?',
        meaning=(
            'Something in the readout needs an analyst to look at it before the'
            ' result is used (see the checks below).'
        ),
        tone='error',
    ),
}


def format_number(value: Optional[float], digits: int = 0) -> str:
  """Formats a number with thousands separators."""
  if value is None or (isinstance(value, float) and math.isnan(value)):
    return 'n/a'
  if math.isinf(value):
    return '∞' if value > 0 else '-∞'
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
  if value is None or (isinstance(value, float) and math.isnan(value)):
    return 'n/a'
  if math.isinf(value):
    return '∞' if value > 0 else '-∞'
  sign = '-' if value < 0 else ''
  return f'{sign}{currency_symbol}{abs(value):,.{digits}f}'


def format_percent(
    value: Optional[float], digits: int = 1, signed: bool = False
) -> str:
  """Formats a fraction (0.05) as a percentage (5.0%)."""
  if value is None or (isinstance(value, float) and math.isnan(value)):
    return 'n/a'
  if math.isinf(value):
    return '∞' if value > 0 else '-∞'
  text = f'{value * 100:,.{digits}f}%'
  if signed and value > 0:
    text = '+' + text
  return text


def format_interval(lower: str, upper: str, level: float) -> str:
  """Formats a confidence interval line."""
  return f'{level:.0%} CI: {lower} to {upper}'


def test_objective(
    experiment_type: Optional[api.ExperimentType],
    n_treatment_geos: int,
    n_control_geos: int,
    test_days: int,
    response_is_value: bool = False,
) -> str:
  """Returns a one-sentence description of what the test measured."""
  kpi = 'conversion value' if response_is_value else 'conversions'
  if experiment_type is None:
    change = 'different spend changes in each treatment cell'
  elif experiment_type == api.ExperimentType.HOLDBACK:
    change = 'running the campaign'
  elif experiment_type == api.ExperimentType.HEAVY_UP:
    change = 'increasing spend'
  else:
    change = 'pausing ads'
  return (
      f'Objective: measure the extra {kpi} caused by {change} in'
      f' {n_treatment_geos} treatment geos, compared with'
      f' {n_control_geos} control geos, over a {test_days}-day test.'
  )


def headline(
    verdict: Verdict,
    incremental: str,
    efficiency_label: str,
    efficiency: str,
    response_is_value: bool = False,
) -> str:
  """Returns the business headline of an analysis readout."""
  kpi = 'in extra conversion value' if response_is_value else (
      'extra conversions'
  )
  if verdict == Verdict.TOO_EARLY:
    return (
        f'Interim estimate: {incremental} {kpi} so far at {efficiency} '
        f'{efficiency_label} (not final).'
    )
  if verdict == Verdict.NO:
    return (
        f'Estimated {incremental} {kpi} at {efficiency} {efficiency_label}, '
        'but the effect is not statistically significant.'
    )
  if verdict == Verdict.NEEDS_REVIEW:
    return (
        f'Estimated {incremental} {kpi} at {efficiency} {efficiency_label} — '
        'review the checks before using this result.'
    )
  return f'The ads drove an estimated {incremental} {kpi} at {efficiency} ' + (
      f'{efficiency_label}.'
  )


def next_step(verdict: Verdict, efficiency_label: str) -> str:
  """Returns an actionable but non-automated suggested next step."""
  if verdict == Verdict.YES:
    return (
        f'Suggested next step: use the measured {efficiency_label} and its'
        ' confidence interval as an input to budget planning, and as a'
        ' calibration prior for your marketing mix model (MMM).'
    )
  if verdict == Verdict.NO:
    return (
        'Suggested next step: do not treat this as proof of zero effect. Review'
        ' the detectable effect size, and consider a longer or larger test'
        ' before shifting budget. The confidence interval can still bound the'
        ' effect when calibrating your MMM.'
    )
  if verdict == Verdict.TOO_EARLY:
    return (
        'Suggested next step: keep the test running as planned and avoid'
        ' changing spend in test geos until the planned end date.'
    )
  return (
      'Suggested next step: ask an analyst to review the execution and data'
      ' checks before using this result for budget planning or MMM'
      ' calibration.'
  )


@dataclasses.dataclass(frozen=True)
class ExecutionWording:
  """Wording for the spend execution check."""

  title: str
  sentence: str
  tone: Tone


def execution_wording(
    observed_change: Optional[float],
    planned_change: Optional[float],
    verified: Optional[bool],
    issue_geos: tuple[str, ...] = (),
    has_spend: Optional[bool] = None,
) -> ExecutionWording:
  """Returns wording that separates observed spend from plan verification.

  Args:
    observed_change: Observed change in treatment spend relative to control
      geos (0.3 means +30%), or None when it cannot be computed.
    planned_change: Planned relative change, or None when no plan is known.
    verified: Whether the observed change matches the plan; None if there is
      no plan to compare against.
    issue_geos: Treatment geos whose spend did not change as expected.
    has_spend: Whether spend data was passed. Defaults to whether
      `observed_change` is known.

  Returns:
    The title, sentence and tone for the execution check.
  """
  if has_spend is None:
    has_spend = observed_change is not None
  if not has_spend:
    return ExecutionWording(
        title='Spend execution not checked',
        sentence=(
            'No spend data was passed, so the report cannot show how spend'
            ' changed during the test.'
        ),
        tone='neutral',
    )
  observed = (
      format_percent(observed_change, 0, signed=True)
      if observed_change is not None
      else 'new spend (no pretest spend)'
  )
  if issue_geos:
    names = ', '.join(issue_geos[:5]) + (
        f' and {len(issue_geos) - 5} more' if len(issue_geos) > 5 else ''
    )
    return ExecutionWording(
        title='Spend did not change as planned',
        sentence=(
            f'Observed treatment spend change: {observed} relative to control.'
            ' These'
            f' treatment geos did not follow the plan: {names}.'
        ),
        tone='error',
    )
  if planned_change is None or verified is None:
    return ExecutionWording(
        title='Observed spend change',
        sentence=(
            f'Treatment spend changed by {observed} relative to control. This'
            ' is an observation only; it has not been verified against a planned'
            ' spend target.'
        ),
        tone='neutral',
    )
  planned = format_percent(planned_change, 0, signed=True)
  if verified:
    return ExecutionWording(
        title='Spend changed as planned',
        sentence=(
            f'Observed treatment spend change ({observed}) is in line with the'
            f' planned change ({planned}).'
        ),
        tone='success',
    )
  return ExecutionWording(
      title='Spend differs from plan',
      sentence=(
          f'Observed treatment spend change ({observed}) differs from the'
          f' planned change ({planned}).'
      ),
      tone='warning',
  )


def effect_split_sentence(
    test_effect: str, cooldown_effect: str, cooldown_days: int, kpi: str
) -> str:
  """Describes how the effect splits between the test and cooldown periods."""
  return (
      f'{test_effect} extra {kpi} came during the test period and'
      f' {cooldown_effect} during the {cooldown_days}-day cooldown after it.'
  )
