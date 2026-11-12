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

"""Report data derived from GeoX design and analysis objects.

Everything here is plain pandas/numpy computation on objects the Colab already
produces (`DesignSet`, `Design`, `AnalysisResult`); no model is re-run and no
chart is drawn.
"""

import dataclasses
import datetime
import math
import re
from typing import Any, Iterable, Optional, Sequence, Union, cast

from meridian_geox import api
from meridian_geox import maps
from meridian_geox import report_text
import pandas as pd
from scipy import stats

# A holdback budget goal at least this share of all market spend over the test
# window probably describes a national budget rather than a test budget.
NATIONAL_BUDGET_SHARE = 0.9


@dataclasses.dataclass(frozen=True)
class DesignGoals:
  """Optional design goals.

  Attributes:
    target_mde: Largest acceptable relative MDE, in (0, 1).
    target_budget: Budget goal (positive).
    target_cpic: Target cost per incremental conversion (iCPA, positive).
  """

  target_mde: Optional[float] = None
  target_budget: Optional[float] = None
  target_cpic: Optional[float] = None

  def __post_init__(self):
    if self.target_mde is not None and not 0 < self.target_mde < 1:
      raise ValueError(f'target_mde must be in (0, 1), got {self.target_mde}.')
    if self.target_budget is not None and self.target_budget <= 0:
      raise ValueError(
          f'target_budget must be positive, got {self.target_budget}.'
      )
    if self.target_cpic is not None and self.target_cpic <= 0:
      raise ValueError(
          f'target_cpic must be positive, got {self.target_cpic}.'
      )

  @property
  def any_set(self) -> bool:
    """Whether any goal is set."""
    return any(
        v is not None
        for v in (self.target_mde, self.target_budget, self.target_cpic)
    )


# --------------------------------------------------------------------------
# Geo names.
# --------------------------------------------------------------------------


class GeoNames:
  """Looks up display names of geos."""

  def __init__(self, names: Optional[dict[str, str]] = None):
    self._names = dict(names or {})

  def __call__(self, location: Any) -> str:
    key = str(location)
    if key in self._names:
      return self._names[key]
    normalized = maps.normalize_geo_id(key)
    return self._names.get(normalized, key)

  def __len__(self) -> int:
    return len(self._names)


def resolve_geo_metadata(geo_metadata: Optional[pd.DataFrame]) -> GeoNames:
  """Builds a `GeoNames` lookup from a `location` + `geo_name`/`name` table."""
  if geo_metadata is None or geo_metadata.empty:
    return GeoNames()
  frame = _lower_columns(geo_metadata)
  name_column = next((c for c in ('geo_name', 'name') if c in frame), None)
  if 'location' not in frame or name_column is None:
    raise ValueError(
        'geo_metadata must have a `location` column and a `geo_name` (or '
        '`name`) column.'
    )
  names = {}
  for location, name in zip(frame['location'], frame[name_column]):
    if pd.isna(location) or pd.isna(name):
      continue
    names[str(location)] = str(name)
    names.setdefault(maps.normalize_geo_id(location), str(name))
  return GeoNames(names)


# --------------------------------------------------------------------------
# Design summaries.
# --------------------------------------------------------------------------


def experiment_types(design: api.Design) -> dict[str, api.ExperimentType]:
  """Returns the experiment type of each cell of a design."""
  config_types = (
      design.design_config.experiment_types if design.design_config else None
  )
  if isinstance(config_types, dict):
    types = dict(config_types)
  elif isinstance(config_types, api.ExperimentType):
    types = {cell: config_types for cell in design.designs}
  else:
    types = {}
  return {
      cell: types.get(cell, api.ExperimentType.HOLDBACK)
      for cell in sorted(design.designs, key=report_text.cell_index)
  }


def single_experiment_type(
    design: api.Design,
) -> Optional[api.ExperimentType]:
  """Returns the experiment type, or None for a multi-cell design."""
  types = experiment_types(design)
  return next(iter(types.values())) if len(types) == 1 else None


def group_labels(design: api.Design) -> dict[str, str]:
  """Returns the legend label of each treatment cell."""
  types = experiment_types(design)
  multi = len(types) > 1
  return {
      cell: report_text.treatment_label(cell, et, multi)
      for cell, et in types.items()
  }


def default_goals(design: api.Design, goals: DesignGoals) -> DesignGoals:
  """Fills unset budget and iCPA goals from the design's settings."""
  config, constraints = design.design_config, design.constraints
  if config is None or constraints is None:
    return goals
  # run_design normalizes the budgets and iCPAs to per-cell dicts.
  budget = cast(dict[str, api.Budget], constraints.budget_constraint)
  cpic = cast(dict[str, float], config.cost_per_incremental_conversion)
  holdback = [
      c
      for c, et in experiment_types(design).items()
      if et == api.ExperimentType.HOLDBACK
  ]
  budgets = [
      b for c in holdback if c in budget and (b := budget[c].budget or 0) > 0
  ]
  cpics = [cpic[c] for c in holdback if c in cpic and cpic[c] > 0]
  return DesignGoals(
      goals.target_mde,
      goals.target_budget or (float(sum(budgets)) if budgets else None),
      goals.target_cpic or (float(min(cpics)) if cpics else None),
  )


@dataclasses.dataclass(frozen=True)
class CellSummary:
  """Design metrics of one cell of one candidate."""

  cell: str
  experiment_type: api.ExperimentType
  label: str
  mde: float
  mde_abs: float
  budget: float
  implied_cpic: float
  aa_p_value: float
  r2: float
  treatment_geo_count: int
  treatment_share: float


@dataclasses.dataclass(frozen=True)
class CandidateSummary:
  """Summary of one candidate design, ranked against the goals."""

  design_id: str
  rank: int
  name: str
  cells: dict[str, CellSummary]
  control_geo_count: int
  total_spend: float
  max_mde: float
  # Largest iCPA at which the effect would still be detectable.
  max_detectable_cpic: float
  min_aa_p_value: float
  meets_mde: Optional[bool]
  meets_budget: Optional[bool]
  meets_cpic: Optional[bool]

  @property
  def feasible(self) -> bool:
    """Whether the candidate meets every set goal."""
    return not any(
        v is not None and not v
        for v in (self.meets_mde, self.meets_budget, self.meets_cpic)
    )

  @property
  def treatment_geo_count(self) -> int:
    """Number of treatment geos over all cells."""
    return sum(c.treatment_geo_count for c in self.cells.values())

  @property
  def total_mde_abs(self) -> float:
    """MDE in incremental conversions, summed over cells with a known MDE."""
    return sum(
        c.mde_abs for c in self.cells.values() if not math.isnan(c.mde_abs)
    )


def design_params(design: api.Design) -> tuple[float, float, api.TestType]:
  """Returns the design's (alpha, power, test type), with GeoX defaults."""
  config = design.design_config
  if config is None:
    return 0.1, 0.8, api.TestType.TWO_SIDED
  return float(config.alpha), float(config.power), config.test_type


def z_scores(design: api.Design) -> tuple[float, float]:
  """(critical z, z of the power) of a design."""
  alpha, power, test_type = design_params(design)
  tail = alpha / 2 if test_type == api.TestType.TWO_SIDED else alpha
  return float(stats.norm.ppf(1 - tail)), float(stats.norm.ppf(power))


def _cell_spend(
    cell: 'CellSummary', goals: DesignGoals, n_holdback: int
) -> float:
  """Spend behind one cell, from the budget goal or its spend change."""
  if (
      cell.experiment_type == api.ExperimentType.HOLDBACK
      and goals.target_budget is not None
      and n_holdback > 0
  ):
    return goals.target_budget / n_holdback
  return abs(cell.budget)


def _cell_spends(
    cells: dict[str, 'CellSummary'], goals: DesignGoals
) -> list[tuple['CellSummary', float]]:
  """(cell, spend behind it) for each cell with a usable absolute MDE."""
  n_holdback = sum(
      c.experiment_type == api.ExperimentType.HOLDBACK for c in cells.values()
  )
  return [
      (c, _cell_spend(c, goals, n_holdback))
      for c in cells.values()
      if not math.isnan(c.mde_abs) and c.mde_abs > 0
  ]


def _max_detectable_cpic(
    cells: dict[str, CellSummary], goals: DesignGoals
) -> float:
  """Largest iCPA at which every cell's effect is still detectable."""
  values = [spend / c.mde_abs for c, spend in _cell_spends(cells, goals)]
  return float(min(values)) if values else float('nan')


def design_summaries(
    design_set: api.DesignSet, goals: DesignGoals = DesignGoals()
) -> list[CandidateSummary]:
  """Ranks the candidates of a design set against the goals.

  Candidates that meet every set goal come first; ties keep the order of
  `design_metrics` (best first). Rank 1 is the recommended option.

  Args:
    design_set: Output of `run_design`.
    goals: Design goals (already defaulted with `default_goals`).

  Returns:
    One summary per candidate, ordered by rank.
  """
  # All candidates share the design config, so cell types and labels match.
  any_design = next(iter(design_set.designs.values()))
  types, labels = experiment_types(any_design), group_labels(any_design)
  summaries = []
  metrics = design_set.design_metrics
  for design_id, rows in metrics.groupby('design_id', sort=False):
    design = design_set.designs[str(design_id)]
    metric = {str(row['cell']): row for _, row in rows.iterrows()}
    cells = {
        cell: CellSummary(
            cell=cell,
            experiment_type=types[cell],
            label=labels[cell],
            mde=float(d.minimum_detectable_effect),
            mde_abs=float(metric[cell]['mde_abs']),
            budget=float(d.budget),
            implied_cpic=float(d.design_implied_cpic),
            aa_p_value=float(d.p_value),
            r2=float(metric[cell]['r2']),
            treatment_geo_count=len(d.treatment_geos),
            treatment_share=float(metric[cell]['treatment_conversions_pct'])
            / 100.0,
        )
        for cell, d in sorted(
            design.designs.items(), key=lambda kv: report_text.cell_index(kv[0])
        )
    }
    holdback = [
        c for c in cells.values()
        if c.experiment_type == api.ExperimentType.HOLDBACK
    ]
    max_mde = max(c.mde for c in cells.values())
    max_cpic = _max_detectable_cpic(cells, goals)
    meets_budget: Optional[bool] = None
    if goals.target_budget is not None and holdback:
      spend = sum(abs(c.budget) for c in holdback)
      meets_budget = spend <= goals.target_budget * (1 + 1e-9)
    meets_cpic: Optional[bool] = None
    if goals.target_cpic is not None and not math.isnan(max_cpic):
      meets_cpic = max_cpic >= goals.target_cpic * (1 - 1e-9)
    summaries.append(CandidateSummary(
        design_id=str(design_id),
        rank=0,
        name='',
        cells=cells,
        control_geo_count=len(design.control_geos),
        total_spend=float(sum(abs(c.budget) for c in cells.values())),
        max_mde=max_mde,
        max_detectable_cpic=max_cpic,
        min_aa_p_value=min(c.aa_p_value for c in cells.values()),
        meets_mde=(
            None if goals.target_mde is None else max_mde <= goals.target_mde
        ),
        meets_budget=meets_budget,
        meets_cpic=meets_cpic,
    ))
  ranked = sorted(summaries, key=lambda s: not s.feasible)
  return [
      dataclasses.replace(s, rank=rank, name=report_text.option_name(rank))
      for rank, s in enumerate(ranked, start=1)
  ]


# Chance of a conclusive result from which it is called High / Medium.
HIGH_CHANCE = 0.8
MEDIUM_CHANCE = 0.5


def chance_label(chance: Optional[float]) -> str:
  """Returns a plain-language label for a probability of success."""
  if chance is None or math.isnan(chance):
    return 'Unknown'
  if chance >= HIGH_CHANCE:
    return 'High'
  if chance >= MEDIUM_CHANCE:
    return 'Medium'
  return 'Low'


# --------------------------------------------------------------------------
# Geo tables.
# --------------------------------------------------------------------------


def _is_spend(column: str) -> bool:
  """Whether a (lowercased) column holds spend: `spend` or `spend_cell_N`."""
  return column == api.SPEND or bool(
      re.match(api.MULTICELL_SPEND_REGEX, column)
  )


def _pretest_totals(
    data: Optional[pd.DataFrame],
    excluded_dates: Iterable[Any] = (),
    before: Optional[pd.Timestamp] = None,
) -> dict[str, pd.Series]:
  """Pretest 'conversions' and 'spend' (all spend columns) per location."""
  totals = {
      'conversions': pd.Series(dtype=float), 'spend': pd.Series(dtype=float)
  }
  if data is None or data.empty:
    return totals
  frame = _lower_columns(data)
  if api.LOCATION not in frame:
    return totals
  if api.DATE in frame:
    dates = pd.to_datetime(frame[api.DATE])
    mask = ~dates.isin(pd.to_datetime(list(excluded_dates)))
    if before is not None:
      mask &= dates < before
    frame = frame[mask]
  location = frame[api.LOCATION].astype(str)
  for key, matches in (
      ('conversions', lambda c: c == api.CONVERSIONS),
      ('spend', _is_spend),
  ):
    columns = [c for c in frame.columns if matches(str(c))]
    if columns:
      total = frame[columns].astype(float).sum(axis=1)
      totals[key] = total.groupby(location).sum().astype(float)
  return totals


def pretest_volumes(
    data: Optional[pd.DataFrame],
    before: Optional[pd.Timestamp] = None,
    excluded_dates: Sequence[pd.Timestamp] = (),
) -> pd.Series:
  """Returns total conversions per location (optionally before a date)."""
  return _pretest_totals(data, excluded_dates, before)['conversions']


def pretest_spend(design: api.Design) -> pd.Series:
  """Returns total pretest spend per location (all spend columns)."""
  return _pretest_totals(design.data, design.excluded_dates or ())['spend']


GEO_TABLE_COLUMNS = [
    'location', 'geo_name', 'group', 'cell', 'conversions', 'share', 'spend',
    'spend_share', 'stratum',
]


def _share(values: pd.Series, in_test: pd.Series) -> pd.Series:
  """Share of the in-test total; 0 for excluded geos or a zero total."""
  total = float(values[in_test].sum())
  if total <= 0:
    return pd.Series(0.0, index=values.index)
  return (values / total).where(in_test, 0.0)


def geo_table(
    design: api.Design,
    names: GeoNames,
    volumes: Optional[pd.Series] = None,
) -> pd.DataFrame:
  """Returns one row per geo, sorted by group and descending share.

  Args:
    design: The design.
    names: Geo name lookup.
    volumes: Pretest conversions per location; defaults to the design data.

  Returns:
    A frame with the `GEO_TABLE_COLUMNS` columns.
  """
  totals = _pretest_totals(design.data, design.excluded_dates or ())
  if volumes is None:
    volumes = totals['conversions']
  labels = group_labels(design)
  rows = []
  for cell, per_cell in design.designs.items():
    rows += [
        (str(g), labels[cell], cell) for g in sorted(per_cell.treatment_geos)
    ]
  rows += [(str(g), 'Control', 'control') for g in sorted(design.control_geos)]
  seen = {row[0] for row in rows}
  rows += [
      (str(g), 'Excluded', 'excluded')
      for g in sorted(design.excluded_geos or ())
      if str(g) not in seen
  ]
  frame = pd.DataFrame(rows, columns=['location', 'group', 'cell'])
  if frame.empty:
    return frame.assign(
        geo_name=[], conversions=[], share=[], spend=[], spend_share=[],
        stratum=[],
    )
  in_test = frame['cell'] != 'excluded'
  frame['geo_name'] = [names(loc) for loc in frame['location']]
  frame['conversions'] = frame['location'].map(volumes).fillna(0.0)
  frame['share'] = _share(frame['conversions'], in_test)
  frame['spend'] = frame['location'].map(totals['spend']).fillna(0.0)
  frame['spend_share'] = _share(frame['spend'], in_test)
  strata = stratum_labels(design)
  frame['stratum'] = [strata.get(loc) for loc in frame['location']]
  order = [*labels.values(), 'Control', 'Excluded']
  rank = {group: i for i, group in enumerate(order)}
  frame = frame.sort_values(
      ['group', 'share'],
      ascending=[True, False],
      key=lambda column: column.map(rank) if column.name == 'group' else column,
  )
  return frame.reset_index(drop=True)[GEO_TABLE_COLUMNS]


def stratum_labels(design: api.Design) -> dict[str, int]:
  """Maps each geo to its 1-based stratum, or {} when unavailable."""
  labels = design.geo_stratum_labels
  if labels is None or design.data is None or design.data.empty:
    return {}
  values = [int(v) for v in labels.tolist()]
  # The labels follow geo-name order over the geos left after exclusions.
  frame = _lower_columns(design.data)
  excluded = {str(g) for g in design.excluded_geos or ()}
  geos = sorted({str(g) for g in frame[api.LOCATION]} - excluded)
  if len(geos) != len(values):
    return {}
  offset = 1 if min(values, default=1) == 0 else 0
  return {geo: value + offset for geo, value in zip(geos, values)}


@dataclasses.dataclass(frozen=True)
class AaDrift:
  """Cumulative observed minus predicted conversions in the A/A window."""

  frame: pd.DataFrame
  final: float
  start: pd.Timestamp
  end: pd.Timestamp


def aa_drift(design: api.Design, cell: str) -> Optional[AaDrift]:
  """A/A drift of one cell over the last test-length of pretest, or None."""
  per_cell = design.designs.get(cell)
  if per_cell is None or design.design_config is None:
    return None
  df = per_cell.counterfactual_conversions
  if df is None or df.empty:
    return None
  days = design.design_config.experiment_duration.days
  date_column = 'index' if 'index' in df.columns else str(df.columns[0])
  tail = df.tail(days)
  if days <= 0 or tail.empty:
    return None
  gap = (
      tail['observed'].astype(float) - tail['counterfactual'].astype(float)
  ).cumsum()
  dates = pd.to_datetime(tail[date_column])
  frame = pd.DataFrame({'date': list(dates), 'drift': list(gap)})
  return AaDrift(
      frame=frame,
      final=float(gap.iloc[-1]),
      start=pd.Timestamp(dates.iloc[0]),
      end=pd.Timestamp(dates.iloc[-1]),
  )


@dataclasses.dataclass(frozen=True)
class PlanNumbers:
  """What the planned spend buys for the bottleneck cell of a candidate."""

  spend: float
  cpic: float
  chance: float
  spend_for_power: float
  mde_abs: float
  power: float


def plan_numbers(
    summary: 'CandidateSummary', design: api.Design, goals: DesignGoals
) -> Optional[PlanNumbers]:
  """Plan numbers of the cell least likely to be conclusive, or None."""
  z_crit, z_power = z_scores(design)
  z_sum = z_crit + z_power
  if not 0 < z_sum < math.inf:
    return None
  power = float(stats.norm.cdf(z_power))
  best = None
  for cell, spend in _cell_spends(summary.cells, goals):
    cpic = goals.target_cpic or cell.implied_cpic
    if not cpic or math.isnan(cpic) or cpic <= 0:
      continue
    sigma = cell.mde_abs / z_sum
    expected = spend / cpic
    numbers = PlanNumbers(
        spend=spend,
        cpic=cpic,
        chance=float(stats.norm.cdf(expected / sigma - z_crit)),
        spend_for_power=cell.mde_abs * cpic,
        mde_abs=cell.mde_abs,
        power=power,
    )
    if best is None or numbers.chance < best.chance:
      best = numbers
  return best


def chance_at(
    summary: 'CandidateSummary',
    design: api.Design,
    spend: float,
    cpic: float,
) -> Optional[float]:
  """Lowest per-cell chance of a conclusive result at a spend and iCPA."""
  z_crit, z_power = z_scores(design)
  z_sum = z_crit + z_power
  if cpic <= 0 or not 0 < z_sum < math.inf:
    return None
  chances = [
      float(stats.norm.cdf(abs(spend / cpic) * z_sum / c.mde_abs - z_crit))
      for c in summary.cells.values()
      if not math.isnan(c.mde_abs) and c.mde_abs > 0
  ]
  return min(chances) if chances else None


def pretest_range(design: api.Design) -> Optional[
    tuple[pd.Timestamp, pd.Timestamp]
]:
  """First and last pretest date in the design data."""
  if design.data is None or design.data.empty:
    return None
  frame = _lower_columns(design.data)
  if api.DATE not in frame:
    return None
  dates = pd.to_datetime(frame[api.DATE])
  return pd.Timestamp(dates.min()), pd.Timestamp(dates.max())


# --------------------------------------------------------------------------
# Spend.
# --------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class SpendChange:
  """Planned spend change of one cell."""

  cell: str
  experiment_type: api.ExperimentType
  label: str
  amount: float
  # Planned relative change of treatment spend (-1.0 is -100%), or None.
  planned_change: Optional[float]


def spend_changes(design: api.Design) -> list[SpendChange]:
  """Returns the planned spend change of each cell."""
  types = experiment_types(design)
  labels = group_labels(design)
  budgets: dict[str, api.Budget] = {}
  if design.constraints is not None:
    raw = design.constraints.budget_constraint
    if isinstance(raw, api.Budget):
      budgets = {cell: raw for cell in design.designs}
    elif isinstance(raw, dict):
      budgets = raw
  changes = []
  for cell, et in types.items():
    amount = float(design.designs[cell].budget)
    planned = None
    if et in (api.ExperimentType.GO_DARK, api.ExperimentType.HEAVY_UP):
      budget = budgets.get(cell)
      if budget is not None and budget.budget_pct is not None:
        planned = float(budget.budget_pct)
      else:
        planned = -1.0 if et == api.ExperimentType.GO_DARK else 1.0
    changes.append(SpendChange(cell, et, labels[cell], amount, planned))
  return changes


def _lower_columns(data: pd.DataFrame) -> pd.DataFrame:
  frame = data.copy()
  frame.columns = [str(c).lower() for c in frame.columns]
  return frame


def market_spend(design: api.Design) -> Optional[float]:
  """Total spend across all geos over one test window, from pretest data."""
  if design.data is None or design.design_config is None:
    return None
  frame = _lower_columns(design.data)
  spend_cols = [
      c for c in frame.columns if c == api.SPEND or c.startswith('spend_cell_')
  ]
  if not spend_cols or api.DATE not in frame:
    return None
  daily = frame.groupby(api.DATE)[spend_cols].sum().sum(axis=1)
  if daily.empty:
    return None
  days = design.design_config.experiment_duration.days
  return float(daily.mean() * days)


def looks_national(design: api.Design, target_budget: Optional[float]) -> bool:
  """Whether a holdback budget goal looks like a national budget."""
  if target_budget is None:
    return False
  if single_experiment_type(design) != api.ExperimentType.HOLDBACK:
    return False
  total = market_spend(design)
  return bool(total) and target_budget >= NATIONAL_BUDGET_SHARE * total


# --------------------------------------------------------------------------
# Schedule.
# --------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class DesignChecks:
  """Pretest fit and A/A results of one candidate, judged against the config."""

  r2: float
  min_r2: float
  good_fit: bool
  alpha: float
  aa_p_value: float
  aa_passed: bool
  # (start, end) of the A/A drift window, if known.
  aa_window: Optional[tuple[pd.Timestamp, pd.Timestamp]]
  # Share of pretest conversions in treatment geos.
  treatment_share: float


def design_checks(
    summary: CandidateSummary, design: api.Design
) -> DesignChecks:
  """Returns the fit and A/A results shown in tiles, checks and tables."""
  r2s = [c.r2 for c in summary.cells.values()]
  r2 = math.nan if any(math.isnan(v) for v in r2s) else min(r2s)
  config = design.design_config
  minimum = float(config.min_r2) if config else 0.8
  alpha, _, _ = design_params(design)
  drift = aa_drift(design, next(iter(summary.cells)))
  return DesignChecks(
      r2=r2,
      min_r2=minimum,
      good_fit=not math.isnan(r2) and r2 >= minimum,
      alpha=alpha,
      aa_p_value=summary.min_aa_p_value,
      aa_passed=summary.min_aa_p_value >= alpha,
      aa_window=(drift.start, drift.end) if drift is not None else None,
      treatment_share=sum(c.treatment_share for c in summary.cells.values()),
  )


@dataclasses.dataclass(frozen=True)
class BudgetScope:
  """What a holdback budget goal means if it is a national budget."""

  budget: float
  treatment_budget: float
  share: float
  chance: Optional[float]


def budget_scope(
    summary: CandidateSummary,
    design: api.Design,
    budget: Optional[float],
    plan: Optional[PlanNumbers],
) -> Optional[BudgetScope]:
  """Treatment-geo part of a holdback budget goal, with its chance."""
  holdback = [
      c for c in summary.cells.values()
      if c.experiment_type == api.ExperimentType.HOLDBACK
  ]
  if budget is None or not holdback or plan is None:
    return None
  share = sum(c.treatment_share for c in holdback)
  if share <= 0:
    return None
  treatment_budget = budget * share
  return BudgetScope(
      budget,
      treatment_budget,
      share,
      chance_at(summary, design, treatment_budget, plan.cpic),
  )


def bau_spend(change: SpendChange) -> Optional[float]:
  """Treatment business-as-usual spend implied by a planned change."""
  if not change.planned_change:
    return None
  return abs(change.amount / change.planned_change)


def missed_goals(summary: CandidateSummary) -> list[str]:
  """Goals a candidate misses: 'MDE', 'budget' and/or 'iCPA'."""
  return [
      name
      for name, value in (
          ('MDE', summary.meets_mde),
          ('budget', summary.meets_budget),
          ('iCPA', summary.meets_cpic),
      )
      if value is not None and not value
  ]


@dataclasses.dataclass(frozen=True)
class LargestGeos:
  """How the largest geos split, and whether spend follows conversions."""

  count: int
  treated: int
  spend_tracks: Optional[int]


def largest_geos(table: pd.DataFrame, top: int = 22) -> LargestGeos:
  """Summarizes the `top` largest geos of a `geo_table`."""
  in_test = table[table['cell'] != 'excluded'].sort_values(
      'share', ascending=False
  ).head(top)
  tracks = in_test[
      (in_test['share'] > 0)
      & ((in_test['spend_share'] - in_test['share']).abs()
         <= 0.5 * in_test['share'])
  ]
  return LargestGeos(
      count=len(in_test),
      treated=int((~in_test['cell'].isin(['control'])).sum()),
      spend_tracks=(
          len(tracks) if float(in_test['spend_share'].sum()) > 0 else None
      ),
  )


def drift_ratio(drift: AaDrift, mde_abs: float) -> Optional[float]:
  """Final A/A drift as a share of the absolute MDE."""
  if not mde_abs or mde_abs <= 0:
    return None
  return abs(drift.final) / mde_abs


def planned_cpic(design: api.Design) -> Optional[float]:
  """iCPA the design was planned at (lowest across cells)."""
  config = design.design_config
  planned = config.cost_per_incremental_conversion if config else None
  if isinstance(planned, dict):
    values = [float(v) for v in planned.values() if v]
    planned = min(values) if values else None
  return float(planned) if planned else None


@dataclasses.dataclass(frozen=True)
class DataQuality:
  """Pretest data findings of a design.

  Attributes:
    flagged_geos: Geos flagged by the data quality checks.
    flagged_dates: Dates flagged by the data quality checks.
    user_geos: Geos the user excluded in `Constraints`.
    excluded_dates: Dates excluded from the design.
    user_dates: Excluded dates the checks did not flag.
    no_conversion_geos: Geos without conversions that are not excluded yet.
    spend_no_conversion: Flagged geos with spend but no conversions.
    geo_status: Status label of each pretest geo.
  """

  flagged_geos: list[str]
  flagged_dates: list[pd.Timestamp]
  user_geos: list[str]
  excluded_dates: list[pd.Timestamp]
  user_dates: list[pd.Timestamp]
  no_conversion_geos: list[str]
  spend_no_conversion: set[str]
  geo_status: dict[str, str]

  @property
  def excluded_geos(self) -> list[str]:
    """Flagged and user-excluded geos."""
    return sorted({*self.flagged_geos, *self.user_geos})


def data_quality(design: api.Design) -> DataQuality:
  """Collects the pretest data findings of a design."""
  quality = design.quality_check_result
  flagged_geos = sorted(
      str(g) for g in (quality.outlier_geos if quality else ())
  )
  flagged_dates = sorted(
      pd.Timestamp(d) for d in (quality.outlier_dates if quality else ())
  )
  user_geos = sorted(
      str(g)
      for g in (design.constraints.excluded_geos if design.constraints else ())
  )
  excluded_dates = sorted(
      pd.Timestamp(d) for d in (design.excluded_dates or ())
  )
  volumes = pretest_volumes(design.data)
  spend = pretest_spend(design)
  excluded = {*flagged_geos, *user_geos}
  status = {}
  for geo, total in volumes.items():
    geo = str(geo)
    if geo in flagged_geos:
      status[geo] = report_text.STATUS_FLAGGED
    elif total <= 0:
      status[geo] = report_text.STATUS_NO_CONVERSIONS
    elif geo in user_geos:
      status[geo] = report_text.STATUS_USER
    else:
      status[geo] = report_text.STATUS_OK
  return DataQuality(
      flagged_geos=flagged_geos,
      flagged_dates=flagged_dates,
      user_geos=user_geos,
      excluded_dates=excluded_dates,
      user_dates=[d for d in excluded_dates if d not in set(flagged_dates)],
      no_conversion_geos=sorted(
          str(g) for g, v in volumes.items() if v <= 0 and str(g) not in excluded
      ),
      spend_no_conversion={
          g for g in flagged_geos
          if spend.get(g, 0) > 0 and volumes.get(g, 0) <= 0
      },
      geo_status=status,
  )


# A calendar date, or None when unknown.
DateLike = Union[None, str, datetime.date, pd.Timestamp]


@dataclasses.dataclass(frozen=True)
class ExperimentSchedule:
  """Test and cooldown phases; dates are None when no start date is known."""

  test_days: int
  cooldown_days: int
  test_start: Optional[pd.Timestamp] = None
  # First test days used to ramp spend up; marked separately in the calendar.
  ramp_up_days: int = 0

  @property
  def dated(self) -> bool:
    """Whether a test start date is known."""
    return self.test_start is not None

  def _day(self, offset: int) -> Optional[pd.Timestamp]:
    """The day `offset` days after the test start, if dated."""
    if self.test_start is None:
      return None
    return self.test_start + pd.Timedelta(days=offset)

  @property
  def test_end(self) -> Optional[pd.Timestamp]:
    """Last test day, if dated."""
    return self._day(self.test_days - 1)

  @property
  def cooldown_start(self) -> Optional[pd.Timestamp]:
    """First cooldown day, if dated."""
    return self._day(self.test_days)

  @property
  def cooldown_end(self) -> Optional[pd.Timestamp]:
    """Last cooldown day, if dated."""
    return self._day(self.test_days + self.cooldown_days - 1)

  def days(self) -> pd.DataFrame:
    """One row per test/cooldown day with its phase."""
    if self.test_start is None:
      return pd.DataFrame(columns=['date', 'phase'])
    dates = pd.date_range(
        self.test_start, periods=self.test_days + self.cooldown_days
    )
    phases = (
        ['Ramp-up'] * self.ramp_up_days
        + ['Test'] * (self.test_days - self.ramp_up_days)
        + ['Cooldown'] * self.cooldown_days
    )
    return pd.DataFrame({'date': dates, 'phase': phases})


def experiment_schedule(
    design: api.Design,
    test_start: DateLike = None,
    cooldown_days: int = 14,
    ramp_up_days: int = 0,
) -> ExperimentSchedule:
  """Builds the test, ramp-up and cooldown schedule of a design.

  Args:
    design: The design.
    test_start: First test day, after the pretest window; None if unknown.
    cooldown_days: Cooldown length in days.
    ramp_up_days: Ramp-up days at the start of the test.

  Returns:
    The schedule.

  Raises:
    ValueError: If `cooldown_days` is negative, `ramp_up_days` is negative or
      not shorter than the test, or `test_start` is not after the last
      pretest date.
  """
  if cooldown_days < 0:
    raise ValueError(f'cooldown_days must be >= 0, got {cooldown_days}.')
  test_days = (
      design.design_config.experiment_duration.days
      if design.design_config
      else 0
  )
  start = None
  if test_start is not None:
    start = pd.Timestamp(test_start).normalize()
    pretest = pretest_range(design)
    if pretest and start <= pretest[1]:
      raise ValueError(
          f'test_start ({start.date()}) must be after the last pretest date '
          f'({pretest[1].date()}).'
      )
  if ramp_up_days < 0 or (test_days and ramp_up_days >= test_days):
    raise ValueError(
        f'ramp_up_days must be >= 0 and shorter than the {test_days}-day'
        f' test, got {ramp_up_days}.'
    )
  return ExperimentSchedule(
      test_days, int(cooldown_days), start, int(ramp_up_days)
  )
