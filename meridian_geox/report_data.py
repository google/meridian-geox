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
import math
from typing import Any, Iterable, Optional, cast

from meridian_geox import api
from meridian_geox import maps
from meridian_geox import report_text
import pandas as pd
from scipy import stats


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
  frame = geo_metadata.copy()
  frame.columns = [str(c).lower() for c in frame.columns]
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


def z_alpha(alpha: float, test_type: api.TestType) -> float:
  """Critical z-value of a one- or two-sided test at `alpha`."""
  tail = alpha / 2 if test_type == api.TestType.TWO_SIDED else alpha
  return float(stats.norm.ppf(1 - tail))


def _z_scores(design: api.Design) -> tuple[float, float]:
  alpha, power, test_type = design_params(design)
  return z_alpha(alpha, test_type), float(stats.norm.ppf(power))


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


def _holdback_count(cells: Iterable['CellSummary']) -> int:
  """Number of holdback cells."""
  return sum(
      1 for c in cells if c.experiment_type == api.ExperimentType.HOLDBACK
  )


def _max_detectable_cpic(
    cells: dict[str, CellSummary], goals: DesignGoals
) -> float:
  """Largest iCPA at which every cell's effect is still detectable."""
  values = []
  n_holdback = _holdback_count(cells.values())
  for cell in cells.values():
    if not cell.mde_abs or math.isnan(cell.mde_abs) or cell.mde_abs <= 0:
      continue
    values.append(_cell_spend(cell, goals, n_holdback) / cell.mde_abs)
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


def chance_label(chance: Optional[float]) -> str:
  """Returns a plain-language label for a probability of success."""
  if chance is None or math.isnan(chance):
    return 'Unknown'
  if chance >= 0.8:
    return 'High'
  if chance >= 0.5:
    return 'Medium'
  return 'Low'


@dataclasses.dataclass(frozen=True)
class AaDrift:
  """Cumulative observed minus predicted conversions in the A/A window."""

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
  dates = pd.to_datetime(tail[date_column])
  return AaDrift(
      start=pd.Timestamp(dates.iloc[0]),
      end=pd.Timestamp(dates.iloc[-1]),
  )


@dataclasses.dataclass(frozen=True)
class PlanNumbers:
  """What the planned spend buys for the bottleneck cell of a candidate."""

  spend: float
  cpic: float
  chance: float
  mde_abs: float
  power: float


def plan_numbers(
    summary: 'CandidateSummary', design: api.Design, goals: DesignGoals
) -> Optional[PlanNumbers]:
  """Plan numbers of the cell least likely to be conclusive, or None."""
  z_crit, z_power = _z_scores(design)
  z_sum = z_crit + z_power
  if not 0 < z_sum < math.inf:
    return None
  power = float(stats.norm.cdf(z_power))
  best = None
  n_holdback = _holdback_count(summary.cells.values())
  for cell in summary.cells.values():
    if math.isnan(cell.mde_abs) or cell.mde_abs <= 0:
      continue
    cpic = goals.target_cpic or cell.implied_cpic
    if not cpic or math.isnan(cpic) or cpic <= 0:
      continue
    spend = _cell_spend(cell, goals, n_holdback)
    sigma = cell.mde_abs / z_sum
    expected = spend / cpic
    numbers = PlanNumbers(
        spend=spend,
        cpic=cpic,
        chance=float(stats.norm.cdf(expected / sigma - z_crit)),
        mde_abs=cell.mde_abs,
        power=power,
    )
    if best is None or numbers.chance < best.chance:
      best = numbers
  return best


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


def spend_changes(
    design: api.Design, design_metrics_budget: Optional[dict[str, float]] = None
) -> list[SpendChange]:
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
    if design_metrics_budget and cell in design_metrics_budget:
      amount = design_metrics_budget[cell]
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
