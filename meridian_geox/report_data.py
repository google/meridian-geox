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
from typing import Iterable, Optional

from meridian_geox import api
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
  """Fills unset goals from the design's constraints and config."""
  target_budget = goals.target_budget
  if target_budget is None and design.constraints is not None:
    budgets = design.constraints.budget_constraint
    if isinstance(budgets, api.Budget):
      budgets = {cell: budgets for cell in design.designs}
    if isinstance(budgets, dict):
      types = experiment_types(design)
      values = [
          b.budget
          for cell, b in budgets.items()
          if cell in design.designs
          and b.budget is not None
          and types.get(cell) == api.ExperimentType.HOLDBACK
      ]
      if values:
        target_budget = float(sum(values))
  target_cpic = goals.target_cpic
  if target_cpic is None and design.design_config is not None:
    cpics = design.design_config.cost_per_incremental_conversion
    if isinstance(cpics, dict) and cpics:
      target_cpic = float(min(cpics.values()))
    elif isinstance(cpics, (int, float)):
      target_cpic = float(cpics)
  if target_budget is not None and target_budget <= 0:
    target_budget = None
  if target_cpic is not None and target_cpic <= 0:
    target_cpic = None
  return DesignGoals(goals.target_mde, target_budget, target_cpic)


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
  def recommended(self) -> bool:
    """Whether this is the recommended (rank 1) candidate."""
    return self.rank == 1


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


def _summarize_cells(
    design_set: api.DesignSet, design_id: str, design: api.Design
) -> dict[str, CellSummary]:
  """Per-cell summaries of one candidate, from its design and metrics rows."""
  types = experiment_types(design)
  labels = group_labels(design)
  metrics = design_set.design_metrics
  rows = {}
  if metrics is not None and not metrics.empty and 'design_id' in metrics:
    selected = metrics[metrics['design_id'].astype(str) == str(design_id)]
    rows = {str(row['cell']): row.to_dict() for _, row in selected.iterrows()}
  cells = {}
  for cell, per_cell in sorted(
      design.designs.items(), key=lambda kv: report_text.cell_index(kv[0])
  ):
    row = rows.get(cell, {})
    mde_abs = float(row.get('mde_abs', float('nan')))
    cells[cell] = CellSummary(
        cell=cell,
        experiment_type=types[cell],
        label=labels[cell],
        mde=float(per_cell.minimum_detectable_effect),
        mde_abs=mde_abs,
        budget=float(per_cell.budget),
        implied_cpic=float(per_cell.design_implied_cpic),
        aa_p_value=float(per_cell.p_value),
        r2=float(row.get('r2', float('nan'))),
        treatment_geo_count=len(per_cell.treatment_geos),
        treatment_share=float(row.get('treatment_conversions_pct', 0.0))
        / 100.0,
    )
  return cells


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

  Candidates that meet every set goal come first; ties keep the design set's
  order (smallest MDE first). Rank 1 is the recommended option.

  Args:
    design_set: Output of `run_design`.
    goals: Design goals (already defaulted with `default_goals`).

  Returns:
    One summary per candidate, ordered by rank.
  """
  order = []
  if design_set.design_metrics is not None and not (
      design_set.design_metrics.empty
  ):
    order = [
        str(d)
        for d in pd.unique(design_set.design_metrics['design_id'].astype(str))
    ]
  order += [str(d) for d in design_set.designs if str(d) not in order]
  designs = {str(k): v for k, v in design_set.designs.items()}

  drafts = []
  for design_id in order:
    if design_id not in designs:
      continue
    design = designs[design_id]
    cells = _summarize_cells(design_set, design_id, design)
    if not cells:
      continue
    max_mde = max(c.mde for c in cells.values())
    total_spend = float(sum(abs(c.budget) for c in cells.values()))
    max_cpic = _max_detectable_cpic(cells, goals)
    meets_mde = None if goals.target_mde is None else (
        max_mde <= goals.target_mde
    )
    holdback_spend = sum(
        abs(c.budget)
        for c in cells.values()
        if c.experiment_type == api.ExperimentType.HOLDBACK
    )
    has_holdback = any(
        c.experiment_type == api.ExperimentType.HOLDBACK
        for c in cells.values()
    )
    meets_budget = None
    if goals.target_budget is not None and has_holdback:
      meets_budget = holdback_spend <= goals.target_budget * (1 + 1e-9)
    meets_cpic = None
    if goals.target_cpic is not None and not math.isnan(max_cpic):
      meets_cpic = max_cpic >= goals.target_cpic * (1 - 1e-9)
    drafts.append(
        dict(
            design_id=design_id,
            cells=cells,
            control_geo_count=len(design.control_geos),
            total_spend=total_spend,
            max_mde=max_mde,
            max_detectable_cpic=max_cpic,
            min_aa_p_value=min(c.aa_p_value for c in cells.values()),
            meets_mde=meets_mde,
            meets_budget=meets_budget,
            meets_cpic=meets_cpic,
        )
    )
  goal_keys = ('meets_mde', 'meets_budget', 'meets_cpic')
  feasible = [
      d
      for d in drafts
      if not any(d[k] is not None and not d[k] for k in goal_keys)
  ]
  infeasible = [d for d in drafts if d not in feasible]
  summaries = []
  for rank, draft in enumerate(feasible + infeasible, start=1):
    summaries.append(
        CandidateSummary(
            rank=rank,
            name=report_text.option_name(rank),
            **draft,
        )
    )
  return summaries


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


def alpha_and_sides(design: api.Design) -> tuple[float, str]:
  """Returns the design's alpha and `'one-sided'`/`'two-sided'`."""
  alpha, _, test_type = design_params(design)
  sides = (
      'two-sided' if test_type == api.TestType.TWO_SIDED else 'one-sided'
  )
  return alpha, sides


def min_r2(design: api.Design) -> float:
  """Returns the design's minimum pretest-fit R²."""
  return float(design.design_config.min_r2) if design.design_config else 0.8


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
  minimum = min_r2(design)
  alpha, _ = alpha_and_sides(design)
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
