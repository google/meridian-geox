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
from typing import Any, Iterable, Optional, Sequence, Union

from meridian_geox import api
from meridian_geox import maps
from meridian_geox import report_text
import pandas as pd
from scipy import stats

# A holdback budget goal at least this share of all market spend over the test
# window probably describes a national budget rather than a test budget.
NATIONAL_BUDGET_SHARE = 0.9
# Go-dark treatment geos may keep at most this share of their pretest spend.
GO_DARK_MAX_SPEND_SHARE = 0.1
# Relative tolerance when comparing observed vs. planned spend changes.
PLAN_TOLERANCE = 0.25


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
  """Builds a geo name lookup from a `location` + `geo_name`/`name` table.

  Locations are matched both as given and after `maps.normalize_geo_id`, so
  Google Ads DMA criteria IDs match numeric DMA codes.

  Args:
    geo_metadata: Optional display-name table.

  Returns:
    The name lookup.

  Raises:
    ValueError: If the table lacks the required columns.
  """
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
  short_id: str
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
  success_chance: Optional[float]

  @property
  def feasible(self) -> bool:
    return all(
        v is not False
        for v in (self.meets_mde, self.meets_budget, self.meets_cpic)
    )

  @property
  def recommended(self) -> bool:
    return self.rank == 1


def _z_scores(design: api.Design) -> tuple[float, float]:
  config = design.design_config
  alpha = config.alpha if config else 0.1
  power = config.power if config else 0.8
  two_sided = config is None or config.test_type == api.TestType.TWO_SIDED
  z_alpha = stats.norm.ppf(1 - alpha / 2 if two_sided else 1 - alpha)
  z_power = stats.norm.ppf(power)
  return float(z_alpha), float(z_power)


def detection_power(
    effect_abs: float, mde_abs: float, z_alpha: float, z_power: float
) -> float:
  """Power to detect an absolute effect, given the design MDE.

  The MDE is the effect detected with power `Φ(z_power)`, so the effect's
  standard error is `mde_abs / (z_alpha + z_power)`.

  Args:
    effect_abs: True absolute effect (incremental conversions).
    mde_abs: Absolute minimum detectable effect.
    z_alpha: Critical z-score of the test.
    z_power: z-score of the design power.

  Returns:
    The probability of a significant result.
  """
  if mde_abs <= 0:
    return float('nan')
  sigma = mde_abs / (z_alpha + z_power)
  return float(stats.norm.cdf(abs(effect_abs) / sigma - z_alpha))


def _metrics_by_cell(
    design_set: api.DesignSet, design_id: str
) -> dict[str, dict[str, Any]]:
  metrics = design_set.design_metrics
  if metrics is None or metrics.empty or 'design_id' not in metrics:
    return {}
  rows = metrics[metrics['design_id'].astype(str) == str(design_id)]
  return {str(row['cell']): row.to_dict() for _, row in rows.iterrows()}


def _summarize_cells(
    design_set: api.DesignSet, design_id: str, design: api.Design
) -> dict[str, CellSummary]:
  types = experiment_types(design)
  labels = group_labels(design)
  rows = _metrics_by_cell(design_set, design_id)
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
  """Spend that buys the effect of one cell.

  The budget goal is total campaign spend, split evenly across holdback
  cells; other cells use the design's planned spend change.

  Args:
    cell: The cell summary.
    goals: Defaulted design goals.
    n_holdback: Number of holdback cells in the candidate.

  Returns:
    The spend, as a positive amount.
  """
  if (
      cell.experiment_type == api.ExperimentType.HOLDBACK
      and goals.target_budget is not None
      and n_holdback > 0
  ):
    return goals.target_budget / n_holdback
  return abs(cell.budget)


def _holdback_count(cells: Iterable['CellSummary']) -> int:
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
    chance = None
    if goals.target_cpic is not None:
      z_alpha, z_power = _z_scores(design)
      chances = []
      n_holdback = _holdback_count(cells.values())
      for c in cells.values():
        if math.isnan(c.mde_abs) or c.mde_abs <= 0:
          continue
        spend = _cell_spend(c, goals, n_holdback)
        chances.append(
            detection_power(
                spend / goals.target_cpic, c.mde_abs, z_alpha, z_power
            )
        )
      chance = min(chances) if chances else None
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
            success_chance=chance,
        )
    )
  feasible = [
      d for d in drafts
      if all(d[k] is not False for k in ('meets_mde', 'meets_budget',
                                          'meets_cpic'))
  ]
  infeasible = [d for d in drafts if d not in feasible]
  summaries = []
  for rank, draft in enumerate(feasible + infeasible, start=1):
    summaries.append(
        CandidateSummary(
            rank=rank,
            name=report_text.option_name(rank),
            short_id=report_text.short_id(draft['design_id']),
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


# --------------------------------------------------------------------------
# Geo tables.
# --------------------------------------------------------------------------


def pretest_volumes(
    data: Optional[pd.DataFrame],
    before: Optional[pd.Timestamp] = None,
    excluded_dates: Sequence[pd.Timestamp] = (),
) -> pd.Series:
  """Returns total conversions per location (optionally before a date)."""
  if data is None or data.empty:
    return pd.Series(dtype=float)
  frame = _lower_columns(data)
  if api.CONVERSIONS not in frame or api.LOCATION not in frame:
    return pd.Series(dtype=float)
  if api.DATE in frame:
    dates = pd.to_datetime(frame[api.DATE])
    mask = ~dates.isin(pd.to_datetime(list(excluded_dates)))
    if before is not None:
      mask &= dates < before
    frame = frame[mask]
  return frame.groupby(frame[api.LOCATION].astype(str))[
      api.CONVERSIONS
  ].sum().astype(float)


def geo_table(
    design: api.Design,
    names: GeoNames,
    volumes: Optional[pd.Series] = None,
) -> pd.DataFrame:
  """Returns one row per geo with its group and pretest conversion share.

  Args:
    design: The design.
    names: Geo name lookup.
    volumes: Optional pretest conversions per location; defaults to the
      design's own data.

  Returns:
    A frame with `location`, `geo_name`, `group`, `cell`, `conversions`,
    `share` (of pretest conversions), `spend`, `spend_share` and `stratum`
    columns, sorted by group and descending share.
  """
  if volumes is None:
    volumes = pretest_volumes(
        design.data, excluded_dates=sorted(design.excluded_dates or ())
    )
  labels = group_labels(design)
  rows = []
  seen = set()
  for cell, per_cell in design.designs.items():
    for geo in sorted(per_cell.treatment_geos):
      rows.append((str(geo), labels[cell], cell))
      seen.add(str(geo))
  for geo in sorted(design.control_geos):
    rows.append((str(geo), 'Control', 'control'))
    seen.add(str(geo))
  for geo in sorted(design.excluded_geos or ()):
    if str(geo) not in seen:
      rows.append((str(geo), 'Excluded', 'excluded'))
  frame = pd.DataFrame(rows, columns=['location', 'group', 'cell'])
  if frame.empty:
    return frame.assign(
        geo_name=[], conversions=[], share=[], spend=[], spend_share=[],
        stratum=[],
    )
  frame['geo_name'] = [names(loc) for loc in frame['location']]
  conversions = pd.Series(
      [float(volumes.get(loc, 0.0)) for loc in frame['location']],
      index=frame.index,
      dtype=float,
  )
  frame['conversions'] = conversions
  in_test = frame['cell'] != 'excluded'
  total = float(conversions[in_test].sum())
  if total > 0:
    frame['share'] = (conversions / total).where(in_test, 0.0)
  else:
    frame['share'] = 0.0
  spend_by_geo = pretest_spend(design)
  spend = pd.Series(
      [float(spend_by_geo.get(loc, 0.0)) for loc in frame['location']],
      index=frame.index,
      dtype=float,
  )
  frame['spend'] = spend
  total_spend = float(spend[in_test].sum())
  if total_spend > 0:
    frame['spend_share'] = (spend / total_spend).where(in_test, 0.0)
  else:
    frame['spend_share'] = 0.0
  strata = stratum_labels(design)
  frame['stratum'] = [strata.get(loc) for loc in frame['location']]
  order = {label: i for i, label in enumerate(list(labels.values()) + [
      'Control', 'Excluded'])}
  frame['_order'] = frame['group'].map(order)
  frame = frame.sort_values(['_order', 'share'], ascending=[True, False])
  return frame.drop(columns='_order').reset_index(drop=True)[
      GEO_TABLE_COLUMNS
  ]


GEO_TABLE_COLUMNS = [
    'location', 'geo_name', 'group', 'cell', 'conversions', 'share', 'spend',
    'spend_share', 'stratum',
]


def pretest_spend(design: api.Design) -> pd.Series:
  """Returns total pretest spend per location (all spend columns)."""
  if design.data is None or design.data.empty:
    return pd.Series(dtype=float)
  frame = _lower_columns(design.data)
  columns = [
      c for c in frame.columns
      if c == api.SPEND or str(c).startswith('spend_cell_')
  ]
  if not columns or api.LOCATION not in frame:
    return pd.Series(dtype=float)
  if api.DATE in frame:
    dates = pd.to_datetime(frame[api.DATE])
    excluded = pd.to_datetime(sorted(design.excluded_dates or ()))
    frame = frame[~dates.isin(excluded)]
  total = frame[columns].astype(float).sum(axis=1)
  return total.groupby(frame[api.LOCATION].astype(str)).sum().astype(float)


def stratum_labels(design: api.Design) -> dict[str, int]:
  """Maps each geo to its stratum (1-based) for stratified designs.

  `Design.geo_stratum_labels` is ordered by geo name over the geos left after
  exclusions; this rebuilds that order from the design data.

  Args:
    design: The design.

  Returns:
    Location to stratum number, or an empty dict when unavailable.
  """
  labels = design.geo_stratum_labels
  if labels is None or design.data is None or design.data.empty:
    return {}
  values = [int(v) for v in labels.tolist()]
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
  """Returns the A/A drift of one cell over the last test-length of pretest.

  When nothing changed, treatment geos should stay close to the prediction
  from control geos; large drift relative to the MDE means noise alone could
  look like a lift.

  Args:
    design: The design.
    cell: The cell name.

  Returns:
    The drift, or None without pretest series.
  """
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

  cell: str
  spend: float
  cpic: float
  chance: float
  expected: float
  worst_case: float
  spend_for_power: float
  max_cpic: float
  mde_abs: float
  power: float


def plan_numbers(
    summary: 'CandidateSummary', design: api.Design, goals: DesignGoals
) -> Optional[PlanNumbers]:
  """Chance, expected and worst-case extra conversions at the goals.

  Uses the cell with the lowest chance of a conclusive result. Spend is the
  budget goal for holdback cells (else the design's spend change) and the
  iCPA is the target iCPA (else the design's implied iCPA).

  Args:
    summary: The candidate summary.
    design: The candidate's design.
    goals: Defaulted design goals.

  Returns:
    The plan numbers, or None without a usable MDE.
  """
  z_alpha, z_power = _z_scores(design)
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
    sigma = cell.mde_abs / (z_alpha + z_power)
    expected = spend / cpic
    numbers = PlanNumbers(
        cell=cell.cell,
        spend=spend,
        cpic=cpic,
        chance=float(stats.norm.cdf(expected / sigma - z_alpha)),
        expected=expected,
        worst_case=expected - z_alpha * sigma,
        spend_for_power=cell.mde_abs * cpic,
        max_cpic=spend / cell.mde_abs,
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
  if cpic <= 0:
    return None
  z_alpha, z_power = _z_scores(design)
  chances = [
      detection_power(spend / cpic, c.mde_abs, z_alpha, z_power)
      for c in summary.cells.values()
      if not math.isnan(c.mde_abs) and c.mde_abs > 0
  ]
  return min(chances) if chances else None


def alpha_and_sides(design: api.Design) -> tuple[float, str]:
  """Returns the design's alpha and `'one-sided'`/`'two-sided'`."""
  config = design.design_config
  if config is None:
    return 0.1, 'two-sided'
  sides = 'two-sided' if config.test_type == api.TestType.TWO_SIDED else (
      'one-sided'
  )
  return float(config.alpha), sides


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


def spend_column(cell: str, data: pd.DataFrame, cell_count: int) -> Optional[
    str
]:
  """Returns the spend column of a cell, if present."""
  columns = {str(c).lower() for c in data.columns}
  candidates = [f'spend_{cell}']
  if cell_count == 1:
    candidates.insert(0, api.SPEND)
  return next((c for c in candidates if c in columns), None)


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
class ExperimentSchedule:
  """Test and cooldown phases; dates are None when no start date is known."""

  test_days: int
  cooldown_days: int
  test_start: Optional[pd.Timestamp] = None
  # First test days used to ramp spend up; marked separately in the calendar.
  ramp_up_days: int = 0

  @property
  def dated(self) -> bool:
    return self.test_start is not None

  @property
  def test_end(self) -> Optional[pd.Timestamp]:
    if self.test_start is None:
      return None
    return self.test_start + pd.Timedelta(days=self.test_days - 1)

  @property
  def cooldown_end(self) -> Optional[pd.Timestamp]:
    if self.test_start is None:
      return None
    return self.test_start + pd.Timedelta(
        days=self.test_days + self.cooldown_days - 1
    )

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
    test_start: Union[None, str, datetime.date, pd.Timestamp] = None,
    cooldown_days: int = 14,
    ramp_up_days: int = 0,
) -> ExperimentSchedule:
  """Builds the experiment schedule.

  Args:
    design: The design.
    test_start: Planned first test day. Must be after the pretest window.
    cooldown_days: Planned cooldown length in days (0 means none).
    ramp_up_days: First test days used to ramp spend up (0 means none).

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
    if design.data is not None and api.DATE in _lower_columns(design.data):
      last = pd.to_datetime(_lower_columns(design.data)[api.DATE]).max()
      if start <= last:
        raise ValueError(
            f'test_start ({start.date()}) must be after the last pretest date '
            f'({last.date()}).'
        )
  if ramp_up_days < 0 or (test_days and ramp_up_days >= test_days):
    raise ValueError(
        f'ramp_up_days must be >= 0 and shorter than the {test_days}-day'
        f' test, got {ramp_up_days}.'
    )
  return ExperimentSchedule(
      test_days, int(cooldown_days), start, int(ramp_up_days)
  )


# --------------------------------------------------------------------------
# Analysis.
# --------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class AnalysisWindow:
  """Planned and observed analysis dates."""

  start: pd.Timestamp
  end: pd.Timestamp
  planned_days: int

  @property
  def observed_days(self) -> int:
    return int((self.end - self.start).days) + 1

  @property
  def planned_end(self) -> pd.Timestamp:
    return self.start + pd.Timedelta(days=self.planned_days - 1)

  @property
  def complete(self) -> bool:
    return self.observed_days >= self.planned_days

  @property
  def cooldown_days(self) -> int:
    return max(0, self.observed_days - self.planned_days)


def analysis_window(result: api.AnalysisResult) -> AnalysisWindow:
  """Returns the analysis window of a result."""
  config = result.analysis_config
  design_config = config.design.design_config
  start = pd.Timestamp(config.analysis_start_date)
  end = pd.Timestamp(config.analysis_end_date)
  planned = (
      design_config.experiment_duration.days
      if design_config
      else int((end - start).days) + 1
  )
  return AnalysisWindow(start, end, planned)


def confidence_level(result: api.AnalysisResult) -> float:
  """Returns the confidence level (1 - alpha) of an analysis."""
  alpha = result.analysis_config.alpha
  if alpha is None:
    design_config = result.analysis_config.design.design_config
    alpha = design_config.alpha if design_config else 0.1
  return 1.0 - float(alpha)


def estimate_consistent(estimate: api.Estimate) -> bool:
  """Whether the point estimate is finite and inside its CI."""
  point = estimate.point_estimate
  lower, upper = estimate.lower_bound, estimate.upper_bound
  if point is None or math.isnan(point) or math.isinf(point):
    return False
  if math.isnan(lower) or math.isnan(upper):
    return False
  p_value = estimate.p_value
  if p_value is None or not math.isfinite(p_value):
    return False
  tolerance = 1e-6 * max(1.0, abs(point))
  return lower - tolerance <= point <= upper + tolerance


def is_significant(estimate: api.Estimate, alpha: float) -> bool:
  """Whether the CI excludes zero and the p-value is below alpha."""
  excludes_zero = estimate.lower_bound > 0 or estimate.upper_bound < 0
  return excludes_zero and estimate.p_value < alpha


def verdict(
    metrics: api.AnalysisMetrics,
    window: AnalysisWindow,
    alpha: float,
    execution_ok: bool = True,
) -> report_text.Verdict:
  """Returns the plain-language verdict of one cell."""
  if not window.complete:
    return report_text.Verdict.TOO_EARLY
  if not estimate_consistent(metrics.lift) or not execution_ok:
    return report_text.Verdict.NEEDS_REVIEW
  if is_significant(metrics.lift, alpha):
    if metrics.lift.point_estimate > 0:
      return report_text.Verdict.YES
    return report_text.Verdict.NEEDS_REVIEW
  return report_text.Verdict.NO


@dataclasses.dataclass(frozen=True)
class Efficiency:
  """Marketer-facing efficiency estimate (iCPA or iROAS)."""

  label: str
  point: float
  lower: float
  upper: float
  is_money: bool


def efficiency(
    metrics: api.AnalysisMetrics, response_is_value: bool = False
) -> Optional[Efficiency]:
  """Converts iCPD into iCPA (or iROAS for value responses).

  iCPD is incremental conversions per unit of spend; iCPA is its reciprocal.
  Bounds at or below zero map to an infinite iCPA bound.

  Args:
    metrics: Cell metrics.
    response_is_value: Whether conversions are monetary value.

  Returns:
    The efficiency estimate, or None without spend data.
  """
  icpd = metrics.icpd
  if icpd is None or math.isnan(icpd.point_estimate):
    return None
  lower, upper = sorted((icpd.lower_bound, icpd.upper_bound))
  if response_is_value:
    return Efficiency('iROAS', icpd.point_estimate, lower, upper, False)

  def reciprocal(value: float) -> float:
    if math.isnan(value):
      return float('nan')
    return 1.0 / value if value > 0 else float('inf')

  return Efficiency(
      'iCPA',
      reciprocal(icpd.point_estimate),
      reciprocal(upper),
      reciprocal(lower),
      True,
  )


def implied_spend(metrics: api.AnalysisMetrics) -> Optional[float]:
  """Incremental spend implied by lift / iCPD."""
  icpd = metrics.icpd
  if icpd is None or not icpd.point_estimate:
    return None
  if math.isnan(icpd.point_estimate):
    return None
  return float(metrics.lift.point_estimate / icpd.point_estimate)


@dataclasses.dataclass(frozen=True)
class EffectSplit:
  """Effect during the test vs. during the cooldown."""

  test_effect: float
  cooldown_effect: float
  cooldown_days: int


def effect_split(
    metrics: api.AnalysisMetrics, window: AnalysisWindow
) -> Optional[EffectSplit]:
  """Splits the cumulative effect at the planned test end."""
  cumulative = metrics.cumulative_lift
  if window.cooldown_days <= 0 or cumulative is None or cumulative.empty:
    return None
  lift = cumulative['lift'].astype(float)
  lift.index = pd.to_datetime(lift.index)
  in_test = lift[lift.index <= window.planned_end]
  if in_test.empty or len(in_test) == len(lift):
    return None
  test_effect = float(in_test.iloc[-1])
  total = float(lift.iloc[-1])
  return EffectSplit(test_effect, total - test_effect, window.cooldown_days)


@dataclasses.dataclass(frozen=True)
class ExecutionCheck:
  """Observed vs. planned spend change of one cell."""

  cell: str
  label: str
  experiment_type: api.ExperimentType
  has_spend: bool
  pretest_daily_spend: float = float('nan')
  test_daily_spend: float = float('nan')
  observed_change: Optional[float] = None
  planned_change: Optional[float] = None
  verified: Optional[bool] = None
  issue_geos: tuple[str, ...] = ()

  @property
  def ok(self) -> bool:
    return not self.issue_geos and self.verified is not False


def execution_checks(
    result: api.AnalysisResult,
    data: Optional[pd.DataFrame],
    names: Optional[GeoNames] = None,
) -> list[ExecutionCheck]:
  """Checks whether treatment spend changed as planned in each cell.

  Args:
    result: The analysis result.
    data: Analysis input data with spend columns.
    names: Geo name lookup used to name problem geos.

  Returns:
    One check per cell.
  """
  names = names or GeoNames()
  design = result.analysis_config.design
  window = analysis_window(result)
  plans = {c.cell: c for c in spend_changes(design)}
  types = experiment_types(design)
  if data is None:
    return [
        ExecutionCheck(cell, plans[cell].label, et, has_spend=False)
        for cell, et in types.items()
    ]
  checks = []
  frame = _lower_columns(data)
  for cell, et in types.items():
    label = plans[cell].label
    column = spend_column(cell, frame, len(design.designs))
    if column is None:
      checks.append(ExecutionCheck(cell, label, et, has_spend=False))
      continue
    dates = pd.to_datetime(frame[api.DATE])
    pretest = frame[dates < window.start]
    test = frame[(dates >= window.start) & (dates <= min(
        window.end, window.planned_end))]
    geos = {str(g) for g in design.designs[cell].treatment_geos}
    pre_geo = pretest[pretest[api.LOCATION].astype(str).isin(geos)]
    test_geo = test[test[api.LOCATION].astype(str).isin(geos)]
    pre_days = max(pre_geo[api.DATE].nunique(), 1)
    test_days = max(test_geo[api.DATE].nunique(), 1)
    pre_by_geo = pre_geo.groupby(pre_geo[api.LOCATION].astype(str))[
        column
    ].sum() / pre_days
    test_by_geo = test_geo.groupby(test_geo[api.LOCATION].astype(str))[
        column
    ].sum() / test_days
    pre_daily = float(pre_by_geo.sum())
    test_daily = float(test_by_geo.sum())
    # Control geos keep business-as-usual spend, so their change over the
    # same windows captures seasonality; treatment changes are measured
    # relative to it (difference-in-differences on spend).
    control = {str(g) for g in design.control_geos}
    pre_control = _daily_spend(pretest, control, column)
    test_control = _daily_spend(test, control, column)
    trend = (
        test_control / pre_control
        if pre_control > 0 and test_control > 0
        else 1.0
    )
    observed = (
        test_daily / (pre_daily * trend) - 1.0 if pre_daily > 0 else None
    )

    issues = []
    for geo in sorted(geos):
      before = float(pre_by_geo.get(geo, 0.0)) * trend
      during = float(test_by_geo.get(geo, 0.0))
      if et == api.ExperimentType.GO_DARK:
        bad = during > GO_DARK_MAX_SPEND_SHARE * before and during > 0
      elif et == api.ExperimentType.HEAVY_UP:
        bad = during <= before
      else:
        bad = during <= 0
      if bad:
        issues.append(names(geo))

    planned = plans[cell].planned_change
    verified = None
    if planned is not None and observed is not None:
      tolerance = max(PLAN_TOLERANCE * abs(planned), 0.1)
      verified = abs(observed - planned) <= tolerance and not issues
    checks.append(
        ExecutionCheck(
            cell=cell,
            label=label,
            experiment_type=et,
            has_spend=True,
            pretest_daily_spend=pre_daily,
            test_daily_spend=test_daily,
            observed_change=observed,
            planned_change=planned,
            verified=verified,
            issue_geos=tuple(issues),
        )
    )
  return checks


def _daily_spend(frame: pd.DataFrame, geos: set[str], column: str) -> float:
  """Average daily total spend of `geos` in `frame`."""
  rows = frame[frame[api.LOCATION].astype(str).isin(geos)]
  days = rows[api.DATE].nunique()
  return float(rows[column].sum()) / days if days else 0.0


def daily_spend_by_group(
    result: api.AnalysisResult, data: pd.DataFrame, cell: str
) -> pd.DataFrame:
  """Daily total spend of a cell's treatment geos and of control geos."""
  design = result.analysis_config.design
  frame = _lower_columns(data)
  column = spend_column(cell, frame, len(design.designs))
  if column is None:
    return pd.DataFrame(columns=['date', 'group', 'spend'])
  labels = group_labels(design)
  treatment = {str(g) for g in design.designs[cell].treatment_geos}
  control = {str(g) for g in design.control_geos}
  locations = frame[api.LOCATION].astype(str)
  group = pd.Series(None, index=frame.index, dtype=object)
  group[locations.isin(control)] = 'Control'
  group[locations.isin(treatment)] = labels[cell]
  frame = frame.assign(group=group)
  frame = frame[frame['group'].notna()]
  frame[api.DATE] = pd.to_datetime(frame[api.DATE])
  out = frame.groupby([api.DATE, 'group'])[column].sum().reset_index()
  return out.rename(columns={api.DATE: 'date', column: 'spend'})


def robust_z_scores(data: pd.DataFrame, max_geos: int = 30) -> pd.DataFrame:
  """Per-geo median/MAD robust z-scores of daily conversions.

  Args:
    data: Data with `date`, `location` and `conversions`.
    max_geos: Number of largest geos to keep.

  Returns:
    Long frame with `date`, `location`, `conversions` and `z` columns.
  """
  frame = _lower_columns(data)
  frame[api.DATE] = pd.to_datetime(frame[api.DATE])
  frame[api.LOCATION] = frame[api.LOCATION].astype(str)
  totals = frame.groupby(api.LOCATION)[api.CONVERSIONS].sum()
  top = totals.sort_values(ascending=False).head(max_geos).index
  frame = frame[frame[api.LOCATION].isin(top)][
      [api.DATE, api.LOCATION, api.CONVERSIONS]
  ].copy()
  frame['z'] = frame.groupby(api.LOCATION)[api.CONVERSIONS].transform(
      _robust_z
  )
  return frame.reset_index(drop=True)


def _robust_z(values: pd.Series) -> pd.Series:
  """Median/MAD robust z-scores (0 when the MAD is 0)."""
  median = values.median()
  mad = (values - median).abs().median()
  scale = 1.4826 * mad
  if scale <= 0:
    return pd.Series(0.0, index=values.index)
  return (values - median) / scale


def weekly_quality(
    data: Optional[pd.DataFrame],
    excluded_geos: Sequence[str] = (),
    max_geos: int = 16,
) -> pd.DataFrame:
  """Weekly conversions per geo with zero-week flags and robust z-scores.

  Keeps the largest geos plus every geo that has a week without conversions
  or an excluded status, so tracking gaps are always visible.

  Args:
    data: Data with `date`, `location` and `conversions`.
    excluded_geos: Geos excluded from the design.
    max_geos: Number of largest geos to keep.

  Returns:
    Long frame with `week`, `location`, `conversions`, `zero` and `z`.
  """
  columns = ['week', 'location', 'conversions', 'zero', 'z']
  if data is None or data.empty:
    return pd.DataFrame(columns=columns)
  frame = _lower_columns(data)
  if not {api.DATE, api.LOCATION, api.CONVERSIONS} <= set(frame.columns):
    return pd.DataFrame(columns=columns)
  frame[api.DATE] = pd.to_datetime(frame[api.DATE])
  frame[api.LOCATION] = frame[api.LOCATION].astype(str)
  frame['week'] = frame[api.DATE].dt.to_period('W-SUN').dt.start_time
  weekly = (
      frame.groupby(['week', api.LOCATION])[api.CONVERSIONS]
      .sum()
      .reset_index()
  )
  weekly['zero'] = weekly[api.CONVERSIONS] <= 0
  totals = weekly.groupby(api.LOCATION)[api.CONVERSIONS].sum()
  keep = set(totals.sort_values(ascending=False).head(max_geos).index)
  keep |= set(weekly.loc[weekly['zero'], api.LOCATION])
  keep |= {str(g) for g in excluded_geos} & set(totals.index)
  weekly = weekly[weekly[api.LOCATION].isin(keep)].copy()
  weekly['z'] = weekly.groupby(api.LOCATION)[api.CONVERSIONS].transform(
      _robust_z
  )
  return weekly.rename(
      columns={api.LOCATION: 'location', api.CONVERSIONS: 'conversions'}
  ).reset_index(drop=True)[columns]
