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

"""Synthetic GeoX objects for reporting tests.

These helpers build small but realistic `DesignSet` and `AnalysisResult`
objects without running the (slow) design and analysis pipelines.
"""

import dataclasses
import datetime
from typing import Any, Mapping, Optional, Sequence, cast

from meridian_geox import api
import numpy as np
import pandas as pd

TEST_DAYS = 30
PRETEST_START = pd.Timestamp('2020-01-01')
PRETEST_DAYS = 90
DMA_CODES = (
    '501', '803', '602', '807', '623', '504', '511', '506', '618', '539',
    '524', '753',
)


def locations(n_geos: int, dma: bool = True) -> list[str]:
  """Returns geo locations: Google Ads DMA criteria IDs or names."""
  if dma:
    return [str(200000 + int(code)) for code in DMA_CODES[:n_geos]]
  return [f'City {i}' for i in range(1, n_geos + 1)]


def make_data(
    geos: Sequence[str],
    n_days: int = PRETEST_DAYS,
    start: pd.Timestamp = PRETEST_START,
    cells: int = 1,
    with_spend: bool = True,
    seed: int = 0,
) -> pd.DataFrame:
  """Returns daily geo data with conversions and spend."""
  rng = np.random.default_rng(seed)
  dates = pd.date_range(start, periods=n_days)
  rows = []
  for i, geo in enumerate(geos):
    base = 50 + 10 * (len(geos) - i)
    for date in dates:
      row = {
          api.DATE: date,
          api.LOCATION: geo,
          api.CONVERSIONS: float(base + rng.normal(0, 3)),
      }
      if with_spend:
        if cells == 1:
          row[api.SPEND] = float(base * 2)
        else:
          for c in range(1, cells + 1):
            row[f'spend_cell_{c}'] = float(base * 2)
      rows.append(row)
  return pd.DataFrame(rows)


def make_design(
    geos: Sequence[str],
    treatment: Mapping[str, Sequence[str]],
    experiment_types: Optional[dict[str, api.ExperimentType]] = None,
    mde: float = 0.02,
    budget: float = 10_000.0,
    data: Optional[pd.DataFrame] = None,
    target_budget: Optional[float] = 12_000.0,
    budget_pct: Optional[dict[str, float]] = None,
    excluded: Sequence[str] = (),
) -> api.Design:
  """Returns a design with the given treatment geos per cell."""
  if experiment_types is None:
    experiment_types = {cell: api.ExperimentType.HOLDBACK for cell in treatment}
  in_treatment = set()
  for gs in treatment.values():
    in_treatment.update(gs)
  control = {g for g in geos if g not in in_treatment and g not in excluded}
  config = api.DesignConfig(
      experiment_duration=datetime.timedelta(days=TEST_DAYS),
      experiment_types=dict(experiment_types),
      cell_count=len(treatment),
      cost_per_incremental_conversion={
          cell: 1.0
          for cell, et in experiment_types.items()
          if et == api.ExperimentType.HOLDBACK
      },
  )
  budgets = {}
  for cell in treatment:
    if budget_pct and cell in budget_pct:
      budgets[cell] = api.Budget(budget_pct=budget_pct[cell])
    elif (
        target_budget is not None
        and experiment_types[cell] == api.ExperimentType.HOLDBACK
    ):
      budgets[cell] = api.Budget(budget=target_budget)
  constraints = api.Constraints(budget_constraint=budgets or None)
  dates = pd.date_range(PRETEST_START, periods=PRETEST_DAYS)
  designs = {}
  for i, (cell, cell_geos) in enumerate(treatment.items()):
    sign = -1.0 if experiment_types[cell] == api.ExperimentType.GO_DARK else 1.0
    observed = 100 + 5 * np.sin(np.arange(PRETEST_DAYS) / 5) + i
    designs[cell] = api.PerCellDesign(
        treatment_geos=set(cell_geos),
        minimum_detectable_effect=mde * (1 + 0.1 * i),
        design_implied_cpic=1.0,
        p_value=0.5,
        budget=sign * budget,
        counterfactual_conversions=pd.DataFrame({
            'index': dates,
            'observed': observed,
            'counterfactual': observed * 0.99,
        }),
    )
  return api.Design(
      designs=designs,
      control_geos=control,
      excluded_geos=set(excluded),
      excluded_dates={PRETEST_START + pd.Timedelta(days=10)},
      design_config=config,
      constraints=constraints,
      data=data,
      quality_check_result=api.QualityCheckResult(
          quality_check_config=api.QualityCheckConfig(),
          quality_metrics=pd.DataFrame(),
          outlier_geos=set(excluded),
          outlier_dates={PRETEST_START + pd.Timedelta(days=10)},
      ),
  )


def with_strata(design: api.Design, labels: Sequence[int]) -> api.Design:
  """Returns a copy of `design` with stratum labels (ordered by geo name)."""
  return dataclasses.replace(
      design, geo_stratum_labels=cast(Any, np.array(labels))
  )


def make_design_set(
    n_candidates: int = 4,
    n_geos: int = 12,
    cells: int = 1,
    experiment_types: Optional[dict[str, api.ExperimentType]] = None,
    dma: bool = True,
    target_budget: Optional[float] = 12_000.0,
    budget_pct: Optional[dict[str, float]] = None,
) -> api.DesignSet:
  """Returns a design set with candidates of increasing MDE and budget."""
  geos = locations(n_geos, dma=dma)
  data = make_data(geos, cells=cells)
  designs = {}
  metrics = []
  for k in range(n_candidates):
    shift = k % (n_geos - cells * 2)
    treatment = {
        f'cell_{c}': [geos[(shift + 2 * (c - 1) + j) % n_geos] for j in (0, 1)]
        for c in range(1, cells + 1)
    }
    design_id = f'cand{k:04d}-0000-0000-0000-000000000000'
    mde = 0.015 + 0.005 * k
    budget = 8_000.0 + 2_500.0 * k
    design = make_design(
        geos,
        treatment,
        experiment_types=experiment_types,
        mde=mde,
        budget=budget,
        data=data,
        target_budget=target_budget,
        budget_pct=budget_pct,
    )
    design = with_strata(design, [i % 3 for i in range(n_geos)])
    designs[design_id] = design
    for cell, per_cell in design.designs.items():
      metrics.append({
          'design_id': design_id,
          'cell': cell,
          'design_methodology': 'STRATIFIED_SAMPLING-TBR',
          'r2': 0.9 - 0.01 * k,
          'mde': per_cell.minimum_detectable_effect,
          'mde_abs': budget,
          'p_value (AA)': per_cell.p_value,
          'budget': per_cell.budget,
          'design_implied_cpic': per_cell.design_implied_cpic,
          'treatment_conversions_pct': 20.0,
          'treatment_geo_count': len(per_cell.treatment_geos),
      })
  return api.DesignSet(designs=designs, design_metrics=pd.DataFrame(metrics))
