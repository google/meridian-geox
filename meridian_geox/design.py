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

"""GeoX design library API."""

import dataclasses
import logging
from typing import Any, Callable, Optional
import uuid

import jax
import jax.numpy as jnp
from meridian_geox import api
from meridian_geox import generate_candidates
from meridian_geox import util
from meridian_geox.data_quality import data_quality
from meridian_geox.methodology import tbr
import numpy as np
import pandas as pd
from scipy import stats


@dataclasses.dataclass
class ProcessedData:
  """Processed data for experiment design."""

  selection_train: api.JnpArray
  selection_eval: api.JnpArray
  estimation_train: api.JnpArray
  estimation_eval: api.JnpArray
  # Training period to generate candidates.
  training_period: list[api.Timestamp]
  # Filtered data used for design.
  filtered_data: pd.DataFrame
  # Spend data for slope check.
  selection_train_spend: Optional[api.JnpArray] = None
  # Evaluation spend data for budget calculation.
  estimation_eval_spend: Optional[api.JnpArray] = None


@dataclasses.dataclass
class ScoredCandidates:
  """Scored candidates for experiment design."""

  candidates: jnp.ndarray
  mde_abs: jnp.ndarray
  mde_pct: jnp.ndarray
  p_values: jnp.ndarray
  r2_scores: jnp.ndarray
  observed_conversions: jnp.ndarray
  counterfactual_conversions: jnp.ndarray


def prepare_data(
    data: pd.DataFrame,
    experiment_duration: int,
    constraints: api.Constraints,
) -> ProcessedData:
  """Processes data for experiment design."""
  # Filter excluded geos.
  if constraints.excluded_geos:
    data = data[~data[api.LOCATION].isin(list(constraints.excluded_geos))]

  # Filter excluded dates.
  if constraints.excluded_dates:
    data = data[~data[api.DATE].isin(list(constraints.excluded_dates))]

  # T0: Training period (for selecting designs).
  # T1: Validation period (for selecting designs).
  # T2: Estimation period (for calculating MDE).
  # For selection phase, we train on T0 and eval on T1.
  # For estimation phase, we train on T0 + T1 and eval on T2.
  pivoted_data = util.pivot_and_sort_data(data, api.CONVERSIONS)

  n_dates = len(pivoted_data)

  t2_start_idx = n_dates - experiment_duration
  t1_start_idx = t2_start_idx - experiment_duration

  t0_data = pivoted_data.iloc[:t1_start_idx]
  t1_data = pivoted_data.iloc[t1_start_idx:t2_start_idx]
  t2_data = pivoted_data.iloc[t2_start_idx:]

  estimation_train_data = pivoted_data.iloc[:t2_start_idx]

  selection_train_spend = None
  estimation_eval_spend = None
  if api.SPEND in data.columns:
    pivoted_spend = util.pivot_and_sort_data(data, api.SPEND)
    t0_spend = pivoted_spend.iloc[:t1_start_idx]
    t2_spend = pivoted_spend.iloc[t2_start_idx:]
    selection_train_spend = jnp.array(t0_spend.values)
    estimation_eval_spend = jnp.array(t2_spend.values)

  return ProcessedData(
      selection_train=jnp.array(t0_data.values),
      selection_eval=jnp.array(t1_data.values),
      estimation_train=jnp.array(estimation_train_data.values),
      estimation_eval=jnp.array(t2_data.values),
      training_period=t0_data.index.tolist(),
      filtered_data=data,
      selection_train_spend=selection_train_spend,
      estimation_eval_spend=estimation_eval_spend,
  )


@dataclasses.dataclass
class MdeParams:
  """Parameters for MDE calculation."""

  top_candidates: jnp.ndarray
  keys: jax.Array
  z_score_sum: float
  top_indices: jnp.ndarray


def _get_params_for_mde_calculation(
    design_config: api.DesignConfig,
    r2_scores: jnp.ndarray,
    candidates: jnp.ndarray,
    key: jax.Array,
) -> MdeParams:
  """Prepares parameters for MDE calculation."""
  # 1. Select top candidates based on out of sample R2.
  sorted_indices = jnp.argsort(r2_scores)[::-1]
  n_top = min(design_config.n_ranked_candidates, len(r2_scores))
  top_indices = sorted_indices[:n_top]
  top_candidates = candidates[top_indices]

  # 2. Generate keys for A/A testing.
  keys = jax.random.split(key, n_top)

  # 3. Calculate Z-score sum for MDE calculation.
  if design_config.test_type == api.TestType.TWO_SIDED:
    z_alpha = stats.norm.ppf(1 - design_config.alpha / 2)
  else:
    z_alpha = stats.norm.ppf(1 - design_config.alpha)

  z_power = stats.norm.ppf(design_config.power)
  z_score_sum = z_alpha + z_power

  return MdeParams(
      top_candidates=top_candidates,
      keys=keys,
      z_score_sum=z_score_sum,
      top_indices=top_indices,
  )


def _filter_results_by_aa_test(
    scored_candidates: ScoredCandidates,
    design_config: api.DesignConfig,
) -> ScoredCandidates:
  """Filters designs based on AA test results."""
  is_valid_aa = scored_candidates.p_values >= design_config.alpha

  n_passing = int(jnp.sum(is_valid_aa))

  if n_passing == 0:
    raise ValueError('No designs passed the A/A test (p >= alpha).')

  if n_passing < design_config.design_output_count:
    logging.warning(
        'Only %d designs passed the A/A test. Returning all of them.',
        n_passing,
    )

  return ScoredCandidates(
      candidates=scored_candidates.candidates[is_valid_aa],
      mde_abs=scored_candidates.mde_abs[is_valid_aa],
      mde_pct=scored_candidates.mde_pct[is_valid_aa],
      p_values=scored_candidates.p_values[is_valid_aa],
      r2_scores=scored_candidates.r2_scores[is_valid_aa],
      observed_conversions=scored_candidates.observed_conversions[is_valid_aa],
      counterfactual_conversions=scored_candidates.counterfactual_conversions[
          is_valid_aa
      ],
  )


def _get_design_summary(
    scored_candidates: ScoredCandidates,
    design_config: api.DesignConfig,
    constraints: api.Constraints,
    geos: list[str],
    geo_stratum_labels: jnp.ndarray,
    processed_data: ProcessedData,
    data: pd.DataFrame,
) -> api.DesignSet:
  """Converts the results to a DesignSet."""
  designs = {}
  metrics_list = []

  # Convert JAX arrays to NumPy.
  top_candidates_np = np.array(scored_candidates.candidates)
  mde_abs_np = np.array(scored_candidates.mde_abs)
  mde_pct_np = np.array(scored_candidates.mde_pct)
  p_values_np = np.array(scored_candidates.p_values)
  r2_scores_np = np.array(scored_candidates.r2_scores)
  observed_conversions_np = np.array(scored_candidates.observed_conversions)
  counterfactual_conversions_np = np.array(
      scored_candidates.counterfactual_conversions
  )
  estimation_eval_np = np.array(processed_data.estimation_eval)
  estimation_eval_spend_np = (
      np.array(processed_data.estimation_eval_spend)
      if processed_data.estimation_eval_spend is not None
      else None
  )

  is_go_dark_or_heavy_up = util.is_go_dark_or_heavy_up(
      design_config.experiment_types
  )

  # Get full dates for plotting.
  pivoted_data = util.pivot_and_sort_data(data, api.CONVERSIONS)
  full_dates = pivoted_data.index

  for i in range(len(top_candidates_np)):
    mask = top_candidates_np[i]
    treatment_geos_dict = {}
    control_geos = set()

    # Identify control geos (mask == 0).
    control_indices = np.where(mask == 0)[0]
    for idx in control_indices:
      control_geos.add(geos[idx])

    # Identify treatment geos for each cell.
    for cell_id in range(1, design_config.cell_count + 1):
      cell_indices = np.where(mask == cell_id)[0]
      treatment_geos_dict[f'cell_{cell_id}'] = {
          geos[idx] for idx in cell_indices
      }

    # Identify all treated units (mask > 0).
    all_treated_indices = np.where(mask > 0)[0]

    # Calculate budget.
    # 1. For GO_DARK and HEAVY_UP:
    #    - If constraints.budget is set, use it.
    #    - If constraints.budget_percent is set, use
    #      treatment_geo_cost * budget_percent.
    #    - Otherwise, use treatment_geo_cost.
    #    - Defaults to 0 if spend data is missing.
    # 2. For HOLDBACK:
    #    - Use MDE * treatment_conversion_volume * CPIC.
    # TODO: Add output estimated cpic for go dark and heavy up
    # studies.
    if is_go_dark_or_heavy_up:
      if estimation_eval_spend_np is None:
        logging.warning(
            'Spend data missing for GO_DARK or HEAVY_UP experiment. '
            'Setting budget to 0.'
        )
        required_budget = 0.0
      elif constraints.budget is not None:
        required_budget = constraints.budget
      else:
        treatment_geo_cost = float(
            np.sum(estimation_eval_spend_np[:, all_treated_indices])
        )
        if constraints.budget_percent is not None:
          required_budget = treatment_geo_cost * constraints.budget_percent
        else:
          required_budget = treatment_geo_cost
    else:
      treatment_conversion_volume = np.sum(
          estimation_eval_np[:, all_treated_indices]
      )
      required_budget = (
          mde_pct_np[i]
          * treatment_conversion_volume
          * design_config.cost_per_incremental_conversion
      )
      if (
          constraints.budget is not None
          and constraints.budget < required_budget
      ):
        logging.warning(
            'Budget input (%.2f) is less than required budget (%.2f).',
            constraints.budget,
            required_budget,
        )

    design_id = str(uuid.uuid4())

    # TODO: Add multicell support for metrics.
    # Currently only a single metric is calculated for the whole design.
    # We assign this metric to all cells for now.
    cell_designs = {}
    for cell_name, treatment_geos in treatment_geos_dict.items():
      cell_designs[cell_name] = api.PerCellDesign(
          treatment_geos=treatment_geos,
          minimum_detectable_effect=float(mde_pct_np[i]),
          p_value=float(p_values_np[i]),
          budget=float(required_budget),
          counterfactual_conversions=pd.DataFrame({
              'date': full_dates,
              'observed': observed_conversions_np[i],
              'counterfactual': counterfactual_conversions_np[i],
          }),
      )

    design_obj = api.Design(
        designs=cell_designs,
        control_geos=control_geos,
        excluded_geos=constraints.excluded_geos,
        design_config=design_config,
        constraints=constraints,
        geo_stratum_labels=geo_stratum_labels,
        data=data,
    )
    designs[design_id] = design_obj

    metrics_list.append({
        'design_id': design_id,
        'cell_id': 'cell_1',
        'design_methodology': (
            f'{design_config.geo_assignment_rule.name}-'
            f'{design_config.methodology.name}'
        ),
        'r2': r2_scores_np[i],
        'mde_abs': mde_abs_np[i],
        'mde_pct': mde_pct_np[i],
        'p_value': p_values_np[i],
        'budget': required_budget,
    })
  design_metrics = pd.DataFrame(metrics_list)

  # TODO: Add more design metrics to rank the designs.
  design_metrics = design_metrics.sort_values(by='mde_pct').head(
      design_config.design_output_count
  ).reset_index(drop=True)
  designs = {
      design_id: designs[design_id] for design_id in design_metrics['design_id']
  }

  # TODO: Populate design_data.
  design_data = pd.DataFrame()

  return api.DesignSet(
      designs=designs,
      design_metrics=design_metrics,
      design_data=design_data,
  )


def run_design(
    data: pd.DataFrame,
    design_config: api.DesignConfig,
    constraints: api.Constraints,
    # An option to enable and configure automatic data quality checks.
    data_quality_check_config: data_quality.QualityCheckConfig = (
        data_quality.QualityCheckConfig()
    ),
    # A custom design scorer that allows power users to rank designs based on
    # their own criteria. API will be specified later.
    design_scorer: Optional[Callable[..., Any]] = None,
) -> api.DesignSet:
  """Designs GeoX experiments."""
  del data_quality_check_config, design_scorer  # Unused in skeleton.

  # TODO: Complete the design method following the steps below.
  # 1. Preprocess data.
  error_messages: list[str] = util.validate_design_input(
      data, design_config, constraints
  )
  if error_messages:
    raise ValueError(f'Data validation failed: {error_messages}')

  # TODO: Run data quality checks.

  processed_data: ProcessedData = prepare_data(
      data, design_config.experiment_duration, constraints
  )

  key = jax.random.PRNGKey(design_config.seed)
  candidates_key, mde_key = jax.random.split(key)

  # 2. Generate candidates.
  geo_stratum_labels = None
  if design_config.geo_assignment_rule == api.GeoAssignmentRule.RANDOM:
    candidates: jnp.ndarray = generate_candidates.get_random_candidates(
        filtered_data=processed_data.filtered_data,
        design_config=design_config,
        constraints=constraints,
        key=candidates_key,
        selection_train=processed_data.selection_train,
        selection_train_spend=processed_data.selection_train_spend,
    )
  elif (
      design_config.geo_assignment_rule
      == api.GeoAssignmentRule.STRATIFIED_SAMPLING
  ):
    cluster_key, sampling_key = jax.random.split(candidates_key)
    geo_stratum_labels = generate_candidates.cluster_geos(
        processed_data,
        design_config,
        cluster_key,
    ).labels
    candidates: jnp.ndarray = (
        generate_candidates.get_stratified_sampling_candidates(
            selection_train=processed_data.selection_train,
            filtered_data=processed_data.filtered_data,
            design_config=design_config,
            constraints=constraints,
            geo_stratum_labels=geo_stratum_labels,
            key=sampling_key,
            selection_train_spend=processed_data.selection_train_spend,
        )
    )
  else:
    raise ValueError(
        f'Unsupported geo assignment rule: {design_config.geo_assignment_rule}'
    )

  # 3. Fast score based on out of sample R2 and select top X candidates for full
  # scoring.
  if design_config.methodology == api.Methodology.TBR:
    r2_scores: jnp.ndarray = tbr.get_r2(
        processed_data.selection_train,
        processed_data.selection_eval,
        candidates,
    )
  else:
    raise ValueError(f'Unsupported methodology: {design_config.methodology}')

  # 4. Filter and score candidates.
  mde_params: MdeParams = _get_params_for_mde_calculation(
      design_config, r2_scores, candidates, mde_key
  )
  geos = sorted(processed_data.filtered_data[api.LOCATION].unique())
  if design_config.methodology == api.Methodology.TBR:
    mde_results = tbr.get_mde(
        processed_data.estimation_train,
        processed_data.estimation_eval,
        mde_params.top_candidates,
        mde_params.keys,
        design_config.n_aa_test_iterations,
        mde_params.z_score_sum,
        design_config.test_type,
    )
  else:
    raise ValueError(f'Unsupported methodology: {design_config.methodology}')

  # Filter by AA test.
  scored_candidates = ScoredCandidates(
      candidates=mde_params.top_candidates,
      mde_abs=mde_results.mde_abs,
      mde_pct=mde_results.mde_pct,
      p_values=mde_results.p_value,
      r2_scores=r2_scores[mde_params.top_indices],
      observed_conversions=mde_results.observed_conversions,
      counterfactual_conversions=mde_results.counterfactual_conversions,
  )
  filtered_scored_candidates = _filter_results_by_aa_test(
      scored_candidates, design_config
  )

  # 5. Post process and return.
  return _get_design_summary(
      filtered_scored_candidates,
      design_config,
      constraints,
      geos,
      geo_stratum_labels,
      processed_data,
      data,
  )


def compare_designs(
    data: pd.DataFrame,
    design_requirements: list[tuple[api.DesignConfig, api.Constraints]],
    design_output_count: int = 10,
) -> api.DesignSet:
  """Compares designs based on different design configurations."""
  design_sets = [
      run_design(data, config, constraints)
      for config, constraints in design_requirements
  ]

  return concat_design_reports(
      design_sets, design_output_count=design_output_count
  )


def concat_design_reports(
    design_sets: list[api.DesignSet], design_output_count: int = 10
) -> api.DesignSet:
  """Concatenates a list of design sets and return the top N designs."""
  all_designs = {}
  all_metrics = []

  for ds in design_sets:
    all_designs.update(ds.designs)
    all_metrics.append(ds.design_metrics)

  # TODO: Populate design_data.
  design_data = pd.DataFrame()

  if not all_metrics:
    raise ValueError('No design sets to concatenate.')

  combined_metrics = pd.concat(all_metrics, ignore_index=True)

  if combined_metrics.empty:
    raise ValueError('No design metrics to concatenate.')

  top_metrics = combined_metrics.sort_values(by='mde_pct').head(
      design_output_count
  ).reset_index(drop=True)
  top_designs = {
      design_id: all_designs[design_id]
      for design_id in top_metrics['design_id']
  }

  return api.DesignSet(
      designs=top_designs,
      design_metrics=top_metrics,
      design_data=design_data,
  )


def plot_design(data: pd.DataFrame, design_to_plot: api.Design):
  """Visualizes a design."""
  raise NotImplementedError
