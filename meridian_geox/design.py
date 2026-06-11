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
import re
from typing import Any, Callable, Optional
import uuid

import jax
import jax.numpy as jnp
from matplotlib import ticker
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from meridian_geox import api
from meridian_geox import generate_candidates
from meridian_geox import util
from meridian_geox.data_quality import data_quality
from meridian_geox.methodology import tbr
import numpy as np
import pandas as pd
from scipy import stats
import seaborn as sns


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
  # Spend data for slope check, keyed by cell name.
  selection_train_spend: dict[str, api.JnpArray] = dataclasses.field(
      default_factory=dict
  )
  # Evaluation spend data for budget calculation, keyed by cell name.
  estimation_eval_spend: dict[str, api.JnpArray] = dataclasses.field(
      default_factory=dict
  )


@dataclasses.dataclass
class ScoredCandidates:
  """Scored candidates for experiment design.

  Attributes:
    candidates: (n_candidates, geos) Geo assignment for each candidate.
    mde_abs: (n_candidates, k_cells) Absolute MDE for each candidate.
    mde_pct: (n_candidates, k_cells) Relative MDE (percentage) for each
      candidate.
    p_values: (n_candidates, k_cells) The AA p-value for each candidate.
    r2_scores: (n_candidates, k_cells) The out of sample R2 score for each
      candidate.
    observed_conversions: (n_candidates, k_cells, n_dates) Observed conversions
      for each candidate.
    counterfactual_conversions: (n_candidates, k_cells, n_dates) Counterfactual
      conversions for each candidate.
  """

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

  selection_train_spend = {}
  estimation_eval_spend = {}
  if api.SPEND in data.columns:
    pivoted_spend = util.pivot_and_sort_data(data, api.SPEND)
    t0_spend = pivoted_spend.iloc[:t1_start_idx]
    t2_spend = pivoted_spend.iloc[t2_start_idx:]
    selection_train_spend[api.CELL_1] = jnp.array(t0_spend.values)
    estimation_eval_spend[api.CELL_1] = jnp.array(t2_spend.values)
  elif any(re.match(api.MULTICELL_SPEND_REGEX, col) for col in data.columns):
    for col in data.columns:
      if re.match(api.MULTICELL_SPEND_REGEX, col):
        cell_name = col[6:]  # Removes "spend_" prefix.
        pivoted_spend = util.pivot_and_sort_data(data, col)
        t0_spend = pivoted_spend.iloc[:t1_start_idx]
        t2_spend = pivoted_spend.iloc[t2_start_idx:]
        selection_train_spend[cell_name] = jnp.array(t0_spend.values)
        estimation_eval_spend[cell_name] = jnp.array(t2_spend.values)

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

  # For multicell, we select the minimum R2 score across cells for each
  # candidate.
  r2_scores = jnp.min(r2_scores, axis=1)
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
  is_valid_aa = (
      jnp.min(scored_candidates.p_values, axis=1) >= design_config.alpha
  )

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
  mde_pct_np = np.array(scored_candidates.mde_pct)
  p_values_np = np.array(scored_candidates.p_values)
  r2_scores_np = np.array(scored_candidates.r2_scores)
  observed_conversions_np = np.array(scored_candidates.observed_conversions)
  counterfactual_conversions_np = np.array(
      scored_candidates.counterfactual_conversions
  )
  estimation_eval_np = np.array(processed_data.estimation_eval)
  estimation_eval_spend_np_dict = {
      cell: np.array(spend)
      for cell, spend in processed_data.estimation_eval_spend.items()
  }

  # Get full dates for plotting.
  pivoted_data = util.pivot_and_sort_data(data, api.CONVERSIONS)
  full_dates = pivoted_data.index

  # Rely on centralized normalization performed at the start of run_design.
  # We use type hints to satisfy Pytype since they were normalized earlier in
  # DesignConfig validation or Constraints.normalize().
  experiment_types: dict[str, api.ExperimentType] = (
      design_config.experiment_types  # type: ignore
  )
  # After normalization, budget_constraint is guaranteed to be a dict.
  budget_constraints: dict[str, api.Budget] = (
      constraints.budget_constraint or {}  # type: ignore
  )
  cpics: dict[str, float] = (
      design_config.cost_per_incremental_conversion  # type: ignore
  )

  for i in range(len(top_candidates_np)):
    design_id = str(uuid.uuid4())
    cell_designs = {}

    mask = top_candidates_np[i]
    control_geos = set()

    # Identify control geos (mask == 0).
    control_indices = np.where(mask == 0)[0]
    for idx in control_indices:
      control_geos.add(geos[idx])

    for cell_name, experiment_type in experiment_types.items():
      # Treatment cells are indexed starting from 1, so we need to subtract 1
      # to get the correct index for the metrics array.
      metrics_cell_index = util.cell_id_from_cell_name(cell_name) - 1
      treatment_indices = np.where(
          mask == util.cell_id_from_cell_name(cell_name)
      )[0]
      treatment_geos = {geos[idx] for idx in treatment_indices}

      treatment_conversion_volume = float(
          np.sum(estimation_eval_np[:, treatment_indices])
      )
      mde_pct = float(mde_pct_np[i][metrics_cell_index])
      total_mde_abs = mde_pct * treatment_conversion_volume

      # Calculate budget.
      # 1. For GO_DARK and HEAVY_UP:
      #    - If constraints.budget_percent is set, use
      #      treatment_geo_cost * budget_percent.
      #    - Otherwise, use treatment_geo_cost.
      # 2. For HOLDBACK:
      #    - Use MDE * treatment_conversion_volume * CPIC.
      budget_constraint = budget_constraints.get(cell_name)
      if util.is_go_dark_or_heavy_up(experiment_type):
        budget_percent = (
            budget_constraint.budget_pct
            if budget_constraint and budget_constraint.budget_pct is not None
            else 1.0
        )
        estimation_eval_spend_np = estimation_eval_spend_np_dict[cell_name]
        treatment_geo_cost = float(
            np.sum(estimation_eval_spend_np[:, treatment_indices])
        )
        required_budget = treatment_geo_cost * budget_percent
      else:
        # HOLDBACK
        # cpic is guaranteed to be set for HOLDBACK cells by validation.
        cpic = cpics[cell_name]
        required_budget = total_mde_abs * cpic

      cell_designs[cell_name] = api.PerCellDesign(
          treatment_geos=treatment_geos,
          minimum_detectable_effect=mde_pct,
          design_implied_cpic=(
              required_budget / total_mde_abs if total_mde_abs > 0 else 0.0
          ),
          p_value=float(p_values_np[i][metrics_cell_index]),
          budget=float(required_budget),
          counterfactual_conversions=pd.DataFrame({
              'index': full_dates,
              'observed': observed_conversions_np[i][metrics_cell_index],
              'counterfactual': counterfactual_conversions_np[i][
                  metrics_cell_index
              ],
          }),
      )

      metrics_list.append({
          'design_id': design_id,
          'cell': cell_name,
          'design_methodology': (
              f'{design_config.geo_assignment_rule.name}-'
              f'{design_config.methodology.name}'
          ),
          'r2': r2_scores_np[i][metrics_cell_index],
          'mde': mde_pct,
          'mde_abs': total_mde_abs,
          'p_value': p_values_np[i][metrics_cell_index],
          'budget': required_budget,
      })

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

  design_metrics = pd.DataFrame(metrics_list)

  # Rank by max_mde across cells.
  max_mde_df = (
      design_metrics.groupby('design_id')['mde']
      .max()
      .reset_index(name='max_mde')
  )
  top_design_ids = max_mde_df.sort_values(by='max_mde').head(
      design_config.design_output_count
  )['design_id']
  design_metrics = (
      design_metrics[design_metrics['design_id'].isin(top_design_ids)]
      .sort_values(by='mde')
      .reset_index(drop=True)
  )
  designs = {design_id: designs[design_id] for design_id in top_design_ids}

  # Log warnings for returned candidates that exceed budget.
  for design_id, design_obj in designs.items():
    for cell_name, per_cell_design in design_obj.designs.items():
      if experiment_types.get(cell_name) == api.ExperimentType.HOLDBACK:
        provided_budget = budget_constraints.get(cell_name)
        if provided_budget and provided_budget.budget is not None:
          if per_cell_design.budget > provided_budget.budget:
            logging.warning(
                'Design %s: %s required budget (%.2f) exceeds provided '
                'budget (%.2f).',
                design_id,
                cell_name,
                per_cell_design.budget,
                provided_budget.budget,
            )

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
  # Normalization handles scalar-to-dict conversion in DesignConfig and
  # Constraints.
  experiment_types: dict[str, api.ExperimentType] = (
      design_config.experiment_types  # type: ignore
  )
  constraints.normalize(experiment_types)

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
  cell_ids = jnp.arange(1, design_config.cell_count + 1).astype(jnp.float32)
  if design_config.methodology == api.Methodology.TBR:
    r2_scores: jnp.ndarray = tbr.get_r2(
        processed_data.selection_train,
        processed_data.selection_eval,
        candidates,
        cell_ids,
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
        cell_ids=cell_ids,
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

  top_metrics = (
      combined_metrics.sort_values(by='mde')
      .head(design_output_count)
      .reset_index(drop=True)
  )
  top_designs = {
      design_id: all_designs[design_id]
      for design_id in top_metrics['design_id']
  }

  return api.DesignSet(
      designs=top_designs,
      design_metrics=top_metrics,
      design_data=design_data,
  )


def plot_design(design_to_plot: api.Design):
  """Visualizes pre-test alignment for all cells in a single shared plot."""
  sns.set_style('ticks')
  plt.figure(figsize=(15, 7))
  ax = plt.gca()

  cells = sorted(design_to_plot.designs.keys())
  # Using sequential palettes to distinguish multiple cells clearly
  blues = sns.color_palette('Blues_d', len(cells))
  greens = sns.color_palette('Greens_d', len(cells))

  for i, cell_id in enumerate(cells):
    cell_design = design_to_plot.designs[cell_id]
    df = cell_design.counterfactual_conversions
    if df is None or df.empty:
      continue

    # Plot Observed
    ax.plot(
        df['index'],
        df['observed'],
        label=f'Observed conversions of {cell_id}',
        color=blues[i],
        linewidth=2.5,
    )

    # Plot Counterfactual
    ax.plot(
        df['index'],
        df['counterfactual'],
        label=f'Counterfactual conversions of {cell_id}',
        color=greens[i],
        linestyle='-',
        linewidth=2.5,
        alpha=0.8,
    )

  ax.grid(
      True, which='major', axis='both', linestyle=':', alpha=0.6, linewidth=0.8
  )
  ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
  ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=7, prune='both'))

  ax.set_title(
      'Observed vs Counterfactual total conversions',
      fontsize=14,
      fontweight='bold',
  )
  ax.set_ylabel('Total conversions')
  ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', frameon=False)
  plt.show()
