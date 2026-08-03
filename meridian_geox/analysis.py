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

"""GeoX analysis library API."""

import dataclasses
import re
from typing import Optional, cast

import jax
import jax.numpy as jnp
from matplotlib import ticker
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from meridian_geox import api
from meridian_geox import design
from meridian_geox import generate_candidates
from meridian_geox import util
from meridian_geox import validation
from meridian_geox.data_quality import data_quality
from meridian_geox.methodology import tbr
import numpy as np
import pandas as pd
import seaborn as sns


@dataclasses.dataclass
class TimeSeries:
  """Time series arrays for analysis."""

  pretest: jnp.ndarray
  test: jnp.ndarray
  pretest_dates: pd.Index
  test_dates: pd.Index


@dataclasses.dataclass
class TreatmentMask:
  """Treatment mask and corresponding geo names."""

  mask: jnp.ndarray
  geos: list[str]


def _get_full_mask(
    treatment_mask: jnp.ndarray,
    placebo_mask: jnp.ndarray,
) -> jnp.ndarray:
  """Returns the full mask for a GeoX experiment."""
  mask = jnp.zeros_like(treatment_mask)
  control_indices = jnp.argwhere(treatment_mask == 0).reshape(-1)
  return mask.at[control_indices].set(placebo_mask)


def _prepare_data(
    data: pd.DataFrame,
    analysis_config: api.AnalysisConfig,
) -> pd.DataFrame:
  """Processes data for experiment analysis."""
  if analysis_config.design.excluded_geos:
    data = data[
        ~data[api.LOCATION].isin(list(analysis_config.design.excluded_geos))
    ]

  if analysis_config.excluded_dates:
    data = data[~data[api.DATE].isin(list(analysis_config.excluded_dates))]

  return data


def _get_time_series(
    data: pd.DataFrame,
    column: str,
    analysis_config: api.AnalysisConfig,
) -> TimeSeries:
  """Extracts pre-test and test time series for a given column."""
  pivoted_data = util.pivot_and_sort_data(data, column)
  if analysis_config.pretest_end_date is not None:
    pretest_data = pivoted_data[
        pivoted_data.index <= analysis_config.pretest_end_date
    ]
  else:
    pretest_data = pivoted_data[
        pivoted_data.index < analysis_config.analysis_start_date
    ]
  pretest = jnp.array(pretest_data.values)
  test_data = pivoted_data[
      (pivoted_data.index >= analysis_config.analysis_start_date)
      & (pivoted_data.index <= analysis_config.analysis_end_date)
  ]
  test = jnp.array(test_data.values)
  return TimeSeries(
      pretest=pretest,
      test=test,
      pretest_dates=pretest_data.index,
      test_dates=test_data.index,
  )


def _get_placebo_masks(
    design_obj: api.Design,
    treatment: TreatmentMask,
    design_config: api.DesignConfig,
    analysis_config: api.AnalysisConfig,
    key: jax.Array,
) -> jnp.ndarray:
  """Generates placebo masks for a GeoX experiment."""
  if design_obj.data is None:
    raise ValueError('Design data is required for placebo mask generation.')

  placebo_data = design_obj.data[
      ~design_obj.data[api.LOCATION].isin(treatment.geos)
  ]
  processed_placebo_data = design.prepare_data(
      data=placebo_data,
      experiment_duration=design_config.experiment_duration,
      constraints=design_obj.constraints,  # pyrefly: ignore[bad-argument-type]
  )

  if design_config.geo_assignment_rule == api.GeoAssignmentRule.RANDOM:
    placebo_masks_without_treatment_geos = (
        generate_candidates.get_random_candidates(
            filtered_data=processed_placebo_data.filtered_data,
            design_config=design_config,
            constraints=design_obj.constraints,  # pyrefly: ignore[bad-argument-type]
            key=key,
            selection_train=processed_placebo_data.selection_train,
            selection_train_spend=processed_placebo_data.selection_train_spend,
        )
    )
  elif (
      design_config.geo_assignment_rule
      == api.GeoAssignmentRule.STRATIFIED_SAMPLING
  ):
    if design_obj.geo_stratum_labels is None:
      raise ValueError(
          'geo_stratum_labels is required for stratified sampling.'
      )
    placebo_masks_without_treatment_geos = (
        generate_candidates.get_stratified_sampling_candidates(
            selection_train=processed_placebo_data.selection_train,
            filtered_data=processed_placebo_data.filtered_data,
            design_config=design_config,
            constraints=design_obj.constraints,  # pyrefly: ignore[bad-argument-type]
            geo_stratum_labels=design_obj.geo_stratum_labels[
                treatment.mask == 0
            ],
            key=key,
            selection_train_spend=processed_placebo_data.selection_train_spend,
        )
    )
  else:
    raise ValueError(
        f'Unsupported geo assignment rule: {design_config.geo_assignment_rule}'
    )

  experiment_types = _get_experiment_types(design_config)
  cell_ids = jnp.array(
      [util.cell_id_from_cell_name(cell) for cell in experiment_types.keys()]
  )

  # Compute R2 for all placebo candidates on T1
  r2_scores = tbr.get_r2(
      data_pre=processed_placebo_data.selection_train,
      data_val=processed_placebo_data.selection_eval,
      treatment_masks=placebo_masks_without_treatment_geos,
      cell_ids=cell_ids,
  )
  r2_scores_agg = jnp.min(r2_scores, axis=1)

  # Filter by R2 threshold
  error_message = (
      'Insufficient valid placebo candidates (found {count}, required at least '
      '{min_count_error}). Only placebo designs with '
      'an out-of-sample R-squared score >= {min_r2} are '
      'kept for analysis. This is usually caused by heterogeneity and '
      'volatility among geos, such that the exchangeability assumption '
      'among geos is violated. Consider providing more stationary pre-test '
      "data or lowering the 'min_placebo_r2' threshold in your AnalysisConfig."
  )

  warning_message = (
      'Low number of valid placebo candidates (found {count}, recommended at '
      'least {min_count_warning}). Only placebo designs with an out-of-sample '
      'R-squared score >= {min_r2} were kept. This is usually caused by '
      'heterogeneity and volatility among geos. Confidence intervals and '
      'p-values may be less reliable as a result. Consider providing more '
      "stationary pre-test data or lowering the 'min_placebo_r2' threshold in "
      'your AnalysisConfig.'
  )

  valid_mask = util.filter_by_r2(
      r2_scores=r2_scores_agg,
      min_r2=analysis_config.min_placebo_r2,
      min_count_error=analysis_config.min_placebo_count_error,
      min_count_warning=analysis_config.min_placebo_count_warning,
      error_message=error_message,
      warning_message=warning_message,
  )

  valid_placebos = placebo_masks_without_treatment_geos[valid_mask]
  valid_r2 = r2_scores_agg[valid_mask]

  # Sort by R2 descending
  indices = jnp.argsort(valid_r2)[::-1]
  selected_subset = valid_placebos[
      indices[: design_config.n_aa_test_iterations]
  ]

  return jax.vmap(_get_full_mask, in_axes=(None, 0))(
      treatment.mask, selected_subset
  )


def _prepare_design_config(
    analysis_config: api.AnalysisConfig,
) -> api.DesignConfig:
  """Prepares the design config for analysis."""
  if analysis_config.design.design_config is None:
    raise ValueError('Design config is required for analysis.')

  design_config = dataclasses.replace(
      analysis_config.design.design_config,
      n_candidates=analysis_config.n_placebo_candidates,
  )

  if not analysis_config.alpha:
    analysis_config.alpha = design_config.alpha

  if analysis_config.test_type is None:
    analysis_config.test_type = design_config.test_type

  if design_config.methodology != api.Methodology.TBR:
    raise ValueError(
        f'Unsupported methodology: {design_config.methodology}. Only TBR is'
        ' supported.'
    )

  return design_config


def _get_experiment_types(
    design_config: api.DesignConfig,
) -> dict[str, api.ExperimentType]:
  """Extracts experiment types for analysis."""
  experiment_types = design_config.experiment_types
  if isinstance(experiment_types, api.ExperimentType):
    experiment_types = {api.CELL_1: experiment_types}

  return experiment_types


def _get_treatment_mask(
    data: pd.DataFrame,
    design_obj: api.Design,
) -> TreatmentMask:
  """Creates a treatment mask and returns treatment geo names."""
  geos = sorted(data[api.LOCATION].unique())
  geo_to_idx = {geo: i for i, geo in enumerate(geos)}
  treatment_mask = np.zeros(len(geos), dtype=np.float32)
  all_treatment_geos = set()

  for cell_name, per_cell_design in design_obj.designs.items():
    cell_id = util.cell_id_from_cell_name(cell_name)
    cell_geos = per_cell_design.treatment_geos
    all_treatment_geos.update(cell_geos)
    for geo in cell_geos:
      treatment_mask[geo_to_idx[geo]] = float(cell_id)

  treatment_geos = sorted(list(all_treatment_geos))
  return TreatmentMask(mask=jnp.array(treatment_mask), geos=treatment_geos)


def _get_estimated_bau_spend_holdback(
    treatment_mask: np.ndarray,
    control_mask: np.ndarray,
    pretest_conversions: jnp.ndarray,
    test_spend: jnp.ndarray,
) -> Optional[float]:
  """Computes estimated BAU spend for a HOLDBACK experiment."""
  treatment_pretest_conversions = jnp.sum(
      pretest_conversions[:, treatment_mask]
  )
  control_pretest_conversions = jnp.sum(pretest_conversions[:, control_mask])

  if treatment_pretest_conversions <= 0:
    return None

  treatment_test_spend = jnp.sum(test_spend[:, treatment_mask])
  control_predicted_spend = (
      control_pretest_conversions / treatment_pretest_conversions
  ) * treatment_test_spend

  estimated_bau_spend = treatment_test_spend + control_predicted_spend
  return float(estimated_bau_spend)


def _get_estimated_bau_spend_go_dark_or_heavy_up(
    treatment_mask: np.ndarray,
    control_mask: np.ndarray,
    test_spend: jnp.ndarray,
    result: Optional[tbr.TbrAnalysisResult] = None,
) -> Optional[float]:
  """Computes estimated BAU spend for a GO_DARK or HEAVY_UP experiment."""
  num_treatment_geos = jnp.sum(treatment_mask.astype(jnp.float32))
  num_control_geos = jnp.sum(control_mask.astype(jnp.float32))

  if num_treatment_geos == 0 or num_control_geos == 0:
    return None

  if result is None or result.counterfactual_spend is None:
    return None

  counterfactual_spend = cast(jnp.ndarray, result.counterfactual_spend)
  if jnp.sum(counterfactual_spend) <= 0:
    return None

  control_test_spend = jnp.sum(test_spend[:, control_mask])
  treatment_predicted_spend = jnp.sum(counterfactual_spend)
  return float(control_test_spend + treatment_predicted_spend)


def _get_estimated_bau_spend(
    analysis_config: api.AnalysisConfig,
    cell: str,
    geos: list[str],
    spend: dict[str, TimeSeries],
    conversions: TimeSeries,
    experiment_type: api.ExperimentType,
    result: Optional[tbr.TbrAnalysisResult] = None,
) -> Optional[float]:
  """Computes estimated BAU spend for a cell during the test period."""
  if cell not in spend:
    return None

  cell_spend = spend[cell]
  pretest_dates = cell_spend.pretest_dates
  test_dates = cell_spend.test_dates

  num_pretest_days = len(pretest_dates)
  num_test_days = len(test_dates)

  if num_pretest_days == 0 or num_test_days == 0:
    return None

  cell_design = analysis_config.design.designs.get(cell)
  if not cell_design or not cell_design.treatment_geos:
    return None
  treatment_mask = np.array([geo in cell_design.treatment_geos for geo in geos])
  control_mask = np.array(
      [geo in analysis_config.design.control_geos for geo in geos]
  )

  if experiment_type == api.ExperimentType.HOLDBACK:
    return _get_estimated_bau_spend_holdback(
        treatment_mask=treatment_mask,
        control_mask=control_mask,
        pretest_conversions=conversions.pretest,
        test_spend=cell_spend.test,
    )

  if experiment_type in (
      api.ExperimentType.GO_DARK,
      api.ExperimentType.HEAVY_UP,
  ):
    return _get_estimated_bau_spend_go_dark_or_heavy_up(
        treatment_mask=treatment_mask,
        control_mask=control_mask,
        test_spend=cell_spend.test,
        result=result,
    )

  return None


def _get_analysis_summary(
    tbr_results: dict[str, tbr.TbrAnalysisResult],
    pretest_dates: pd.Index,
    test_dates: pd.Index,
    analysis_config: api.AnalysisConfig,
    geos: list[str],
    spend: dict[str, TimeSeries],
    conversions: TimeSeries,
    experiment_types: dict[str, api.ExperimentType],
    quality_check_result: Optional[api.QualityCheckResult] = None,
) -> api.AnalysisResult:
  """Converts raw analysis results into an AnalysisResult object."""
  results = {}
  full_dates = pretest_dates.append(test_dates)

  for cell, tbr_result in tbr_results.items():
    cumulative_lift = pd.DataFrame(
        data=tbr_result.cumulative_lift_with_cis,
        index=test_dates,
        columns=['lift', 'lower_bound', 'upper_bound'],
    )

    cumulative_icpd = None
    if tbr_result.cumulative_icpd_with_cis is not None:
      cumulative_icpd = pd.DataFrame(
          data=tbr_result.cumulative_icpd_with_cis,
          index=test_dates,
          columns=['icpd', 'lower_bound', 'upper_bound'],
      )

    counterfactual_conversions = pd.DataFrame(
        data=tbr_result.counterfactual_conversions_with_cis,
        index=full_dates,
        columns=['observed', 'counterfactual', 'lower_bound', 'upper_bound'],
    )

    pointwise_difference = pd.DataFrame(
        data=tbr_result.pointwise_difference_with_cis,
        index=full_dates,
        columns=['difference', 'lower_bound', 'upper_bound'],
    )

    descriptive_metrics = api.DescriptiveMetrics(
        estimated_bau_spend=_get_estimated_bau_spend(
            analysis_config=analysis_config,
            cell=cell,
            geos=geos,
            spend=spend,
            conversions=conversions,
            experiment_type=experiment_types[cell],
            result=tbr_result,
        )
    )

    results[cell] = api.AnalysisMetrics(
        lift=tbr_result.lift,
        percent_lift=tbr_result.percent_lift,
        cumulative_lift=cumulative_lift,
        counterfactual_conversions=counterfactual_conversions,
        pointwise_difference=pointwise_difference,
        icpd=tbr_result.icpd,
        cumulative_icpd=cumulative_icpd,
        descriptive_metrics=descriptive_metrics,
    )

  excluded_geos = (
      set(analysis_config.design.excluded_geos)
      if analysis_config.design.excluded_geos
      else set()
  )

  excluded_dates = (
      set(analysis_config.excluded_dates)
      if analysis_config.excluded_dates
      else set()
  )
  if (
      quality_check_result is not None
      and quality_check_result.quality_check_config.exclude_outlier_dates
      and quality_check_result.outlier_dates
  ):
    excluded_dates.update(quality_check_result.outlier_dates)

  return api.AnalysisResult(
      results=results,
      analysis_config=analysis_config,
      excluded_geos=excluded_geos,
      excluded_dates=excluded_dates,
      quality_check_result=quality_check_result,
  )


def analyze(
    data: pd.DataFrame,
    analysis_config: api.AnalysisConfig,
    # An option to enable and configure automatic data quality checks.
    data_quality_check_config: api.QualityCheckConfig = api.QualityCheckConfig(),
) -> api.AnalysisResult:
  """Analyzes a GeoX experiment."""

  error_messages: list[str] = validation.validate_analysis_input(
      data, analysis_config
  )
  if error_messages:
    raise ValueError(f'Data validation failed: {error_messages}')

  # 1. Prepare configuration.
  design_config = _prepare_design_config(analysis_config)
  experiment_types = _get_experiment_types(design_config)

  # 2. Prepare data and masks.
  data = _prepare_data(data, analysis_config)
  treatment = _get_treatment_mask(data, analysis_config.design)

  # Run data quality checks.
  quality_result = data_quality.check_analysis_data_quality(
      data,
      analysis_config,
      data_quality_check_config,
  )

  # Filter out outlier dates identified in the quality check, if configured.
  if (
      quality_result.outlier_dates
      and data_quality_check_config.exclude_outlier_dates
  ):
    data = data[~data[api.DATE].isin(list(quality_result.outlier_dates))]

  # 3. Extract time series arrays.
  conversions = _get_time_series(data, api.CONVERSIONS, analysis_config)

  spend = {}
  if design_config.cell_count == 1:
    if api.SPEND in data.columns:
      spend[api.CELL_1] = _get_time_series(data, api.SPEND, analysis_config)
  elif design_config.cell_count > 1:
    for col in data.columns:
      if re.match(api.MULTICELL_SPEND_REGEX, col):
        cell_name = col[len('spend_') :]  # Removes "spend_" prefix.
        spend[cell_name] = _get_time_series(data, col, analysis_config)

  # 4. Generate placebo masks.
  # We use a key derived from the design config seed to ensure that placebo
  # mask generation is deterministic but distinct from the design phase.
  # We use a fold-in of a constant value to distinguish this key from others.
  # TODO: After refactoring to move placebo design generation to
  # the design phase, replace the constant fold-in with key splitting.
  analysis_key = jax.random.fold_in(jax.random.key(design_config.seed), 12345)
  placebo_masks = _get_placebo_masks(
      design_obj=analysis_config.design,
      treatment=treatment,
      design_config=design_config,
      analysis_config=analysis_config,
      key=analysis_key,
  )

  # 5. Run methodology analysis.
  tbr_results = {}
  experiment_duration_days = design_config.experiment_duration.days
  pretest_train = conversions.pretest[:-experiment_duration_days]
  pretest_val = conversions.pretest[-experiment_duration_days:]

  for cell, experiment_type in experiment_types.items():
    tbr_results[cell] = tbr.analyze(
        pretest_train_conversions=pretest_train,
        pretest_val_conversions=pretest_val,
        test_conversions=conversions.test,
        treatment_mask=treatment.mask,
        placebo_masks=placebo_masks,
        alpha=analysis_config.alpha,  # pyrefly: ignore[bad-argument-type]
        experiment_type=experiment_type,
        test_type=analysis_config.test_type,  # pyrefly: ignore[bad-argument-type]
        treatment_cell_id=util.cell_id_from_cell_name(cell),
        pretest_spend=spend[cell].pretest if cell in spend else None,
        test_spend=spend[cell].test if cell in spend else None,
    )

  geos = sorted(data[api.LOCATION].unique())

  # 6. Package and return results.
  return _get_analysis_summary(
      tbr_results=tbr_results,
      pretest_dates=conversions.pretest_dates,
      test_dates=conversions.test_dates,
      analysis_config=analysis_config,
      geos=geos,
      spend=spend,
      conversions=conversions,
      experiment_types=experiment_types,
      quality_check_result=quality_result,
  )


def plot_analysis(analysis_result: api.AnalysisResult):
  """Visualizes the 4-plot suite with divided timelines."""
  sns.set_style('whitegrid')
  start_dt = analysis_result.analysis_config.analysis_start_date
  test_type = analysis_result.analysis_config.test_type
  has_icpd = any(
      m.cumulative_icpd is not None for m in analysis_result.results.values()
  )
  n_rows = 3 + has_icpd

  cells = sorted(analysis_result.results.keys())
  color_b = 'tab:blue'
  color_g = 'tab:green'
  color_o = 'tab:orange'

  for cell_id in cells:
    _, axes = plt.subplots(
        n_rows,
        1,
        figsize=(15, 7 * n_rows),
        squeeze=False,
        constrained_layout=True,
        gridspec_kw={'hspace': 0.3},
    )

    metrics = analysis_result.results[cell_id]

    # 1. Initialize local min/max for Y-axis scaling for the current cell.
    local_min_cf, local_max_cf = np.inf, -np.inf
    local_min_pw, local_max_pw = np.inf, -np.inf
    local_min_l, local_max_l = np.inf, -np.inf
    local_min_i, local_max_i = np.inf, -np.inf

    # 2. Collect all finite values to determine local y-limits.
    # Observed vs. Counterfactual
    df_cf = metrics.counterfactual_conversions
    df_cf_test = df_cf[df_cf.index >= start_dt].copy()
    low_cf = df_cf_test['lower_bound']
    high_cf = df_cf_test['upper_bound']
    valid_obs = df_cf['observed'][np.isfinite(df_cf['observed'])]
    valid_cf = df_cf['counterfactual'][np.isfinite(df_cf['counterfactual'])]
    valid_low_cf = low_cf[np.isfinite(low_cf)]
    valid_high_cf = high_cf[np.isfinite(high_cf)]
    all_finite_cf = pd.concat(
        [valid_obs, valid_cf, valid_low_cf, valid_high_cf]
    )
    if not all_finite_cf.empty:
      local_min_cf = all_finite_cf.min()
      local_max_cf = all_finite_cf.max()

    # Pointwise Differences
    df_pw = metrics.pointwise_difference
    df_pw_test = df_pw[df_pw.index >= start_dt].copy()
    low_pw = df_pw_test['lower_bound']
    high_pw = df_pw_test['upper_bound']
    valid_diff = df_pw['difference'][np.isfinite(df_pw['difference'])]
    valid_low_pw = low_pw[np.isfinite(low_pw)]
    valid_high_pw = high_pw[np.isfinite(high_pw)]
    all_finite_pw = pd.concat([valid_diff, valid_low_pw, valid_high_pw])
    if not all_finite_pw.empty:
      local_min_pw = all_finite_pw.min()
      local_max_pw = all_finite_pw.max()

    # Cumulative Lift
    df_l_test = metrics.cumulative_lift[
        metrics.cumulative_lift.index >= start_dt
    ]
    low_l = df_l_test['lower_bound']
    high_l = df_l_test['upper_bound']
    valid_lift = df_l_test['lift'][np.isfinite(df_l_test['lift'])]
    valid_low_l = low_l[np.isfinite(low_l)]
    valid_high_l = high_l[np.isfinite(high_l)]
    all_finite_l = pd.concat([valid_lift, valid_low_l, valid_high_l])
    if not all_finite_l.empty:
      local_min_l = all_finite_l.min()
      local_max_l = all_finite_l.max()

    # Cumulative iCPD
    if has_icpd and metrics.cumulative_icpd is not None:
      df_i_test = metrics.cumulative_icpd[
          metrics.cumulative_icpd.index >= start_dt
      ]
      low_i = df_i_test['lower_bound']
      high_i = df_i_test['upper_bound']
      valid_icpd = df_i_test['icpd'][np.isfinite(df_i_test['icpd'])]
      valid_low_i = low_i[np.isfinite(low_i)]
      valid_high_i = high_i[np.isfinite(high_i)]
      all_finite_vals = pd.concat([valid_icpd, valid_low_i, valid_high_i])
      if not all_finite_vals.empty:
        local_min_i = all_finite_vals.min()
        local_max_i = all_finite_vals.max()

    # 3. Calculate and Set Y-limits.
    def calculate_ylim(local_min, local_max):
      if local_min == np.inf:  # No finite data found
        return 0, 1, 0.15
      y_range = local_max - local_min
      buffer = max(1, y_range * 0.15)
      y_min_final = local_min - buffer
      y_max_final = local_max + buffer
      return y_min_final, y_max_final, buffer

    y_min_cf_final, y_max_cf_final, buffer_cf = calculate_ylim(
        local_min_cf, local_max_cf
    )
    axes[0, 0].set_ylim(y_min_cf_final, y_max_cf_final)

    y_min_pw_final, y_max_pw_final, buffer_pw = calculate_ylim(
        local_min_pw, local_max_pw
    )
    axes[1, 0].set_ylim(y_min_pw_final, y_max_pw_final)

    y_min_l_final, y_max_l_final, buffer_l = calculate_ylim(
        local_min_l, local_max_l
    )
    axes[2, 0].set_ylim(y_min_l_final, y_max_l_final)

    y_min_i_final, y_max_i_final, buffer_i = None, None, None
    if has_icpd:
      y_min_i_final, y_max_i_final, buffer_i = calculate_ylim(
          local_min_i, local_max_i
      )
      axes[3, 0].set_ylim(y_min_i_final, y_max_i_final)

    # --- Plot 1: Observed vs. Counterfactual (Full Timeline) ---
    ax_diag = axes[0, 0]
    df_cf = metrics.counterfactual_conversions
    ax_diag.plot(
        df_cf.index,
        df_cf['observed'],
        label=f'Observed conversions of {cell_id}',
        color=color_b,
        linewidth=2.5,
    )
    ax_diag.plot(
        df_cf.index,
        df_cf['counterfactual'],
        label=f'Counterfactual conversions of {cell_id}',
        color=color_g,
        linewidth=2.5,
    )
    df_cf_test = df_cf[df_cf.index >= start_dt].copy()
    low_cf = df_cf_test['lower_bound']
    high_cf = df_cf_test['upper_bound']

    if test_type == api.TestType.ONE_SIDED:
      cap_upper_cf = y_max_cf_final + buffer_cf
      cap_lower_cf = y_min_cf_final - buffer_cf
      temp_high_cf = np.where(high_cf == np.inf, cap_upper_cf, high_cf)
      temp_low_cf = np.where(low_cf == -np.inf, cap_lower_cf, low_cf)
      ax_diag.fill_between(
          df_cf_test.index,
          temp_low_cf,
          temp_high_cf,
          where=(temp_low_cf <= temp_high_cf),
          alpha=0.25,
          color=color_g,
          label=(
              f'Confidence interval on counterfactual conversions of {cell_id}'
          ),
      )
    else:
      ax_diag.fill_between(
          df_cf_test.index,
          low_cf,
          high_cf,
          where=(low_cf <= high_cf),
          alpha=0.25,
          color=color_g,
          label=(
              f'Confidence interval on counterfactual conversions of {cell_id}'
          ),
      )

    # --- Plot 2: Pointwise Differences (Full Timeline) ---
    ax_pw = axes[1, 0]
    df_pw = metrics.pointwise_difference
    ax_pw.plot(
        df_pw.index,
        df_pw['difference'],
        label=f'Pointwise difference of {cell_id}',
        color=color_o,
        linewidth=2.5,
    )
    df_pw_test = df_pw[df_pw.index >= start_dt].copy()
    low_pw = df_pw_test['lower_bound']
    high_pw = df_pw_test['upper_bound']

    if test_type == api.TestType.ONE_SIDED:
      cap_upper_pw = y_max_pw_final + buffer_pw
      cap_lower_pw = y_min_pw_final - buffer_pw
      temp_high_pw = np.where(high_pw == np.inf, cap_upper_pw, high_pw)
      temp_low_pw = np.where(low_pw == -np.inf, cap_lower_pw, low_pw)
      ax_pw.fill_between(
          df_pw_test.index,
          temp_low_pw,
          temp_high_pw,
          where=(temp_low_pw <= temp_high_pw),
          alpha=0.25,
          color=color_o,
          label=f'Confidence interval of {cell_id}',
      )
    else:
      ax_pw.fill_between(
          df_pw_test.index,
          low_pw,
          high_pw,
          where=(low_pw <= high_pw),
          alpha=0.25,
          color=color_o,
          label=f'Confidence interval of {cell_id}',
      )

    # --- Plot 3: Cumulative Lift (Test Period Only) ---
    ax_l = axes[2, 0]
    df_l_test = metrics.cumulative_lift[
        metrics.cumulative_lift.index >= start_dt
    ]
    ax_l.plot(
        df_l_test.index,
        df_l_test['lift'],
        label=f'Cumulative lift of {cell_id}',
        color=color_b,
        linewidth=2.5,
    )
    low_l = df_l_test['lower_bound']
    high_l = df_l_test['upper_bound']

    if test_type == api.TestType.ONE_SIDED:
      cap_upper_l = y_max_l_final + buffer_l
      cap_lower_l = y_min_l_final - buffer_l
      temp_low_l = np.where(low_l == -np.inf, cap_lower_l, low_l)
      temp_high_l = np.where(high_l == np.inf, cap_upper_l, high_l)
      ax_l.fill_between(
          df_l_test.index,
          temp_low_l,
          temp_high_l,
          where=(temp_low_l <= temp_high_l),
          alpha=0.25,
          color=color_b,
          label=f'Confidence interval of {cell_id}',
      )
    else:  # TWO_SIDED
      ax_l.fill_between(
          df_l_test.index,
          low_l,
          high_l,
          where=(low_l <= high_l),
          alpha=0.25,
          color=color_b,
          label=f'Confidence interval of {cell_id}',
      )

    # --- Plot 4: Cumulative iCPD (Test Period Only) ---
    if has_icpd and metrics.cumulative_icpd is not None:
      ax_i = axes[3, 0]
      df_i_test = metrics.cumulative_icpd[
          metrics.cumulative_icpd.index >= start_dt
      ]
      ax_i.plot(
          df_i_test.index,
          df_i_test['icpd'],
          label=f'Cumulative iCPD of {cell_id}',
          color=color_b,
          linewidth=2.5,
      )
      low_i = df_i_test['lower_bound']
      high_i = df_i_test['upper_bound']

      if test_type == api.TestType.ONE_SIDED:
        cap_lower_i = y_min_i_final - buffer_i  # pyrefly: ignore[unsupported-operation]
        cap_upper_i = y_max_i_final + buffer_i  # pyrefly: ignore[unsupported-operation]
        temp_low_i = np.where(low_i == -np.inf, cap_lower_i, low_i)
        temp_high_i = np.where(high_i == np.inf, cap_upper_i, high_i)
        ax_i.fill_between(
            df_i_test.index,
            temp_low_i,
            temp_high_i,
            where=(temp_low_i <= temp_high_i),
            alpha=0.25,
            color=color_b,
            label=f'Confidence interval of {cell_id}',
        )
      else:  # TWO_SIDED
        ax_i.fill_between(
            df_i_test.index,
            low_i,
            high_i,
            where=(low_i <= high_i),
            alpha=0.25,
            color=color_b,
            label=f'Confidence interval of {cell_id}',
        )

    # Final Layout Formatting with Divider
    pretest_end_date = analysis_result.analysis_config.pretest_end_date
    for j, ax in enumerate(axes.flatten()):
      ax.axvline(
          start_dt,
          color='grey',
          linestyle='--',
          linewidth=1.5,
          alpha=0.8,
          label='Test start date',
      )  # Period Divider

      # Add pretest end line for the first two plots if it is set.
      if pretest_end_date is not None and j < 2:
        ax.axvline(
            pretest_end_date,
            color='grey',
            linestyle='-',
            linewidth=1.5,
            alpha=0.8,
            label='Pretest end date',
        )

      # Deduplicate labels in the legend
      handles, labels = ax.get_legend_handles_labels()
      unique_labels = {}
      for handle, label in zip(handles, labels):
        unique_labels[label] = handle
      ax.legend(
          unique_labels.values(),
          unique_labels.keys(),
          bbox_to_anchor=(1.02, 1),
          loc='upper left',
          frameon=False,
      )
      ax.grid(
          True,
          which='major',
          axis='both',
          linestyle=':',
          alpha=0.6,
          linewidth=0.8,
      )
      ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
      ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=7, prune='both'))

    # Set Ordered Titles and Labels
    axes[0, 0].set_title(
        f'Observed vs. Counterfactual total conversions ({cell_id})',
        fontsize=14,
    )
    axes[0, 0].set_ylabel('Total conversions')
    axes[1, 0].set_title(
        f'Pointwise differences time series ({cell_id})', fontsize=14
    )
    axes[1, 0].set_ylabel('Pointwise differences')
    axes[2, 0].set_title(
        f'Cumulative lift time series ({cell_id})', fontsize=14
    )
    axes[2, 0].set_ylabel('Cumulative incremental conversions')
    if has_icpd:
      axes[3, 0].set_title(
          f'Cumulative iCPD time series ({cell_id})', fontsize=14
      )
      axes[3, 0].set_ylabel('Cumulative iCPD')

  plt.show()
