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

"""TBR methodology implementation."""

import dataclasses
import functools
from typing import Optional

import jax
import jax.numpy as jnp
from jax.scipy import stats
from meridian_geox import api
from meridian_geox.methodology import util as methodology_util
import numpy as np


@dataclasses.dataclass
class MdeResults:
  """Results of the MDE calculation.

  Attributes:
    mde_abs: (N_designs, k_cells) Absolute MDE.
    mde_pct: (N_designs, k_cells) Relative MDE (percentage).
    p_value: (N_designs, k_cells) The AA p-value.
    observed_conversions: (N_designs, k_cells, T) Observed conversions.
    counterfactual_conversions: (N_designs, k_cells, T) Counterfactual
      conversions.
  """

  mde_abs: jnp.ndarray
  mde_pct: jnp.ndarray
  p_value: jnp.ndarray
  observed_conversions: jnp.ndarray
  counterfactual_conversions: jnp.ndarray


jax.tree_util.register_pytree_node(
    MdeResults,
    lambda node: (
        (
            node.mde_abs,
            node.mde_pct,
            node.p_value,
            node.observed_conversions,
            node.counterfactual_conversions,
        ),
        None,
    ),
    lambda _, children: MdeResults(*children),
)


@dataclasses.dataclass
class IcpdResults:
  """Results of the ICPD calculation.

  Attributes:
    cumulative_icpd: (T,) Cumulative ICPD.
    lower_bound: (T,) Lower bound of the cumulative ICPD.
    upper_bound: (T,) Upper bound of the cumulative ICPD.
    cumulative_incremental_spend: (T,) Cumulative incremental spend.
    counterfactual_spend: (T,) Predicted counterfactual spend.
  """

  cumulative_icpd: jnp.ndarray
  lower_bound: jnp.ndarray
  upper_bound: jnp.ndarray
  cumulative_incremental_spend: jnp.ndarray
  counterfactual_spend: jnp.ndarray


jax.tree_util.register_pytree_node(
    IcpdResults,
    lambda node: (
        (
            node.cumulative_icpd,
            node.lower_bound,
            node.upper_bound,
            node.cumulative_incremental_spend,
            node.counterfactual_spend,
        ),
        None,
    ),
    lambda _, children: IcpdResults(*children),
)


@dataclasses.dataclass
class TbrAnalysisResult:
  """Results of the TBR analysis.

  Attributes:
    lift: The incremental lift estimate.
    cumulative_lift_with_cis: (T, 3) Cumulative lift estimates with CIs.
    percent_lift: The relative lift estimate.
    icpd: The incremental cost per dollar estimate.
    cumulative_icpd_with_cis: (T, 3) Cumulative ICPD estimates with CIs.
    counterfactual_conversions_with_cis: (T, 4) Observed, counterfactual, and
      CIs.
    pointwise_difference_with_cis: (T, 3) Pointwise difference and CIs.
    counterfactual_spend: (T,) Predicted counterfactual spend.
  """

  lift: api.Estimate
  cumulative_lift_with_cis: np.ndarray
  percent_lift: api.Estimate
  icpd: Optional[api.Estimate] = None
  cumulative_icpd_with_cis: Optional[np.ndarray] = None
  counterfactual_conversions_with_cis: np.ndarray = dataclasses.field(
      default_factory=lambda: np.array([])
  )
  pointwise_difference_with_cis: np.ndarray = dataclasses.field(
      default_factory=lambda: np.array([])
  )
  counterfactual_spend: Optional[np.ndarray] = None


@jax.jit
def _fit_linear_regression(
    x: jnp.ndarray, y: jnp.ndarray
) -> tuple[jnp.ndarray, jnp.ndarray]:
  """Fits a simple linear regression y ~ alpha + beta * x.

  Args:
    x: (T,) Independent variable.
    y: (T,) Dependent variable.

  Returns:
    alpha: Intercept.
    beta: Slope.
  """
  # We compute directly as opposed to using jnp.linalg.lstsq for better
  # performance with JAX. The latter has much bigger diffs when comparing to
  # standard NumPy.
  x_mean = jnp.mean(x)
  y_mean = jnp.mean(y)
  ss_xx = jnp.sum((x - x_mean) ** 2)
  ss_xy = jnp.sum((x - x_mean) * (y - y_mean))
  # TODO: Raise an error if x is constant (ss_xx == 0).
  slope = jnp.where(ss_xx > 1e-10, ss_xy / ss_xx, 0.0)
  # Ensure slope is non-negative.
  slope = jnp.maximum(0.0, slope)
  intercept = y_mean - slope * x_mean

  return intercept, slope


@jax.jit
def _compute_group_mean(
    data: jnp.ndarray,
    mask: jnp.ndarray,
    cell_id: float,
) -> jnp.ndarray:
  """Computes mean time series for a given cell.

  Args:
    data: (T, N) Time series data.
    mask: (N,) Treatment mask (0.0 for control, positive integer for treatment).
    cell_id: Cell ID.

  Returns:
    mean_ts: (T,) Average time series for the cell.
  """
  binary_mask = (mask == cell_id).astype(jnp.float32)
  n_geos = jnp.sum(binary_mask)
  mean_ts = jnp.dot(data, binary_mask) / jnp.maximum(n_geos, 1.0)

  return mean_ts


# TODO: Decouple methodology-specific estimation from the general
# placebo inference framework to enhance modularity as new methods are
# integrated.
def _compute_placebo_effect_from_mask(
    data_pretest: jnp.ndarray,
    data_pretest_train: jnp.ndarray,
    data_pretest_val: jnp.ndarray,
    data_test: jnp.ndarray,
    mask: jnp.ndarray,
    p_mask: jnp.ndarray,
    treatment_cell_id: float,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
  """Computes the effect on test data and out-of-sample RMSE for a single placebo simulation.

  Args:
    data_pretest: (T_pre, Geos) Pre-period data.
    data_pretest_train: (T_pre_train, Geos) Training pre-period data.
    data_pretest_val: (T_pre_val, Geos) Validation pre-period data.
    data_test: (T_val, Geos) Test-period data.
    mask: (Geos,) Treatment mask (1.0 for treated, 0.0 for control).
    p_mask: (Geos,) Placebo treatment mask (1.0 for placebo treated, 0.0 for
      control).
    treatment_cell_id: The Cell ID to use for the treated group.

  Returns:
    Placebo treatment response, Placebo predicted response, Placebo RMSE
    (out-of-sample), Placebo log RMSE (out-of-sample).
  """
  # Create a mask for valid control units (excluding original treated units).
  # p_mask is positive for placebo treated, 0.0 otherwise.
  # mask is positive for original treated, 0.0 otherwise.
  valid_control_mask = ((mask == 0.0) & (p_mask == 0.0)).astype(jnp.float32)
  # We only want to compare against the specified treatment cell.
  p_treatment_mask = (p_mask == treatment_cell_id).astype(jnp.float32)

  def _compute_placebo_means(data, t_mask, c_mask):
    n_t = jnp.sum(t_mask)
    n_c = jnp.sum(c_mask)
    y_m = jnp.dot(data, t_mask) / jnp.maximum(n_t, 1.0)
    x_m = jnp.dot(data, c_mask) / jnp.maximum(n_c, 1.0)
    return y_m, x_m

  # Part 1. Fit on the training split to calculate the out-of-sample validation
  # RMSE.

  py_train, px_train = _compute_placebo_means(
      data_pretest_train, p_treatment_mask, valid_control_mask
  )
  p_alpha_train, p_beta_train = _fit_linear_regression(px_train, py_train)

  py_val, px_val = _compute_placebo_means(
      data_pretest_val, p_treatment_mask, valid_control_mask
  )
  py_pred_val = p_alpha_train + p_beta_train * px_val
  p_rmse = jnp.sqrt(jnp.mean((py_val - py_pred_val) ** 2))

  scale_val = jnp.mean(jnp.abs(py_val))
  p_log_errors = methodology_util.compute_regularized_log_ratio(
      py_val, py_pred_val, scale=scale_val
  )
  p_log_rmse = jnp.sqrt(jnp.mean(p_log_errors**2))

  # Part 2. Fit on combined pretest for the actual test prediction

  py_pretest, px_pretest = _compute_placebo_means(
      data_pretest, p_treatment_mask, valid_control_mask
  )
  p_alpha, p_beta = _fit_linear_regression(px_pretest, py_pretest)

  # Predict on val.
  py_test, px_test = _compute_placebo_means(
      data_test, p_treatment_mask, valid_control_mask
  )
  # We want pointwise results so that we can obtain a pointwise analysis times
  # series with CIs.
  py_pred = p_alpha + p_beta * px_test

  return py_test, py_pred, p_rmse, p_log_rmse


@jax.jit
def get_r2(
    data_pre: jnp.ndarray,
    data_val: jnp.ndarray,
    treatment_masks: jnp.ndarray,
    cell_ids: jnp.ndarray,
) -> jnp.ndarray:
  """Calculates validation R-squared for a batch of designs.

  Args:
    data_pre: (T_pre, Geos) Pre-period data.
    data_val: (T_val, Geos) Validation-period data.
    treatment_masks: (N_designs, Geos) Batch of treatment masks.
    cell_ids: (k_cells,) Cell IDs.

  Returns:
    r2_vals: (N_designs, k_cells) Out-of-sample R-squared on validation period.
  """

  def _get_one_r2(mask):
    # Ensure mask is float.
    mask = mask.astype(jnp.float32)

    # Calculate means for treated and control groups in pre-period.
    x_pre = _compute_group_mean(data_pre, mask, 0.0)
    y_pre_by_cell = jax.vmap(_compute_group_mean, in_axes=(None, None, 0))(
        data_pre, mask, cell_ids
    )

    # Simple Linear Regression: y_pre ~ alpha + beta * x_pre.
    alpha_by_cell, beta_by_cell = jax.vmap(
        _fit_linear_regression, in_axes=(None, 0)
    )(x_pre, y_pre_by_cell)

    # Validation period.
    x_val = _compute_group_mean(data_val, mask, 0.0)
    y_val_by_cell = jax.vmap(_compute_group_mean, in_axes=(None, None, 0))(
        data_val, mask, cell_ids
    )

    # Predict.
    y_pred_by_cell = alpha_by_cell[:, None] + beta_by_cell[:, None] * x_val

    # Calculate R2.
    residuals_by_cell = y_val_by_cell - y_pred_by_cell
    mse_by_cell = jnp.mean(residuals_by_cell**2, axis=1)
    var_y_by_cell = jnp.var(y_val_by_cell, axis=1)

    # Avoid division by zero.
    r2_by_cell = jax.vmap(
        lambda mse, var_y: jnp.where(var_y > 1e-10, 1.0 - (mse / var_y), 0.0),
        in_axes=(0, 0),
    )(mse_by_cell, var_y_by_cell)

    # Return nan if invalid design (no treated or no control).
    n_control = jnp.sum((mask == 0.0).astype(jnp.float32))

    n_treated_by_cell = jnp.sum(
        (mask == cell_ids[:, None]).astype(jnp.float32), axis=1
    )
    is_valid = (n_control > 0) & jnp.all(n_treated_by_cell > 0)
    return jnp.where(
        is_valid,
        r2_by_cell,
        jnp.full(len(cell_ids), jnp.nan),
    )

  return jax.vmap(_get_one_r2)(treatment_masks)


@functools.partial(jax.jit, static_argnames=['test_type'])
def _get_mde(
    data_pre: jnp.ndarray,
    data_val: jnp.ndarray,
    treatment_masks: jnp.ndarray,
    z_score_sum: float,
    cell_ids: jnp.ndarray,
    test_type: api.TestType = api.TestType.TWO_SIDED,
) -> MdeResults:
  """Calculates MDE using simplified design aware placebo method."""

  def _get_effect(mask, treatment_cell_id):
    y_pre = _compute_group_mean(data_pre, mask, treatment_cell_id)
    x_pre = _compute_group_mean(data_pre, mask, 0.0)
    alpha, beta = _fit_linear_regression(x_pre, y_pre)
    y_val = _compute_group_mean(data_val, mask, treatment_cell_id)
    x_val = _compute_group_mean(data_val, mask, 0.0)
    y_pred = alpha + beta * x_val

    n_treated = jnp.sum(mask == treatment_cell_id)
    real_effect = jnp.mean(y_val - y_pred)
    baseline = jnp.mean(y_val)

    # Combine pre and val for full time series.
    y_full = jnp.concatenate([y_pre, y_val])
    y_pred_full = jnp.concatenate([alpha + beta * x_pre, y_pred])

    return real_effect, baseline, n_treated * y_full, n_treated * y_pred_full

  def _get_mde_and_p_value(effects, baselines):
    se = jnp.std(effects)
    mde_abs_val = se * z_score_sum

    # Broadcast to shape (N_designs,).
    mde_abs = jnp.full_like(effects, mde_abs_val)
    mde_pct = jnp.where(baselines > 1e-9, mde_abs / baselines, jnp.nan)

    # P-values.
    z_scores = effects / jnp.maximum(se, 1e-10)
    if test_type == api.TestType.TWO_SIDED:
      p_value = 2.0 * (1.0 - stats.norm.cdf(jnp.abs(z_scores)))
    else:
      # One-sided: Upper tail.
      p_value = 1.0 - stats.norm.cdf(z_scores)
    p_value = jnp.where(se > 1e-10, p_value, 1.0)

    return mde_abs, mde_pct, p_value

  _get_effect_over_masks = jax.vmap(_get_effect, in_axes=(0, None))
  # shapes:
  # effects, baselines: (k_cells, N_designs)
  # observed, counterfactual: (k_cells, N_designs, T_pre + T_val)
  effects, baselines, observed, counterfactual = jax.vmap(
      _get_effect_over_masks,
      in_axes=(None, 0),
  )(treatment_masks, cell_ids)

  # shapes:
  # mde_abs, mde_pct, p_values: (k_cells, N_designs)
  (
      mde_abs,
      mde_pct,
      p_values,
  ) = (
      jax.vmap(_get_mde_and_p_value, in_axes=(0, 0))
  )(effects, baselines)

  # shapes:
  # mde_abs, mde_pct, p_values: (N_designs, k_cells)
  # observed, counterfactual: (N_designs, k_cells, T_pre + T_val)
  return MdeResults(
      mde_abs=jnp.transpose(mde_abs),
      mde_pct=jnp.transpose(mde_pct),
      p_value=jnp.transpose(p_values),
      observed_conversions=jnp.transpose(observed, axes=(1, 0, 2)),
      counterfactual_conversions=jnp.transpose(counterfactual, axes=(1, 0, 2)),
  )


def get_mde(
    data_pre: jnp.ndarray,
    data_val: jnp.ndarray,
    treatment_masks: jnp.ndarray,
    z_score_sum: float,
    test_type: api.TestType = api.TestType.TWO_SIDED,
    cell_ids: Optional[jnp.ndarray] = None,
) -> MdeResults:
  """Runs placebo check and calculates MDE for a batch of designs."""
  if cell_ids is None:
    cell_ids = jnp.array([1.0])

  return _get_mde(
      data_pre,
      data_val,
      treatment_masks,
      z_score_sum,
      cell_ids,
      test_type,
  )


@jax.jit
def check_slope_similarity(
    candidates: jnp.ndarray,
    conversion_data: jnp.ndarray,
    spend_data: jnp.ndarray,
    tolerance: float,
    cell_id: float,
) -> jnp.ndarray:
  """Checks if the slope between conversion and spend is similar.

  Args:
    candidates: (N_designs, N_geos) Treatment masks.
    conversion_data: (T, N_geos) Conversion time series data.
    spend_data: (T, N_geos) Spend time series data.
    tolerance: Maximum allowed symmetric difference.
    cell_id: Cell ID for multicell designs.

  Returns:
    mask: (N_designs,) Boolean mask where True indicates the design passed the
      check.
  """
  def _get_single_slope_diff(mask):
    # 1. Conversion slope.
    # y = treated, x = control.
    y_conv = _compute_group_mean(conversion_data, mask, cell_id)
    x_conv = _compute_group_mean(conversion_data, mask, 0.0)
    _, b_conv = _fit_linear_regression(x_conv, y_conv)

    # 2. Spend slope.
    y_spend = _compute_group_mean(spend_data, mask, cell_id)
    x_spend = _compute_group_mean(spend_data, mask, 0.0)
    _, b_spend = _fit_linear_regression(x_spend, y_spend)

    # 3. Calculate symmetric percentage difference.
    # diff = 2 * |b1 - b2| / (|b1| + |b2|)
    denom = jnp.abs(b_conv) + jnp.abs(b_spend)
    is_zero = (jnp.abs(b_conv) < 1e-10) | (jnp.abs(b_spend) < 1e-10)

    diff = jnp.where(
        is_zero,
        jnp.inf,
        2.0 * jnp.abs(b_conv - b_spend) / denom,
    )

    return diff <= tolerance

  return jax.vmap(_get_single_slope_diff)(candidates)


@jax.jit
def _compute_icpd(
    pretest_spend: jnp.ndarray,
    test_spend: jnp.ndarray,
    treatment_mask: jnp.ndarray,
    total_cumul_effect: jnp.ndarray,
    total_lower_cis: jnp.ndarray,
    total_upper_cis: jnp.ndarray,
    incremental_spend_sign: float,
    treatment_cell_id: float = 1.0,
):
  """Computes ICPD metrics using JAX arrays."""
  y_spend_pre = _compute_group_mean(
      pretest_spend, treatment_mask, treatment_cell_id
  )
  x_spend_pre = _compute_group_mean(pretest_spend, treatment_mask, 0.0)
  pred_alpha_spend, pred_beta_spend = _fit_linear_regression(
      x_spend_pre, y_spend_pre
  )

  y_spend_test = _compute_group_mean(
      test_spend, treatment_mask, treatment_cell_id
  )
  x_spend_test = _compute_group_mean(test_spend, treatment_mask, 0.0)
  y_spend_pred = pred_alpha_spend + pred_beta_spend * x_spend_test
  n_treatment_geos = jnp.sum(
      (treatment_mask == treatment_cell_id).astype(jnp.float32)
  )

  incremental_spend = (
      incremental_spend_sign * n_treatment_geos * (y_spend_test - y_spend_pred)
  )
  cumulative_incremental_spend = jnp.cumsum(incremental_spend)

  cumulative_icpd = total_cumul_effect / cumulative_incremental_spend
  icpd_lower = total_lower_cis / cumulative_incremental_spend
  icpd_upper = total_upper_cis / cumulative_incremental_spend

  return IcpdResults(
      cumulative_icpd=cumulative_icpd,
      lower_bound=icpd_lower,
      upper_bound=icpd_upper,
      cumulative_incremental_spend=cumulative_incremental_spend,
      counterfactual_spend=y_spend_pred * n_treatment_geos,
  )


def analyze(
    pretest_train_conversions: jnp.ndarray,
    pretest_val_conversions: jnp.ndarray,
    test_conversions: jnp.ndarray,
    treatment_mask: jnp.ndarray,
    placebo_masks: jnp.ndarray,
    alpha: float,
    experiment_type: api.ExperimentType,
    test_type: api.TestType = api.TestType.TWO_SIDED,
    treatment_cell_id: float = 1.0,
    pretest_spend: Optional[jnp.ndarray] = None,
    test_spend: Optional[jnp.ndarray] = None,
) -> TbrAnalysisResult:
  """Generates analysis metrics for a GeoX experiment using JAX inputs."""

  # Fit model on the training split
  py_train = _compute_group_mean(
      pretest_train_conversions, treatment_mask, treatment_cell_id
  )
  px_train = _compute_group_mean(pretest_train_conversions, treatment_mask, 0.0)
  pred_alpha, pred_beta = _fit_linear_regression(px_train, py_train)

  # Compute out-of-sample RMSE on the validation split
  py_val = _compute_group_mean(
      pretest_val_conversions, treatment_mask, treatment_cell_id
  )
  px_val = _compute_group_mean(pretest_val_conversions, treatment_mask, 0.0)
  py_pred_val = pred_alpha + pred_beta * px_val
  rmse = jnp.sqrt(jnp.mean((py_val - py_pred_val) ** 2))
  scale_val = jnp.mean(jnp.abs(py_val))
  log_errors = methodology_util.compute_regularized_log_ratio(
      py_val, py_pred_val, scale=scale_val
  )
  log_rmse = jnp.sqrt(jnp.mean(log_errors**2))

  # Fit model on the full pretest period for predicting on post-experiment data
  pretest_conversions = jnp.concatenate(
      [pretest_train_conversions, pretest_val_conversions], axis=0
  )
  y_pretest = _compute_group_mean(
      pretest_conversions, treatment_mask, treatment_cell_id
  )
  x_pretest = _compute_group_mean(pretest_conversions, treatment_mask, 0.0)
  pred_alpha, pred_beta = _fit_linear_regression(x_pretest, y_pretest)

  y_pred_pre = pred_alpha + pred_beta * x_pretest

  y_test = _compute_group_mean(
      test_conversions, treatment_mask, treatment_cell_id
  )
  x_test = _compute_group_mean(test_conversions, treatment_mask, 0.0)
  y_pred = pred_alpha + pred_beta * x_test
  y_pred_cumul = jnp.cumsum(y_pred)
  y_test_cumul = jnp.cumsum(y_test)

  geo_avg_cumul_effect = y_test_cumul - y_pred_cumul
  n_treatment_geos = jnp.sum(
      (treatment_mask == treatment_cell_id).astype(float)
  )
  total_cumul_effect = n_treatment_geos * geo_avg_cumul_effect
  total_y_test_cumul = n_treatment_geos * y_test_cumul
  total_y_pred_cumul = n_treatment_geos * y_pred_cumul

  (
      placebo_y_test,
      placebo_y_pred,
      placebo_rmses,
      placebo_log_rmses,
  ) = jax.vmap(
      _compute_placebo_effect_from_mask,
      in_axes=(None, None, None, None, None, 0, None),
  )(
      pretest_conversions,
      pretest_train_conversions,
      pretest_val_conversions,
      test_conversions,
      treatment_mask,
      placebo_masks,
      treatment_cell_id,
  )

  placebo_y_test_cumul = jnp.cumsum(placebo_y_test, axis=1)
  placebo_y_pred_cumul = jnp.cumsum(placebo_y_pred, axis=1)
  placebo_estimates = placebo_y_test_cumul - placebo_y_pred_cumul
  t_placebo = placebo_estimates / jnp.maximum(placebo_rmses[:, None], 1e-9)

  # For Go Dark studies, we negate the incremental metrics.
  sign = -1.0 if experiment_type == api.ExperimentType.GO_DARK else 1.0
  signed_geo_avg_cumul_effect = sign * geo_avg_cumul_effect
  signed_total_cumul_effect = sign * total_cumul_effect
  signed_t_placebo = sign * t_placebo

  geo_averaged_lower_cis, geo_averaged_upper_cis = jax.vmap(
      methodology_util.compute_cis, in_axes=(0, None, 1, None, None)
  )(signed_geo_avg_cumul_effect, rmse, signed_t_placebo, alpha, test_type)
  total_lower_cis = n_treatment_geos * geo_averaged_lower_cis
  total_upper_cis = n_treatment_geos * geo_averaged_upper_cis

  p_value = methodology_util.compute_studentized_p_value(
      signed_geo_avg_cumul_effect[-1],
      rmse,
      signed_t_placebo[:, -1],
      test_type,
  )
  lift_standard_deviation = n_treatment_geos * methodology_util.compute_se(
      rmse, signed_t_placebo[:, -1]  # pyrefly: ignore[bad-argument-type]
  )
  lift = api.Estimate(
      point_estimate=float(signed_total_cumul_effect[-1]),
      lower_bound=float(total_lower_cis[-1]),
      upper_bound=float(total_upper_cis[-1]),
      standard_deviation=float(lift_standard_deviation),
      p_value=float(p_value),
  )
  percent_lift = methodology_util.get_percent_lift(
      total_y_test_cumul[-1],
      total_y_pred_cumul[-1],
      log_rmse,
      placebo_y_test_cumul[:, -1],
      placebo_y_pred_cumul[:, -1],
      placebo_log_rmses,
      alpha,
      experiment_type,
      test_type,
  )

  cumulative_lift_with_cis = np.array(
      jnp.stack([signed_total_cumul_effect, total_lower_cis, total_upper_cis]).T
  )

  # Pointwise metrics.
  geo_avg_pointwise_effect = y_test - y_pred
  total_pointwise_effect = n_treatment_geos * geo_avg_pointwise_effect
  placebo_pointwise_estimates = placebo_y_test - placebo_y_pred
  t_placebo_pointwise = placebo_pointwise_estimates / jnp.maximum(
      placebo_rmses[:, None], 1e-9
  )

  signed_total_pointwise_effect = sign * total_pointwise_effect
  signed_t_placebo_pointwise = sign * t_placebo_pointwise

  geo_averaged_pointwise_lower_cis, geo_averaged_pointwise_upper_cis = jax.vmap(
      methodology_util.compute_cis, in_axes=(0, None, 1, None, None)
  )(
      signed_total_pointwise_effect / n_treatment_geos,
      rmse,
      signed_t_placebo_pointwise,
      alpha,
      test_type,
  )
  total_pointwise_lower_cis = (
      n_treatment_geos * geo_averaged_pointwise_lower_cis
  )
  total_pointwise_upper_cis = (
      n_treatment_geos * geo_averaged_pointwise_upper_cis
  )

  total_y_test = n_treatment_geos * y_test
  total_y_pred = n_treatment_geos * y_pred
  if sign == 1:
    cf_lower = total_y_test - total_pointwise_upper_cis
    cf_upper = total_y_test - total_pointwise_lower_cis
  else:
    cf_lower = total_y_test + total_pointwise_lower_cis
    cf_upper = total_y_test + total_pointwise_upper_cis

  # Combine pre-test and test period for pointwise metrics.
  total_y_pretest = n_treatment_geos * y_pretest
  total_y_pred_pre = n_treatment_geos * y_pred_pre
  total_pretest_pointwise_effect = sign * (total_y_pretest - total_y_pred_pre)

  n_pre = len(y_pretest)
  nan_cis_pre = jnp.full((n_pre,), jnp.nan)

  # (T_pre + T_test, 3)
  pointwise_difference_with_cis = np.array(
      jnp.concatenate([
          jnp.stack([
              total_pretest_pointwise_effect,
              nan_cis_pre,
              nan_cis_pre,
          ]).T,
          jnp.stack([
              signed_total_pointwise_effect,
              total_pointwise_lower_cis,
              total_pointwise_upper_cis,
          ]).T,
      ])
  )

  # (T_pre + T_test, 4)
  counterfactual_conversions_with_cis = np.array(
      jnp.concatenate([
          jnp.stack([
              total_y_pretest,
              total_y_pred_pre,
              nan_cis_pre,
              nan_cis_pre,
          ]).T,
          jnp.stack([total_y_test, total_y_pred, cf_lower, cf_upper]).T,
      ])
  )

  icpd = None
  cumulative_icpd_with_cis = None
  counterfactual_spend = None
  if pretest_spend is not None and test_spend is not None:
    icpd_results = _compute_icpd(
        pretest_spend,
        test_spend,
        treatment_mask,
        signed_total_cumul_effect,
        total_lower_cis,
        total_upper_cis,
        sign,
        treatment_cell_id,
    )
    icpd_standard_deviation = (
        lift_standard_deviation / icpd_results.cumulative_incremental_spend[-1]
    )
    icpd = api.Estimate(
        point_estimate=float(icpd_results.cumulative_icpd[-1]),
        lower_bound=float(icpd_results.lower_bound[-1]),
        upper_bound=float(icpd_results.upper_bound[-1]),
        standard_deviation=float(icpd_standard_deviation),
        p_value=float(p_value),
    )
    cumulative_icpd_with_cis = np.array(
        jnp.stack([
            icpd_results.cumulative_icpd,
            icpd_results.lower_bound,
            icpd_results.upper_bound,
        ]).T
    )
    counterfactual_spend = np.array(icpd_results.counterfactual_spend)

  return TbrAnalysisResult(
      lift=lift,
      cumulative_lift_with_cis=cumulative_lift_with_cis,
      percent_lift=percent_lift,
      icpd=icpd,
      cumulative_icpd_with_cis=cumulative_icpd_with_cis,
      counterfactual_conversions_with_cis=counterfactual_conversions_with_cis,
      pointwise_difference_with_cis=pointwise_difference_with_cis,
      counterfactual_spend=counterfactual_spend,
  )
