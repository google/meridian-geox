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

"""Utility functions for GeoX methodologies."""

import functools
import math

import jax
import jax.numpy as jnp
from meridian_geox import api


def _get_order_statistic(
    values: jnp.ndarray, level: float, scale: jnp.ndarray | float
) -> jnp.ndarray:
  """Returns `scale` times the finite-sample `level` quantile of `values`.

  The quantile is the k-th smallest value with k = ceil(level * (n + 1)), which
  makes placebo confidence bounds exactly dual to the placebo p-values.

  Args:
    values: A 1D array of n values.
    level: The quantile level.
    scale: The multiplier applied to the quantile.

  Returns:
    `scale` times the k-th smallest value, or inf if k > n.
  """
  n = values.shape[0]
  # The tolerance avoids rounding up when level * (n + 1) is an integer.
  k = math.ceil(level * (n + 1) - 1e-9)
  if k > n:
    return jnp.array(jnp.inf)
  return jnp.sort(values)[max(k, 1) - 1] * scale


def _get_two_sided_p_value(
    t_obs: jnp.ndarray, t_placebo: jnp.ndarray
) -> jnp.ndarray:
  """Returns (1 + #{|t_j| >= |t_obs|}) / (J + 1)."""
  n_extreme = jnp.sum(jnp.abs(t_placebo) >= jnp.abs(t_obs))
  return (1.0 + n_extreme) / (1.0 + t_placebo.shape[0])


def _get_one_sided_p_value(
    t_obs: jnp.ndarray, t_placebo: jnp.ndarray
) -> jnp.ndarray:
  """Returns the one-sided p-value for a positive alternative."""
  p_two_sided = _get_two_sided_p_value(t_obs, t_placebo)
  return jnp.where(t_obs > 0, p_two_sided / 2.0, 1.0 - p_two_sided / 2.0)


@functools.partial(
    jax.jit, static_argnames=['test_type', 'use_absolute_placebo_statistics']
)
def compute_studentized_p_value(
    estimate: float,
    rmse: float,
    t_placebo: jnp.ndarray,
    test_type: api.TestType = api.TestType.TWO_SIDED,
    use_absolute_placebo_statistics: bool = False,
) -> jnp.ndarray:
  """Computes a studentized p-value based on placebo simulations.

  If `use_absolute_placebo_statistics` is True, the two-sided p-value compares
  |t_obs| with the absolute placebo t-statistics, and the one-sided p-value, for
  a positive alternative, is half the two-sided p-value if t_obs > 0 and one
  minus that otherwise.

  Args:
    estimate: The observed effect estimate.
    rmse: The Root Mean Squared Error from the real fit.
    t_placebo: An array of t-statistics from placebo simulations.
    test_type: The type of test to perform (e.g., TWO_SIDED).
    use_absolute_placebo_statistics: Whether to use the absolute placebo
      t-statistics.

  Returns:
    The computed p-value.
  """
  t_obs = estimate / jnp.maximum(rmse, 1e-9)
  if use_absolute_placebo_statistics:
    if test_type == api.TestType.TWO_SIDED:
      return _get_two_sided_p_value(t_obs, t_placebo)
    return _get_one_sided_p_value(t_obs, t_placebo)

  n_placebo = len(t_placebo)
  if test_type == api.TestType.TWO_SIDED:
    n_extreme_left = jnp.sum(t_placebo <= t_obs)
    n_extreme_right = jnp.sum(t_placebo >= t_obs)
    n_extreme = 2 * jnp.minimum(n_extreme_left, n_extreme_right)
    n_extreme = jnp.minimum(n_extreme, n_placebo)
  else:
    n_extreme = jnp.sum(t_placebo >= t_obs)

  p_value = (1.0 + n_extreme) / (1.0 + n_placebo)
  return p_value


@jax.jit
def compute_se(
    rmse: float,
    t_placebo: jnp.ndarray,
):
  """Computes the standard error based on placebo simulations."""
  return jnp.std(t_placebo) * rmse


@functools.partial(jax.jit, static_argnames=['test_type'])
def _compute_signed_cis(
    estimate: jnp.ndarray | float,
    rmse: jnp.ndarray | float,
    t_placebo: jnp.ndarray,
    alpha: float,
    test_type: api.TestType,
) -> tuple[jnp.ndarray, jnp.ndarray]:
  """Computes confidence intervals from the signed placebo t-statistics."""
  if test_type == api.TestType.TWO_SIDED:
    lower = estimate - jnp.quantile(t_placebo, 1 - alpha / 2) * rmse
    upper = estimate - jnp.quantile(t_placebo, alpha / 2) * rmse
  else:
    lower = estimate - jnp.quantile(t_placebo, 1 - alpha) * rmse
    upper = jnp.array(jnp.inf)

  return lower, upper


@functools.partial(jax.jit, static_argnames=['alpha', 'test_type'])
def _compute_absolute_cis(
    estimate: jnp.ndarray | float,
    rmse: jnp.ndarray | float,
    t_placebo: jnp.ndarray,
    alpha: float,
    test_type: api.TestType,
) -> tuple[jnp.ndarray, jnp.ndarray]:
  """Computes confidence intervals from the absolute placebo t-statistics."""
  if test_type == api.TestType.TWO_SIDED:
    margin = _get_order_statistic(jnp.abs(t_placebo), 1.0 - alpha, rmse)
    lower = estimate - margin
    upper = estimate + margin
  else:
    margin = _get_order_statistic(jnp.abs(t_placebo), 1.0 - 2.0 * alpha, rmse)
    lower = estimate - margin
    upper = jnp.array(jnp.inf)

  return lower, upper


def compute_cis(
    estimate: jnp.ndarray | float,
    rmse: jnp.ndarray | float,
    t_placebo: jnp.ndarray,
    alpha: float,
    test_type: api.TestType = api.TestType.TWO_SIDED,
    use_absolute_placebo_statistics: bool = False,
) -> tuple[jnp.ndarray, jnp.ndarray]:
  """Computes confidence intervals based on placebo simulations.

  If `use_absolute_placebo_statistics` is True, the two-sided interval is
  centered on the estimate, with half-width equal to the finite-sample
  (1 - alpha) quantile of the absolute placebo t-statistics times rmse, and the
  one-sided lower bound is the lower end of the two-sided interval at level
  2 * alpha. These bounds are infinite if there are too few placebos for the
  given alpha.

  Args:
    estimate: The observed effect estimate.
    rmse: The Root Mean Squared Error from the real fit.
    t_placebo: An array of t-statistics from placebo simulations.
    alpha: The significance level (e.g., 0.1 for a 90% CI).
    test_type: The type of test to perform (e.g., TWO_SIDED).
    use_absolute_placebo_statistics: Whether to use the absolute placebo
      t-statistics.

  Returns:
    The lower and upper confidence bounds.
  """
  if use_absolute_placebo_statistics:
    return _compute_absolute_cis(estimate, rmse, t_placebo, alpha, test_type)
  return _compute_signed_cis(estimate, rmse, t_placebo, alpha, test_type)


def _get_signed_percent_lift_inference(
    log_estimate: jnp.ndarray,
    log_rmse: jnp.ndarray,
    t_placebo: jnp.ndarray,
    alpha: float,
    experiment_type: api.ExperimentType,
    test_type: api.TestType,
) -> tuple[jnp.ndarray, jnp.ndarray | float, jnp.ndarray | float]:
  """Returns the p-value and percent lift bounds from signed t-statistics."""
  t_obs = log_estimate / jnp.maximum(log_rmse, 1e-9)
  n_placebo = len(t_placebo)

  if test_type == api.TestType.TWO_SIDED:
    n_extreme_left = jnp.sum(t_placebo <= t_obs)
    n_extreme_right = jnp.sum(t_placebo >= t_obs)
    n_extreme = 2 * jnp.minimum(n_extreme_left, n_extreme_right)
    n_extreme = jnp.minimum(n_extreme, n_placebo)
    lower_ci = (
        jnp.exp(
            log_estimate - jnp.quantile(t_placebo, 1.0 - alpha / 2.0) * log_rmse
        )
        - 1.0
    )
    upper_ci = (
        jnp.exp(log_estimate - jnp.quantile(t_placebo, alpha / 2.0) * log_rmse)
        - 1.0
    )
  else:
    if experiment_type == api.ExperimentType.GO_DARK:
      n_extreme = jnp.sum(t_placebo <= t_obs)
      lower_ci = -jnp.inf
      upper_ci = (
          jnp.exp(log_estimate - jnp.quantile(t_placebo, alpha) * log_rmse)
          - 1.0
      )
    else:
      n_extreme = jnp.sum(t_placebo >= t_obs)
      lower_ci = (
          jnp.exp(
              log_estimate - jnp.quantile(t_placebo, 1.0 - alpha) * log_rmse
          )
          - 1.0
      )
      upper_ci = jnp.inf

  p_value = (1.0 + n_extreme) / (1.0 + n_placebo)
  return p_value, lower_ci, upper_ci


def _get_absolute_percent_lift_inference(
    log_estimate: jnp.ndarray,
    log_rmse: jnp.ndarray,
    t_placebo: jnp.ndarray,
    alpha: float,
    experiment_type: api.ExperimentType,
    test_type: api.TestType,
) -> tuple[jnp.ndarray, jnp.ndarray | float, jnp.ndarray | float]:
  """Returns the p-value and percent lift bounds from absolute t-statistics."""
  t_obs = log_estimate / jnp.maximum(log_rmse, 1e-9)

  if test_type == api.TestType.TWO_SIDED:
    p_value = _get_two_sided_p_value(t_obs, t_placebo)
    log_margin = _get_order_statistic(jnp.abs(t_placebo), 1.0 - alpha, log_rmse)
    lower_ci = jnp.exp(log_estimate - log_margin) - 1.0
    upper_ci = jnp.exp(log_estimate + log_margin) - 1.0
  else:
    log_margin = _get_order_statistic(
        jnp.abs(t_placebo), 1.0 - 2.0 * alpha, log_rmse
    )
    if experiment_type == api.ExperimentType.GO_DARK:
      # Go Dark tests for a decrease.
      p_value = _get_one_sided_p_value(-t_obs, t_placebo)
      lower_ci = -jnp.inf
      upper_ci = jnp.exp(log_estimate + log_margin) - 1.0
    else:
      p_value = _get_one_sided_p_value(t_obs, t_placebo)
      lower_ci = jnp.exp(log_estimate - log_margin) - 1.0
      upper_ci = jnp.inf

  return p_value, lower_ci, upper_ci


def get_percent_lift(
    y_test: jnp.ndarray,
    y_pred: jnp.ndarray,
    log_rmse: jnp.ndarray,
    y_test_placebos: jnp.ndarray,
    y_pred_placebos: jnp.ndarray,
    placebo_log_rmses: jnp.ndarray,
    alpha: float,
    experiment_type: api.ExperimentType,
    test_type: api.TestType = api.TestType.TWO_SIDED,
    use_absolute_placebo_statistics: bool = False,
) -> api.Estimate:
  """Calculates percent lift with placebo-adjusted confidence intervals.

    This uses a log transformation to calculate the CI for the percent lift. We
    compute the estimate and CI of log(y_test / y_pred) and then exponentiate
    and subtract 1 to get the estimate and CI for the percent lift:
      percent_lift = (y_test - y_pred) / y_pred = y_test / y_pred - 1

  Args:
    y_test: The observed treatment response.
    y_pred: The predicted counterfactual treatment response.
    log_rmse: The Root Mean Squared Error of the log-transformed values from the
      real fit.
    y_test_placebos: Array of observed treatment responses from placebo
      simulations.
    y_pred_placebos: Array of predicted counterfactual treatment responses from
      placebo simulations.
    placebo_log_rmses: Array of log RMSEs from placebo simulations.
    alpha: The confidence level (e.g., 0.1 for a 90% CI).
    experiment_type: The type of experiment (e.g., GO_DARK).
    test_type: The type of test to perform (e.g., TWO_SIDED).
    use_absolute_placebo_statistics: Whether to use the absolute placebo
      t-statistics.

  Returns:
    An Estimate object containing the point estimate and confidence interval
    for the percent lift.
  """
  log_estimate = compute_regularized_log_ratio(y_test, y_pred)
  log_placebo_estimates = compute_regularized_log_ratio(
      y_test_placebos, y_pred_placebos
  )
  t_placebo = log_placebo_estimates / jnp.maximum(placebo_log_rmses, 1e-9)

  if use_absolute_placebo_statistics:
    p_value, lower_ci, upper_ci = _get_absolute_percent_lift_inference(
        log_estimate, log_rmse, t_placebo, alpha, experiment_type, test_type
    )
  else:
    p_value, lower_ci, upper_ci = _get_signed_percent_lift_inference(
        log_estimate, log_rmse, t_placebo, alpha, experiment_type, test_type
    )

  estimate = jnp.exp(log_estimate) - 1.0

  se_log = compute_se(log_rmse, t_placebo)  # pyrefly: ignore[bad-argument-type]
  standard_deviation = jnp.exp(log_estimate) * se_log

  if experiment_type == api.ExperimentType.GO_DARK:
    estimate = -estimate
    lower_ci, upper_ci = -upper_ci, -lower_ci

  return api.Estimate(
      point_estimate=float(estimate),
      lower_bound=float(lower_ci),
      upper_bound=float(upper_ci),
      standard_deviation=float(standard_deviation),
      p_value=float(p_value),
  )


def compute_regularized_log_ratio(
    y_num: jnp.ndarray | float,
    y_den: jnp.ndarray | float,
    scale: jnp.ndarray | float | None = None,
) -> jnp.ndarray:
  """Computes log(y_num / y_den) with dynamic clipping to avoid instability.

  If `scale` is provided, it is used to determine the clipping threshold.
  Otherwise, a pointwise scale is computed from the inputs.

  Args:
    y_num: Numerator.
    y_den: Denominator.
    scale: Optional scale for determining the clipping threshold.

  Returns:
    The regularized log ratio.
  """
  if scale is None:
    scale = jnp.maximum(jnp.abs(y_num), jnp.abs(y_den))
  threshold = jnp.maximum(1e-3 * scale, 1e-9)
  return jnp.log(
      jnp.maximum(y_num, threshold) / jnp.maximum(y_den, threshold)
  )
