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

import jax
import jax.numpy as jnp
from meridian_geox import api


def _get_order_statistic(
    values: jnp.ndarray,
    level: jnp.ndarray | float,
    scale: jnp.ndarray | float,
) -> jnp.ndarray:
  """Returns `scale` times the finite-sample `level` quantile of `values`.

  The quantile is the k-th smallest value with k = ceil(level * (n + 1)), which
  makes placebo confidence bounds dual to the placebo p-values.

  Args:
    values: A 1D array of n values.
    level: The quantile level.
    scale: The multiplier applied to the quantile.

  Returns:
    `scale` times the k-th smallest value, or inf if k > n.
  """
  n = values.shape[0]
  if n == 0:
    return jnp.array(jnp.inf)
  # The tolerance avoids rounding up when level * (n + 1) is an integer.
  k = jnp.ceil((level - 1e-6) * (n + 1)).astype(jnp.int32)
  finite_bound = jnp.sort(values)[jnp.clip(k - 1, 0, n - 1)] * scale
  return jnp.where(k > n, jnp.inf, finite_bound)


def _compute_absolute_p_value(
    estimate: jnp.ndarray | float,
    rmse: jnp.ndarray | float,
    t_placebo: jnp.ndarray,
    test_type: api.TestType,
) -> float:
  """Computes the p-value from the absolute placebo t-statistics."""
  t_obs = estimate / jnp.maximum(rmse, 1e-9)
  n_placebo = t_placebo.shape[0]
  # Dividing in float64 makes p exactly alpha when the ratio equals alpha.
  if test_type == api.TestType.TWO_SIDED:
    n_extreme = int(jnp.sum(jnp.abs(t_placebo) >= jnp.abs(t_obs)))
    return (1.0 + n_extreme) / (1.0 + n_placebo)
  # The one-sided reference values are t_obs, -t_obs and +-t_j.
  n_extreme = int(
      jnp.sum(t_placebo >= t_obs)
      + jnp.sum(-t_placebo >= t_obs)
      + (-t_obs >= t_obs)
  )
  return (1.0 + n_extreme) / (2.0 * (1.0 + n_placebo))


@functools.partial(jax.jit, static_argnames=['test_type'])
def _compute_signed_p_value(
    estimate: jnp.ndarray | float,
    rmse: jnp.ndarray | float,
    t_placebo: jnp.ndarray,
    test_type: api.TestType,
) -> jnp.ndarray:
  """Computes the p-value from the signed placebo t-statistics."""
  t_obs = estimate / jnp.maximum(rmse, 1e-9)
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


def compute_studentized_p_value(
    estimate: jnp.ndarray | float,
    rmse: jnp.ndarray | float,
    t_placebo: jnp.ndarray,
    test_type: api.TestType = api.TestType.TWO_SIDED,
    use_absolute_placebo_statistics: bool = False,
) -> jnp.ndarray | float:
  """Computes a studentized p-value based on placebo simulations.

  If `use_absolute_placebo_statistics` is True, the two-sided p-value compares
  |t_obs| with the absolute placebo t-statistics, and the one-sided p-value, for
  a positive alternative, compares t_obs with -t_obs and with the placebo
  t-statistics and their negations. For t_obs > 0, the one-sided p-value is
  half the two-sided p-value.

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
  if use_absolute_placebo_statistics:
    return _compute_absolute_p_value(estimate, rmse, t_placebo, test_type)
  return _compute_signed_p_value(estimate, rmse, t_placebo, test_type)


@functools.partial(jax.jit, static_argnames=['use_absolute_placebo_statistics'])
def compute_se(
    rmse: jnp.ndarray | float,
    t_placebo: jnp.ndarray,
    use_absolute_placebo_statistics: bool = False,
) -> jnp.ndarray:
  """Computes the standard error based on placebo simulations.

  If `use_absolute_placebo_statistics` is True, the standard deviation of the
  placebo t-statistics is replaced by their root mean square, which is the
  standard deviation of {t, -t}.

  Args:
    rmse: The Root Mean Squared Error from the real fit.
    t_placebo: An array of t-statistics from placebo simulations.
    use_absolute_placebo_statistics: Whether to use the absolute placebo
      t-statistics.

  Returns:
    The standard error.
  """
  if use_absolute_placebo_statistics:
    return jnp.sqrt(jnp.mean(t_placebo**2)) * rmse
  return jnp.std(t_placebo) * rmse


@functools.partial(
    jax.jit, static_argnames=['test_type', 'use_absolute_placebo_statistics']
)
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
    # The same rmse floor as in the p-value keeps the bounds dual to it.
    scale = jnp.maximum(rmse, 1e-9)
    if test_type == api.TestType.TWO_SIDED:
      margin = _get_order_statistic(jnp.abs(t_placebo), 1.0 - alpha, scale)
      return estimate - margin, estimate + margin
    margin = _get_order_statistic(jnp.abs(t_placebo), 1.0 - 2.0 * alpha, scale)
    return estimate - margin, jnp.array(jnp.inf)

  if test_type == api.TestType.TWO_SIDED:
    lower = estimate - jnp.quantile(t_placebo, 1 - alpha / 2) * rmse
    upper = estimate - jnp.quantile(t_placebo, alpha / 2) * rmse
  else:
    lower = estimate - jnp.quantile(t_placebo, 1 - alpha) * rmse
    upper = jnp.array(jnp.inf)

  return lower, upper


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
  compute the estimate and CI of log(y_test / y_pred) and then exponentiate and
  subtract 1 to get the estimate and CI for the percent lift:
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
    alpha: The significance level (e.g., 0.1 for a 90% CI).
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

  # For Go Dark studies, inference is done on the negated log effects.
  sign = -1.0 if experiment_type == api.ExperimentType.GO_DARK else 1.0
  p_value = compute_studentized_p_value(
      sign * log_estimate,
      log_rmse,
      sign * t_placebo,
      test_type,
      use_absolute_placebo_statistics=use_absolute_placebo_statistics,
  )
  signed_lower, signed_upper = compute_cis(
      sign * log_estimate,
      log_rmse,
      sign * t_placebo,
      alpha,
      test_type,
      use_absolute_placebo_statistics,
  )
  if experiment_type == api.ExperimentType.GO_DARK:
    log_lower, log_upper = -signed_upper, -signed_lower
  else:
    log_lower, log_upper = signed_lower, signed_upper
  # Infinite log bounds stay infinite; exp(-inf) - 1 would be -1.
  lower_ci = jnp.where(jnp.isinf(log_lower), -jnp.inf, jnp.exp(log_lower) - 1.0)
  upper_ci = jnp.where(jnp.isinf(log_upper), jnp.inf, jnp.exp(log_upper) - 1.0)

  estimate = jnp.exp(log_estimate) - 1.0

  se_log = compute_se(
      log_rmse,
      t_placebo,
      use_absolute_placebo_statistics=use_absolute_placebo_statistics,
  )
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
