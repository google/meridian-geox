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


@functools.partial(jax.jit, static_argnames=['test_type'])
def compute_studentized_p_value(
    estimate: float,
    rmse: float,
    t_placebo: jnp.ndarray,
    test_type: api.TestType = api.TestType.TWO_SIDED,
) -> jnp.ndarray:
  """Computes a studentized p-value based on placebo simulations.

  Args:
    estimate: The observed effect estimate.
    rmse: The Root Mean Squared Error from the real fit.
    t_placebo: An array of t-statistics from placebo simulations.
    test_type: The type of test to perform (e.g., TWO_SIDED).

  Returns:
    The computed p-value.
  """
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


@jax.jit
def compute_se(
    rmse: float,
    t_placebo: jnp.ndarray,
):
  """Computes the standard error based on placebo simulations."""
  return jnp.std(t_placebo) * rmse


@functools.partial(jax.jit, static_argnames=['test_type'])
def compute_cis(
    estimate: float,
    rmse: float,
    t_placebo: jnp.ndarray,
    alpha: float,
    test_type: api.TestType = api.TestType.TWO_SIDED,
) -> tuple[jnp.ndarray, jnp.ndarray]:
  """Computes confidence intervals based on placebo simulations."""
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

  Returns:
    An Estimate object containing the point estimate and confidence interval
    for the percent lift.
  """
  log_estimate = compute_regularized_log_ratio(y_test, y_pred)
  t_obs = log_estimate / jnp.maximum(log_rmse, 1e-9)

  log_placebo_estimates = compute_regularized_log_ratio(
      y_test_placebos, y_pred_placebos
  )
  t_placebo = log_placebo_estimates / jnp.maximum(placebo_log_rmses, 1e-9)
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
  estimate = jnp.exp(log_estimate) - 1.0

  se_log = compute_se(log_rmse, t_placebo)
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
