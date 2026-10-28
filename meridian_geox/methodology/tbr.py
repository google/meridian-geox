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

import jax
import jax.numpy as jnp
from meridian_geox import api
from meridian_geox.methodology import base
from meridian_geox.methodology import registry


MdeResults = base.MdeResults
IcpdResults = base.IcpdResults
TbrAnalysisResult = base.MethodologyAnalysisResult


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


@registry.register_methodology(api.Methodology.TBR)
class TbrMethodology(base.CounterfactualMethodology):
  """Time-Based Regression (TBR) counterfactual methodology."""

  def _fit_and_predict_single(
      self,
      data_train: jnp.ndarray,
      data_eval: jnp.ndarray,
      treatment_group_mask: jnp.ndarray,
      control_group_mask: jnp.ndarray,
  ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Fits TBR on `data_train` and predicts geo-average counterfactuals.

    TBR regresses the treated-group mean time series on the control-group mean
    time series with a non-negative slope (see `_fit_linear_regression`).

    Args:
      data_train: (T_train, N_geos) Training period time series.
      data_eval: (T_eval, N_geos) Evaluation period time series.
      treatment_group_mask: (N_geos,) Binary float32 mask of treated geos.
      control_group_mask: (N_geos,) Binary float32 mask of control geos.

    Returns:
      Tuple of (y_train, y_pred_train, y_eval, y_pred_eval) treated-group mean
      time series and their TBR predictions.
    """
    y_train = base.compute_binary_group_mean(data_train, treatment_group_mask)
    x_train = base.compute_binary_group_mean(data_train, control_group_mask)
    alpha, beta = _fit_linear_regression(x_train, y_train)
    y_eval = base.compute_binary_group_mean(data_eval, treatment_group_mask)
    x_eval = base.compute_binary_group_mean(data_eval, control_group_mask)
    return y_train, alpha + beta * x_train, y_eval, alpha + beta * x_eval

  def check_candidate_compatibility(
      self,
      candidates: jnp.ndarray,
      conversion_data: jnp.ndarray,
      spend_data: jnp.ndarray,
      design_config: api.DesignConfig,
      cell_id: float,
  ) -> jnp.ndarray:
    """Checks if the TBR slope between conversion and spend is similar."""
    return check_slope_similarity(
        candidates=candidates,
        conversion_data=conversion_data,
        spend_data=spend_data,
        tolerance=design_config.slope_tolerance,
        cell_id=cell_id,
    )


_TBR_METHODOLOGY = registry.get_methodology(api.Methodology.TBR)

# Backward-compatible module-level entry points. These now dispatch to the
# shared `CounterfactualMethodology` engine via `TbrMethodology`.
get_r2 = _TBR_METHODOLOGY.get_r2
get_mde = _TBR_METHODOLOGY.get_mde
analyze = _TBR_METHODOLOGY.analyze
