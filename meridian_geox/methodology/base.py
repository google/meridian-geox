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

"""Base abstractions and result containers for GeoX methodologies."""

import abc
import dataclasses

import jax
import jax.numpy as jnp
from meridian_geox import api
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
class MethodologyAnalysisResult:
  """Results of a counterfactual methodology analysis.

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
  icpd: api.Estimate | None = None
  cumulative_icpd_with_cis: np.ndarray | None = None
  counterfactual_conversions_with_cis: np.ndarray = dataclasses.field(
      default_factory=lambda: np.array([])
  )
  pointwise_difference_with_cis: np.ndarray = dataclasses.field(
      default_factory=lambda: np.array([])
  )
  counterfactual_spend: np.ndarray | None = None


def compute_binary_group_mean(
    data: jnp.ndarray,
    binary_mask: jnp.ndarray,
) -> jnp.ndarray:
  """Computes mean time series for a binary float32 geo mask.

  Args:
    data: (T, N_geos) Time series data.
    binary_mask: (N_geos,) Binary float32 mask (1.0 for included geos, 0.0
      otherwise).

  Returns:
    mean_ts: (T,) Average time series for the masked group.
  """
  raise NotImplementedError()


def compute_group_mean(
    data: jnp.ndarray,
    mask: jnp.ndarray,
    cell_id: float,
) -> jnp.ndarray:
  """Computes mean time series for a given cell.

  Args:
    data: (T, N_geos) Time series data.
    mask: (N_geos,) Treatment mask (0.0 for control, positive integer for
      treatment).
    cell_id: Cell ID.

  Returns:
    mean_ts: (T,) Average time series for the cell.
  """
  raise NotImplementedError()


class CounterfactualMethodology(abc.ABC):
  """Abstract base class for GeoX counterfactual methodologies.

  Subclasses implement `_fit_and_predict_single` (and optionally override
  `check_candidate_compatibility` or `get_r2`). The base class defines the
  lifecycle interface for out-of-sample R-squared screening, simplified
  design-aware MDE calculation, and post-experiment studentized placebo
  inference.
  """

  @abc.abstractmethod
  def _fit_and_predict_single(
      self,
      data_train: jnp.ndarray,
      data_eval: jnp.ndarray,
      treatment_group_mask: jnp.ndarray,
      control_group_mask: jnp.ndarray,
  ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Fits on `data_train` and predicts geo-average counterfactuals.

    Args:
      data_train: (T_train, N_geos) Training period time series.
      data_eval: (T_eval, N_geos) Evaluation period time series.
      treatment_group_mask: (N_geos,) Binary float32 mask (1.0 for treated geos
        in the target cell, 0.0 otherwise).
      control_group_mask: (N_geos,) Binary float32 mask (1.0 for valid control
        donor geos, 0.0 otherwise).

    Returns:
      Tuple of (y_train, y_pred_train, y_eval, y_pred_eval), each of shape
      (T_train,) or (T_eval,), representing the per-geo average time series for
      the treated group.
    """
    raise NotImplementedError()

  def check_candidate_compatibility(
      self,
      candidates: jnp.ndarray,
      conversion_data: jnp.ndarray,
      spend_data: jnp.ndarray,
      design_config: api.DesignConfig,
      cell_id: float,
  ) -> jnp.ndarray:
    """Checks if candidates satisfy methodology-specific compatibility criteria.

    Args:
      candidates: (N_designs, N_geos) Treatment masks.
      conversion_data: (T, N_geos) Conversion time series data.
      spend_data: (T, N_geos) Spend time series data.
      design_config: Study design configuration.
      cell_id: Cell ID for multicell designs.

    Returns:
      mask: (N_designs,) Boolean mask where True indicates the design passed.
    """
    raise NotImplementedError()

  def get_r2(
      self,
      data_pre: jnp.ndarray,
      data_val: jnp.ndarray,
      treatment_masks: jnp.ndarray,
      cell_ids: jnp.ndarray,
  ) -> jnp.ndarray:
    """Calculates validation R-squared for a batch of designs.

    Args:
      data_pre: (T_pre, N_geos) Pre-period data.
      data_val: (T_val, N_geos) Validation-period data.
      treatment_masks: (N_designs, N_geos) Batch of treatment masks.
      cell_ids: (k_cells,) Cell IDs.

    Returns:
      r2_vals: (N_designs, k_cells) Out-of-sample R-squared on validation
        period.
    """
    raise NotImplementedError()

  def get_mde(
      self,
      data_pre: jnp.ndarray,
      data_val: jnp.ndarray,
      treatment_masks: jnp.ndarray,
      z_score_sum: float,
      test_type: api.TestType = api.TestType.TWO_SIDED,
      cell_ids: jnp.ndarray | None = None,
  ) -> MdeResults:
    """Runs placebo check and calculates MDE for a batch of designs.

    Args:
      data_pre: (T_pre, N_geos) Pre-period data.
      data_val: (T_val, N_geos) Validation-period data.
      treatment_masks: (N_designs, N_geos) Batch of treatment masks.
      z_score_sum: Sum of z-scores for alpha and power.
      test_type: One-sided or two-sided statistical test.
      cell_ids: Optional (k_cells,) Cell IDs. Defaults to [1.0].

    Returns:
      MdeResults containing absolute/relative MDE, AA p-values, and time series.
    """
    raise NotImplementedError()

  def _compute_placebo_effect_from_mask(
      self,
      data_pretest: jnp.ndarray,
      data_pretest_train: jnp.ndarray,
      data_pretest_val: jnp.ndarray,
      data_test: jnp.ndarray,
      mask: jnp.ndarray,
      p_mask: jnp.ndarray,
      treatment_cell_id: float,
  ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Computes test effect and out-of-sample RMSE for a single placebo mask.

    Args:
      data_pretest: (T_pre, N_geos) Pre-period data.
      data_pretest_train: (T_pre_train, N_geos) Training pre-period data.
      data_pretest_val: (T_pre_val, N_geos) Validation pre-period data.
      data_test: (T_test, N_geos) Test-period data.
      mask: (N_geos,) Treatment mask (0.0 for control, positive integer cell ID
        for treatment).
      p_mask: (N_geos,) Placebo treatment mask (0.0 for control, positive
        integer cell ID for placebo treatment).
      treatment_cell_id: The Cell ID to use for the treated group.

    Returns:
      Placebo treatment response, Placebo predicted response, Placebo RMSE
      (out-of-sample), Placebo log RMSE (out-of-sample).
    """
    raise NotImplementedError()

  def _compute_icpd(
      self,
      pretest_spend: jnp.ndarray,
      test_spend: jnp.ndarray,
      treatment_mask: jnp.ndarray,
      total_cumul_effect: jnp.ndarray,
      total_lower_cis: jnp.ndarray,
      total_upper_cis: jnp.ndarray,
      incremental_spend_sign: float,
      treatment_cell_id: float = 1.0,
  ) -> IcpdResults:
    """Computes ICPD metrics using JAX arrays."""
    raise NotImplementedError()

  def analyze(
      self,
      pretest_train_conversions: jnp.ndarray,
      pretest_val_conversions: jnp.ndarray,
      test_conversions: jnp.ndarray,
      treatment_mask: jnp.ndarray,
      placebo_masks: jnp.ndarray,
      alpha: float,
      experiment_type: api.ExperimentType,
      test_type: api.TestType = api.TestType.TWO_SIDED,
      treatment_cell_id: float = 1.0,
      pretest_spend: jnp.ndarray | None = None,
      test_spend: jnp.ndarray | None = None,
  ) -> MethodologyAnalysisResult:
    """Generates analysis metrics for a GeoX experiment using JAX inputs.

    Args:
      pretest_train_conversions: (T_pre_train, N_geos) Training pre-period data.
      pretest_val_conversions: (T_pre_val, N_geos) Validation pre-period data.
      test_conversions: (T_test, N_geos) Test-period conversion data.
      treatment_mask: (N_geos,) Treatment assignment mask.
      placebo_masks: (N_placebos, N_geos) Placebo assignment masks.
      alpha: Significance level for confidence intervals.
      experiment_type: Experiment type (HOLDBACK, GO_DARK, HEAVY_UP).
      test_type: One-sided or two-sided statistical test.
      treatment_cell_id: Target treatment cell ID.
      pretest_spend: Optional (T_pre, N_geos) pre-period spend data.
      test_spend: Optional (T_test, N_geos) test-period spend data.

    Returns:
      MethodologyAnalysisResult with lift, percent_lift, CIs, and optional iCPD.
    """
    raise NotImplementedError()
