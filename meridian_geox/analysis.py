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
from typing import Optional

import jax
import jax.numpy as jnp
from meridian_geox import api
from meridian_geox import design
from meridian_geox import generate_candidates
from meridian_geox import util
from meridian_geox.data_quality import data_quality
from meridian_geox.methodology import tbr
import numpy as np
import pandas as pd


@dataclasses.dataclass
class TimeSeries:
  """Time series arrays for analysis."""

  pretest: jnp.ndarray
  test: jnp.ndarray
  dates: pd.Index


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
  pretest = jnp.array(
      pivoted_data[
          pivoted_data.index < analysis_config.analysis_start_date
      ].values
  )
  test_data = pivoted_data[
      (pivoted_data.index >= analysis_config.analysis_start_date)
      & (pivoted_data.index <= analysis_config.analysis_end_date)
  ]
  test = jnp.array(test_data.values)
  return TimeSeries(pretest=pretest, test=test, dates=test_data.index)


def _get_placebo_masks(
    treatment_mask: jnp.ndarray,
    placebo_data: pd.DataFrame,
    design_config: api.DesignConfig,
    constraints: Optional[api.Constraints],
    key: jax.Array,
    geo_stratum_labels: Optional[jnp.ndarray] = None,
) -> jnp.ndarray:
  """Generates placebo masks for a GeoX experiment."""
  # TODO: Refactor to move placebo mask generation to design phase.
  processed_placebo_data = design.prepare_data(
      data=placebo_data,
      experiment_duration=design_config.experiment_duration,
      constraints=constraints,
  )

  if design_config.geo_assignment_rule == api.GeoAssignmentRule.RANDOM:
    placebo_masks_without_treatment_geos = (
        generate_candidates.get_random_candidates(
            filtered_data=processed_placebo_data.filtered_data,
            design_config=design_config,
            constraints=constraints,
            key=key,
            selection_train=processed_placebo_data.selection_train,
            selection_train_spend=processed_placebo_data.selection_train_spend,
        )
    )
  elif (
      design_config.geo_assignment_rule
      == api.GeoAssignmentRule.STRATIFIED_SAMPLING
  ):
    if geo_stratum_labels is None:
      raise ValueError(
          'geo_stratum_labels is required for stratified sampling.'
      )
    placebo_masks_without_treatment_geos = (
        generate_candidates.get_stratified_sampling_candidates(
            selection_train=processed_placebo_data.selection_train,
            filtered_data=processed_placebo_data.filtered_data,
            design_config=design_config,
            constraints=constraints,
            geo_stratum_labels=geo_stratum_labels[treatment_mask == 0],
            key=key,
            selection_train_spend=processed_placebo_data.selection_train_spend,
        )
    )
  else:
    raise ValueError(
        f'Unsupported geo assignment rule: {design_config.geo_assignment_rule}'
    )

  return jax.vmap(_get_full_mask, in_axes=(None, 0))(
      treatment_mask, placebo_masks_without_treatment_geos
  )


def _prepare_design_config(
    analysis_config: api.AnalysisConfig,
) -> api.DesignConfig:
  """Prepares the design config for analysis."""
  if analysis_config.design.design_config is None:
    raise ValueError('Design config is required for analysis.')

  design_config = dataclasses.replace(
      analysis_config.design.design_config,
      n_candidates=analysis_config.design.design_config.n_aa_test_iterations,
  )

  if not analysis_config.alpha:
    analysis_config.alpha = design_config.alpha

  if analysis_config.test_type is None:
    analysis_config.test_type = design_config.test_type

  return design_config


def _get_experiment_type(
    design_config: api.DesignConfig,
) -> api.ExperimentType:
  """Extracts and validates a single experiment type."""
  experiment_types = design_config.experiment_types
  if isinstance(experiment_types, api.ExperimentType):
    experiment_types = [experiment_types]

  if len(experiment_types) != 1:
    raise ValueError(
        'We currently only support studies with a single experiment type.'
    )

  return experiment_types[0]


def _get_treatment_mask(
    data: pd.DataFrame,
    design_obj: api.Design,
) -> TreatmentMask:
  """Creates a binary treatment mask and returns treatment geo names."""
  geos = sorted(data[api.LOCATION].unique())
  # TODO Add multicell support.
  treatment_geos = sorted(list(list(design_obj.treatment_geos.values())[0]))
  treatment_mask = np.isin(geos, treatment_geos).astype(int)
  return TreatmentMask(mask=jnp.array(treatment_mask), geos=treatment_geos)


def _get_analysis_summary(
    lift: api.Estimate,
    cumulative_lift_with_cis: np.ndarray,
    percent_lift: api.Estimate,
    icpd: Optional[api.Estimate],
    cumulative_icpd_with_cis: Optional[np.ndarray],
    analysis_dates: pd.Index,
    cell_names: list[str],
) -> api.AnalysisResult:
  """Converts raw analysis results into an AnalysisResult object."""
  cumulative_lift_estimates = pd.DataFrame(
      data=cumulative_lift_with_cis,
      index=analysis_dates,
      columns=['lift', 'lift_lower_bound', 'lift_upper_bound'],
  )

  cumulative_icpd_estimates = None
  if cumulative_icpd_with_cis is not None:
    cumulative_icpd_estimates = pd.DataFrame(
        data=cumulative_icpd_with_cis,
        index=analysis_dates,
        columns=['icpd', 'icpd_lower_bound', 'icpd_upper_bound'],
    )

  return api.AnalysisResult(
      results={
          cell_names[0]: api.AnalysisMetrics(
              lift=lift,
              percent_lift=percent_lift,
              cumulative_lift_estimates=cumulative_lift_estimates,
              icpd=icpd,
              cumulative_icpd_estimates=cumulative_icpd_estimates,
          )
      },
      counterfactual_conversions=pd.DataFrame(),
  )


def analyze(
    data: pd.DataFrame,
    analysis_config: api.AnalysisConfig,
    # An option to enable and configure automatic data quality checks.
    data_quality_check_config: data_quality.QualityCheckConfig = data_quality.QualityCheckConfig(),
) -> api.AnalysisResult:
  """Analyzes a GeoX experiment."""
  # TODO: Add data quality checks and move some of the below
  # checks to the data quality check function.
  del data_quality_check_config

  if analysis_config.methodology != api.Methodology.TBR:
    raise ValueError(
        f'Unsupported methodology: {analysis_config.methodology}. Only TBR is'
        ' supported.'
    )

  error_messages: list[str] = util.validate_schema(data)
  if error_messages:
    raise ValueError(f'Data validation failed: {error_messages}')

  # 1. Prepare configuration.
  design_config = _prepare_design_config(analysis_config)
  experiment_type = _get_experiment_type(design_config)

  # 2. Prepare data and masks.
  data = _prepare_data(data, analysis_config)
  treatment = _get_treatment_mask(data, analysis_config.design)

  # 3. Extract time series arrays.
  conversions = _get_time_series(data, api.CONVERSIONS, analysis_config)

  spend = None
  if api.SPEND in data.columns:
    spend = _get_time_series(data, api.SPEND, analysis_config)

  # 4. Generate placebo masks.
  # We use a key derived from the design config seed to ensure that placebo
  # mask generation is deterministic but distinct from the design phase.
  # We use a fold-in of a constant value to distinguish this key from others.
  # TODO: After refactoring to move placebo design generation to
  # the design phase, replace the constant fold-in with key splitting.
  analysis_key = jax.random.fold_in(jax.random.key(design_config.seed), 12345)
  placebo_masks = _get_placebo_masks(
      treatment_mask=treatment.mask,
      placebo_data=data[~data[api.LOCATION].isin(treatment.geos)],
      design_config=design_config,
      constraints=analysis_config.design.constraints,
      key=analysis_key,
      geo_stratum_labels=analysis_config.design.geo_stratum_labels,
  )

  # 5. Run methodology analysis.
  tbr_result = tbr.analyze(
      pretest_conversions=conversions.pretest,
      test_conversions=conversions.test,
      treatment_mask=treatment.mask,
      placebo_masks=placebo_masks,
      alpha=analysis_config.alpha,
      experiment_type=experiment_type,
      test_type=analysis_config.test_type,
      pretest_spend=spend.pretest if spend else None,
      test_spend=spend.test if spend else None,
  )

  # 6. Package and return results.
  return _get_analysis_summary(
      lift=tbr_result.lift,
      cumulative_lift_with_cis=tbr_result.cumulative_lift_with_cis,
      percent_lift=tbr_result.percent_lift,
      icpd=tbr_result.icpd,
      cumulative_icpd_with_cis=tbr_result.cumulative_icpd_with_cis,
      analysis_dates=conversions.dates,
      cell_names=list(analysis_config.design.treatment_geos.keys()),
  )


def plot_analysis(analysis_result: api.AnalysisResult):
  """Visualizes an analysis result."""
  raise NotImplementedError
