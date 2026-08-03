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

"""Data validation functions for GeoX modules."""

import logging
from meridian_geox import api
from meridian_geox import util
import pandas as pd
import pandera.pandas as pa


def validate_schema(data: pd.DataFrame) -> list[str]:
  """Validates the input data using the schema."""
  try:
    api.DataSchema.validate(data)
  except (pa.errors.SchemaError, pa.errors.SchemaErrors) as err:
    return [str(err)]
  return []


def validate_general_checks(data: pd.DataFrame) -> list[str]:
  """Ensures the general data validation checks pass."""
  errors = validate_schema(data)

  # Total Conversions
  if api.CONVERSIONS in data.columns:
    if data[api.CONVERSIONS].sum() <= 0:
      errors.append('Total conversions must be greater than 0.')

  # Data Granularity Pattern
  if api.DATE in data.columns:
    try:
      unique_dates = pd.Series(data[api.DATE].unique()).sort_values()
      if len(unique_dates) > 1:
        date_diffs = unique_dates.diff().dt.days.dropna()
        if (date_diffs == 7).all():
          errors.append('Weekly patterns are not supported.')
    except (ValueError, TypeError, AttributeError):
      pass

  return errors


def validate_design_input(
    data: pd.DataFrame,
    design_config: api.DesignConfig,
    constraints: api.Constraints,
) -> list[str]:
  """Validates the design input data and config using the schema and constraints."""
  errors = validate_general_checks(data)

  # Check data length.
  if api.DATE in data.columns:
    if data[api.DATE].nunique() < 3 * design_config.experiment_duration.days:
      errors.append(
          f'Data has {data[api.DATE].nunique()} dates, but experiment duration'
          f' is {design_config.experiment_duration.days}. Need at least 3 *'
          ' experiment duration dates during design phase.'
      )

  # Check number of geos remaining after excluding excluded geos.
  if api.LOCATION in data.columns:
    filtered_data = data[
        ~data[api.LOCATION].isin(list(constraints.excluded_geos))
    ]
    if filtered_data[api.LOCATION].nunique() < 2 * (
        design_config.cell_count + 1
    ):
      errors.append(
          'Not enough geos left after excluding excluded geos. Need at least 2'
          ' geos in each group.'
      )

  # Check that excluded geos do not overlap with included control geos.
  if constraints.excluded_geos & constraints.included_control_geos:
    errors.append('Excluded geos must not overlap with included control geos.')

  # Check that alpha, power, and min_r2 are between 0 and 1.
  if design_config.alpha <= 0 or design_config.alpha >= 1:
    errors.append('Alpha must be between 0 and 1.')
  if design_config.power <= 0 or design_config.power >= 1:
    errors.append('Power must be between 0 and 1.')
  if design_config.min_r2 <= 0 or design_config.min_r2 >= 1:
    errors.append('min_r2 must be between 0 and 1.')

  # Normalization in DesignConfig ensures experiment_types and
  # cost_per_incremental_conversion are dictionaries.
  experiment_types: dict[str, api.ExperimentType] = (
      design_config.experiment_types  # type: ignore
  )
  if len(experiment_types) != design_config.cell_count:
    errors.append(
        f'Experiment types must be set for {design_config.cell_count} cells,'
        f' but {len(experiment_types)} cells are set.'
    )
    return errors
  for cell_id in range(1, design_config.cell_count + 1):
    cell = f'cell_{cell_id}'
    if cell not in experiment_types:
      errors.append(
          f'Experiment type must be set for {cell} in multicell experiments.'
      )

  cpics: dict[str, float] = (
      design_config.cost_per_incremental_conversion  # type: ignore
  )
  for cell, et in experiment_types.items():
    if et == api.ExperimentType.HOLDBACK and cpics.get(cell, 0) <= 0:
      errors.append(
          'Cost per incremental conversion must be set and larger than 0 '
          f'for cell {cell} as it is a HOLDBACK experiment.'
      )
      break

  # After normalization, budget_constraint is guaranteed to be a dict.
  budget_constraints: dict[str, api.Budget] = (
      constraints.budget_constraint  # type: ignore
  )
  _validate_budget_constraints(experiment_types, budget_constraints)

  # Check that max_conversions_percent per cell is at least 0.1 and that
  # the total max_conversions_percent is at most 0.5.
  if constraints.max_conversions_percent is not None:
    if constraints.max_conversions_percent >= 0.5:
      errors.append('Max conversions percent must be less than 0.5.')
    if constraints.max_conversions_percent / design_config.cell_count < 0.1:
      logging.warning(
          'Max conversions percent is less than 0.1 times the number of cells.'
      )

  # Check that go dark and heavy up cells have the appropriate spend column.
  for cell, experiment_type in experiment_types.items():
    if experiment_type in (
        api.ExperimentType.GO_DARK,
        api.ExperimentType.HEAVY_UP,
    ):
      if cell == 'cell_1':
        if 'spend' not in data.columns and 'spend_cell_1' not in data.columns:
          errors.append(
              f'Spend column must be set for {experiment_type.name} experiments'
              f' in cell {cell}. Expected spend or spend_cell_1.'
          )
      else:
        spend_col = f'spend_{cell}'
        if spend_col not in data.columns:
          errors.append(
              f'Spend column must be set for {experiment_type.name} experiments'
              f' in cell {cell}. Expected {spend_col}.'
          )

  return errors


def _validate_budget_constraints(
    experiment_types: dict[str, api.ExperimentType],
    budget_constraints: dict[str, api.Budget],
) -> None:
  """Validates budget constraints and logs warnings for missing values."""
  for cell_name, experiment_type in experiment_types.items():
    budget_constraint = budget_constraints.get(cell_name)
    if util.is_go_dark_or_heavy_up(experiment_type):
      budget_percent = (
          budget_constraint.budget_pct if budget_constraint else None
      )
      if budget_percent is None:
        logging.warning(
            '%s is %s but budget_pct is not provided. Using 100%% as default.',
            cell_name,
            experiment_type.name,
        )
        if budget_constraint and budget_constraint.budget is not None:
          logging.warning(
              '%s is %s but budget (absolute) is provided instead of'
              ' budget_pct.',
              cell_name,
              experiment_type.name,
          )
    else:
      # HOLDBACK
      budget = budget_constraint.budget if budget_constraint else None
      if budget is None:
        logging.warning('%s is HOLDBACK but no budget is provided.', cell_name)
        if budget_constraint and budget_constraint.budget_pct is not None:
          logging.warning(
              '%s is HOLDBACK but budget_pct is provided.',
              cell_name,
          )


def validate_analysis_input(
    data: pd.DataFrame,
    analysis_config: api.AnalysisConfig,
) -> list[str]:
  """Validates the analysis input data and config."""
  errors = validate_general_checks(data)
  # Check geo set consistency with Design.
  if api.LOCATION in data.columns and analysis_config.design is not None:
    design = analysis_config.design
    design_geos = set(design.control_geos)
    for cell_design in design.designs.values():
      design_geos.update(cell_design.treatment_geos)
    analysis_geos = set(data[api.LOCATION].unique())
    if design.excluded_geos:
      analysis_geos = analysis_geos - set(design.excluded_geos)
    if analysis_geos != design_geos:
      errors.append(
          'The locations in the analysis data do not match the locations from '
          'design.'
      )

  # Check pretest data duration.
  if (
      api.DATE in data.columns
      and analysis_config.design is not None
      and analysis_config.design.design_config is not None
  ):
    unique_dates = pd.DatetimeIndex(data[api.DATE].unique())
    if analysis_config.excluded_dates:
      unique_dates = unique_dates[
          ~unique_dates.isin(analysis_config.excluded_dates)
      ]

    if analysis_config.pretest_end_date is not None:
      pretest_dates = unique_dates[
          unique_dates <= analysis_config.pretest_end_date
      ]
    else:
      pretest_dates = unique_dates[
          unique_dates < analysis_config.analysis_start_date
      ]

    pretest_duration = len(pretest_dates)
    experiment_duration_days = (
        analysis_config.design.design_config.experiment_duration.days
    )
    if pretest_duration < 2 * experiment_duration_days:
      errors.append(
          f'Pretest data has {pretest_duration} dates, but experiment duration'
          f' is {experiment_duration_days}. Need at least 2 * experiment'
          ' duration pretest dates during analysis phase.'
      )

  return errors
