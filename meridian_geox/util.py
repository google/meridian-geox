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

"""Utility functions for GeoX modules."""

from meridian_geox import api
import pandas as pd
import pandera.pandas as pa


def is_go_dark_or_heavy_up(
    experiment_types: api.ExperimentType | list[api.ExperimentType],
) -> bool:
  """Checks if the experiment is go dark or heavy up."""
  # TODO: Add multicell support.
  if not isinstance(experiment_types, list):
    experiment_types = [experiment_types]
  return any(
      et in (api.ExperimentType.GO_DARK, api.ExperimentType.HEAVY_UP)
      for et in experiment_types
  )


def pivot_and_sort_data(
    data: pd.DataFrame,
    value_column: str,
) -> pd.DataFrame:
  """Pivots data and ensures consistent location ordering."""
  pivoted = data.pivot_table(
      index=api.DATE,
      columns=api.LOCATION,
      values=value_column,
      aggfunc='sum',
  ).fillna(0)
  # Ensure columns are sorted to match the order of geos list.
  return pivoted.reindex(sorted(pivoted.columns), axis=1).sort_index()


def validate_schema(data: pd.DataFrame) -> list[str]:
  """Validates the input data using the schema."""
  try:
    api.DataSchema.validate(data)
  except (pa.errors.SchemaError, pa.errors.SchemaErrors) as err:
    return [str(err)]
  return []


def validate_design_input(
    data: pd.DataFrame,
    design_config: api.DesignConfig,
    constraints: api.Constraints,
) -> list[str]:
  """Validates the input data using the schema."""
  # TODO: Include data schema check and other data checks. For
  # example, check the pretest data length, etc. Return a list of error
  # messages. Empty list means no errors.
  # TODO: Add checks for reporting use case.
  errors = validate_schema(data)
  if errors:
    return errors

  # Check data length.
  if data[api.DATE].nunique() < 3 * design_config.experiment_duration:
    errors.append(
        f'Data has {data[api.DATE].nunique()} dates, but experiment duration'
        f' is {design_config.experiment_duration}. Need at least 3 * experiment'
        ' duration dates.'
    )

  # Check number of geos remaining after excluding excluded geos.
  filtered_data = data[
      ~data[api.LOCATION].isin(list(constraints.excluded_geos))
  ]
  if filtered_data[api.LOCATION].nunique() < 4:
    errors.append(
        'Not enough geos left after excluding excluded geos. Need at least 4'
        ' geos.'
    )

  # Check that excluded geos do not overlap with included control geos.
  if constraints.excluded_geos & constraints.included_control_geos:
    errors.append('Excluded geos must not overlap with included control geos.')

  # Check that alpha and power are between 0 and 1.
  if design_config.alpha <= 0 or design_config.alpha >= 1:
    errors.append('Alpha must be between 0 and 1.')
  if design_config.power <= 0 or design_config.power >= 1:
    errors.append('Power must be between 0 and 1.')

  # Check that CPIC is set for HOLDBACK experiments.
  if isinstance(design_config.experiment_types, list):
    experiment_types = design_config.experiment_types
  else:
    experiment_types = [design_config.experiment_types]
  if (
      api.ExperimentType.HOLDBACK in experiment_types
      and not design_config.cost_per_incremental_conversion
  ):
    errors.append(
        'Cost per incremental conversion must be set for HOLDBACK experiments.'
    )

  return errors
