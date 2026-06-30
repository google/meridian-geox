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

import re

from meridian_geox import api
import pandas as pd


def is_go_dark_or_heavy_up(
    experiment_types: api.ExperimentType | dict[str, api.ExperimentType],
) -> bool:
  """Checks if the experiment is go dark or heavy up."""
  # TODO: Add multicell support.
  if isinstance(experiment_types, dict):
    experiment_types_list = list(experiment_types.values())
  else:
    experiment_types_list = [experiment_types]
  return any(
      et in (api.ExperimentType.GO_DARK, api.ExperimentType.HEAVY_UP)
      for et in experiment_types_list
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


def cell_id_from_cell_name(cell_name: str) -> int:
  """Extracts the positive integer cell ID from a cell name.

  Args:
    cell_name: A cell name string of the form "cell_k" where k is a positive
      integer.

  Returns:
    The cell ID as an integer.

  Raises:
    ValueError: If cell_name is not in the form "cell_k" where k is a positive
      integer.
  """
  match = re.fullmatch(r'cell_(\d+)', cell_name)
  if not match or int(match.group(1)) <= 0:
    raise ValueError(
        f'Invalid cell name: {cell_name}. Must be of the form "cell_k" '
        'where k is a positive integer.'
    )
  return int(match.group(1))
