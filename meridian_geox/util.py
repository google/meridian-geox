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
from typing import Optional

from absl import logging
import jax.numpy as jnp
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
      aggfunc="sum",
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
  match = re.fullmatch(r"cell_(\d+)", cell_name)
  if not match or int(match.group(1)) <= 0:
    raise ValueError(
        f'Invalid cell name: {cell_name}. Must be of the form "cell_k" '
        'where k is a positive integer.'
    )
  return int(match.group(1))


def filter_by_r2(
    r2_scores: jnp.ndarray,
    min_r2: float,
    min_count_error: int,
    min_count_warning: int,
    error_message: str,
    warning_message: str,
    cell_names: Optional[list[str]] = None,
) -> jnp.ndarray:
  """Filters candidates based on R2 scores and checks counts.

  Args:
    r2_scores: R2 scores, can be 1D (n_candidates,) or 2D (n_candidates,
      n_cells). If 2D, a candidate passes only if all its cells pass the check.
    min_r2: The minimum R2 score threshold.
    min_count_error: Minimum number of passing candidates required.
    min_count_warning: Minimum number of passing candidates recommended.
    error_message: Error message template. Can use {count}, {min_r2}, etc. If
      cell_names is provided, can also use {cell_details}.
    warning_message: Warning message template.
    cell_names: Optional list of cell names, matching the second dimension of
      r2_scores.

  Returns:
    A boolean mask indicating which candidates passed.

  Raises:
    ValueError: If r2_scores has more than 2 dimensions, or if the
      number of passing candidates is less than min_count_error.
  """
  if r2_scores.ndim == 1:
    valid_mask = r2_scores >= min_r2
    valid_mask_per_cell = valid_mask[:, jnp.newaxis]
    cell_names_internal = ["default"]
  elif r2_scores.ndim == 2:
    valid_mask_per_cell = r2_scores >= min_r2
    valid_mask = jnp.all(valid_mask_per_cell, axis=1)
    cell_names_internal = (
        cell_names
        if cell_names
        else [f"cell_{i}" for i in range(r2_scores.shape[1])]
    )
  else:
    raise ValueError("r2_scores must be 1D or 2D")

  count = int(jnp.sum(valid_mask))

  # Prepare cell details if needed
  cell_details = ""
  if r2_scores.ndim == 2:
    counts_per_cell = jnp.sum(valid_mask_per_cell, axis=0)
    details = []
    for name, c in zip(cell_names_internal, counts_per_cell):
      details.append(f"{name}: {c} passed")
    cell_details = " (" + ", ".join(details) + ")"

  if count < min_count_error:
    raise ValueError(
        error_message.format(
            count=count,
            min_r2=min_r2,
            min_count_error=min_count_error,
            min_count_warning=min_count_warning,
            cell_details=cell_details,
        )
    )
  elif count < min_count_warning:
    logging.warning(
        warning_message.format(
            count=count,
            min_r2=min_r2,
            min_count_error=min_count_error,
            min_count_warning=min_count_warning,
            cell_details=cell_details,
        )
    )

  return valid_mask
