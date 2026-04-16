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

"""Library for generating design candidates."""

import dataclasses
import functools
import logging
from typing import Any, Optional

import jax
import jax.numpy as jnp
from jaxkd import extras
from meridian_geox import api
from meridian_geox import util
from meridian_geox.methodology import tbr
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.tsa import stattools
from tslearn import metrics


@dataclasses.dataclass
class ClusteringResult:
  """The result of clustering geos.

  Attributes:
    means: (k, d) The cluster centers.
    labels: (n,) The cluster labels.
  """

  means: jax.Array
  labels: jax.Array


def linear_fit(x):
  non_nan_mask = ~np.isnan(x)
  return np.polyfit(np.arange(len(x))[non_nan_mask], x[non_nan_mask], 1)[0]


def autocorrelation(x):
  non_nan_mask = ~np.isnan(x)
  autocorr = stattools.acf(x[non_nan_mask], nlags=1, fft=False)
  return autocorr[1] if len(autocorr) > 1 else 0.0


def cluster_geos(
    processed_data: Any,
    design_config: api.DesignConfig,
    key: jax.Array,
) -> ClusteringResult:
  """Clusters geos based on distance.

  Args:
    processed_data: The processed data for experiment design.
    design_config: The design config.
    key: The JAX random key for clustering.

  Returns:
    A ClusteringResult object with cluster centers and labels. The labels are
    ordered by increasing geo labels.
  """
  conversions = processed_data.selection_train.T

  mean = jnp.nanmean(conversions, axis=1)
  coeff_of_variation = jnp.nanstd(conversions, axis=1) / mean

  trend_slope = np.apply_along_axis(linear_fit, axis=1, arr=conversions)
  autocorr = np.apply_along_axis(autocorrelation, axis=1, arr=conversions)

  reference_ts = jnp.nanmean(conversions, axis=0)
  dtw_distance = np.apply_along_axis(
      metrics.dtw, axis=1, arr=np.array(conversions), s2=np.array(reference_ts)
  )

  clustering_features = jnp.nan_to_num(
      jnp.stack(
          [
              mean,
              coeff_of_variation,
              trend_slope,
              autocorr,
              dtw_distance,
          ],
          axis=1,
      )
  )

  means, labels = extras.k_means(
      key,
      jax.nn.standardize(clustering_features, axis=0),
      k=design_config.num_strata,
      steps=design_config.k_means_iterations,
  )
  return ClusteringResult(means=means, labels=labels)


@jax.jit
def _get_intervals(stratum_counts: jnp.ndarray):
  """Maps stratum counts to the unit interval.

    For example, if the input is [50, 25, 50, 75], this function will return
    [0.25, 0.375, 0.625, 1.0].

  Args:
    stratum_counts: An array of integers representing the number of geos per
      stratum.
  """
  norm_counts = stratum_counts / jnp.sum(stratum_counts)
  return jnp.cumsum(norm_counts)


@jax.jit
def _float_to_label(floats: jnp.ndarray, intervals: jnp.ndarray):
  """Maps floats within the unit interval to the appropriate stratum label."""
  return jnp.argmax(floats < intervals)


@functools.partial(jax.jit, static_argnames=['seq_length', 'pad_length'])
def get_minimal_discrepancy_stratum_labels(
    offset: int, stratum_counts: jnp.ndarray, seq_length: int, pad_length: int
):
  """Generates quasirandom sequences of stratum labels with minimal discrepancy.

  A low discrepancy sequence of stratum labels is one where the frequency of a
  stratum label is close to the actual proportion of geos with that label. This
  allows us to perform stratified sampling in a parallelized manner. We use
  Sobol sequences to generate quasirandom sequences of floats between 0 and 1,
  and then map these floats to the appropriate stratum label. We use a random
  offset value to increase the diversity of the generated sequences.

  Args:
    offset: An integer offset used to increase the diversity of the generated
      sequences.
    stratum_counts: An array of integers representing the number of geos per
      stratum. stratum_count[i] is the number of geos in stratum i.
    seq_length: An integer representing the desired length of the sequence of
      stratum labels.
    pad_length: An integer that determines how many extra entries in the Sobol
      sequence to generate. In other words, seq_length + pad_length is the total
      length of the Sobol sequence used. This should be larger than the number
      of study candidates generated for best results.
  """
  sampler = stats.qmc.Sobol(d=1)
  quasi_random_seq = jnp.roll(
      jnp.ravel(sampler.random(seq_length + pad_length)), shift=-offset
  )[:seq_length]
  intervals = _get_intervals(stratum_counts)
  return jax.vmap(_float_to_label, in_axes=(0, None))(
      quasi_random_seq, intervals
  )


@jax.jit
def get_stratified_geo_sequence(
    stratum_seq: jnp.ndarray, geos: jnp.ndarray, geo_stratum_labels: jnp.ndarray
):
  """Generates a sequence of geo indices constrained to the stratum ordering.

  A typical random permutation of geo indices does not guarantee that any
  contiguous subsequence has low discrepancy with respect to the stratum counts.
  However, for the sampling to be stratified instead of just simply random, we
  require our random sequences to be of low discrepancy.

  Args:
    stratum_seq: A sequence of stratum labels of shape (n_geos,).
    geos: A sequence of geo indices of shape (n_geos,).
    geo_stratum_labels: An array representing the stratum label of each geo.
      geo_stratum_labels[i] is the stratum label of geo i.
  """
  # get indices for sorting stratum_seq first by the cluster labels, then by
  # their order of appearance.
  # Ex. stratum_seq = [2, 2, 1, 0, 3, 2] -> [3, 2, 0, 1, 5, 4].
  clusters_lex_sort_indices = jnp.lexsort(
      [jnp.arange(len(stratum_seq)), stratum_seq]
  )
  # get indices for inverting the sort.
  # Ex. stratum_seq_sort_indices = [3, 2, 0, 1, 5, 4] -> [2, 3, 1, 0, 5, 4]
  inverse_clusters_lex_sort_indices = jnp.argsort(clusters_lex_sort_indices)

  # get the geo cluster labels in order of the geo permutation.
  permuted_geo_stratum_labels = jnp.take(geo_stratum_labels, geos)
  # get indices for sorting the geo cluster labels.
  geo_strata_lex_sort_indices = jnp.lexsort([
      jnp.arange(len(permuted_geo_stratum_labels)),
      permuted_geo_stratum_labels,
  ])

  # get the list of geos sorted by cluster labels and order of appearance.
  geos_by_cluster_ordering = jnp.take(geos, geo_strata_lex_sort_indices)

  # returns geos constrained to the stratum_seq ordering.
  return jnp.take(geos_by_cluster_ordering, inverse_clusters_lex_sort_indices)


def compute_mask_maximizing_conversions(
    geos: np.ndarray,
    geo_strata: np.ndarray,
    geo_conversions: np.ndarray,
    max_conversions: float,
):
  """Generates a boolean mask of treatment geos maximizing conversions.

  Goes from left to right, adding geos to treatment. It follows two rules: it
  adds geos from strata in the order specified by geo_strata, and it maximizes
  the conversions while staying under max_conversions. The sequential nature of
  this algorithm
  requires us to use standard numpy instead of JAX.

  Args:
    geos: A sequence of geos of shape (n_geos,).
    geo_strata: A sequence of geo strata of shape (n_geos,). geo_strata[i] is
      the stratum label of geos[i].
    geo_conversions: An array representing the conversions of each geo.
      geo_conversions[i] is the conversions of geo i.
    max_conversions: A float representing the max treatment conversions.

  Returns:
    An integer mask of treatment geos of shape (n_geos,). The value at index i
    is 1 if geo i is a treatment geo, and 0 otherwise.
  """
  geo_sorted_conversions = np.take(geo_conversions, geos)
  treatment_geos = []
  geo_cluster_index = 0
  geo_conversion_index = 0
  conversions = 0.0
  while geo_cluster_index < len(geo_strata) and geo_conversion_index < len(
      geo_sorted_conversions
  ):
    required_cluster = geo_strata[geo_cluster_index]
    added_conversions = geo_sorted_conversions[geo_conversion_index]
    if (
        geo_strata[geo_conversion_index] == required_cluster
        and conversions + added_conversions <= max_conversions
    ):
      treatment_geos.append(geos[geo_conversion_index])
      conversions += added_conversions
      geo_cluster_index += 1
      geo_conversion_index += 1
    else:
      geo_conversion_index += 1

  mask = np.full(len(geos), 0)
  mask[treatment_geos] = 1
  return mask.astype(jnp.int32)


def get_unconstrained_stratified_sampling_candidates(
    design_config: api.DesignConfig,
    stratum_counts: jnp.ndarray,
    geo_stratum_labels: jnp.ndarray,
    geo_conversions: jnp.ndarray,
    max_conversions: float,
    key: jax.Array,
):
  """Generates stratified sampling study candidates.

  The steps to generate one stratified sampling candidate:
  1. Balance strata via Sobol sequence: The algorithm uses a Sobol sequence to
     generate a low-discrepancy sequence of stratum labels. Unlike pure
     randomness, this ensures that any [:k]-subsequence of the sequence has
     proportions consistent with the overall population, providing a balanced
     foundation for geo selection.
  2. Randomize geos: Available geos are independently shuffled into a random
     permutation.
  3. Align geos to the stratum sequence: The shuffled geos are reordered to
     match the stratum labels obtained from the Sobol sequence. If the sequence
     calls for a specific stratum, the next
     available geo from that stratum is selected. This results in a geo list
     that is randomized yet perfectly balanced across strata.
  4. Assign treatment geos: The algorithm greedily adds geos from the reordered
     list to the treatment group. A geo is assigned only if it matches the
     required stratum and its addition doesn't exceed the conversion limit.

  Args:
    design_config: The design config.
    stratum_counts: An array representing the number of geos in each stratum.
    geo_stratum_labels: An array representing the stratum label of each geo.
      geo_stratum_labels[i] is the stratum to which geo i belongs.
    geo_conversions: An array representing the conversions of each geo.
      geo_conversions[i] is the conversions of geo i.
    max_conversions: A float representing the max treatment conversions.
    key: The JAX PRNG key.

  Returns:
    A boolean mask of treatment geos of shape (n_candidates, n_geos). For a
    fixed row (candidate), the value at index i is True if and only if geo i is
    a treatment geo.
  """
  offset_key, permutation_key = jax.random.split(key)
  seq_length = np.sum(np.array(stratum_counts))
  # This is a random offset used to increase the diversity of the generated
  # sequences of stratum labels.
  offsets = jax.random.randint(
      offset_key,
      (design_config.n_candidates,),
      minval=0,
      maxval=design_config.pad_length - 1,
  )
  # Generate sequences of stratum labels, one per candidate, with minimal
  # discrepancy. This means that the frequency of each stratum label in the
  # sequence is close to the proportion of geos in that stratum, even if we
  # only take the first k (k < seq_length) elements of the sequence.
  stratum_seqs = jax.vmap(
      get_minimal_discrepancy_stratum_labels, in_axes=(0, None, None, None)
  )(offsets, stratum_counts, seq_length, design_config.pad_length)
  # Generate random permutations of geo indices, one per candidate.
  geo_permutations = jnp.full(
      (design_config.n_candidates, seq_length), jnp.arange(seq_length)
  )
  geo_permutations = jax.random.permutation(
      permutation_key,
      geo_permutations,
      axis=1,
      independent=True,
  )

  # This takes the random permutations and constrains them to the stratum
  # ordering. This means that for any left-to-right sequence of geos, the
  # stratum labels will be in the order specified by stratum_seqs.
  stratified_random_geos = jax.vmap(
      get_stratified_geo_sequence, in_axes=(0, 0, None)
  )(stratum_seqs, geo_permutations, geo_stratum_labels)
  stratified_random_geos_strata = jnp.take(
      geo_stratum_labels, stratified_random_geos
  )

  get_geo_masks = np.vectorize(
      compute_mask_maximizing_conversions,
      excluded={'geo_conversions', 'max_conversions'},
      signature='(n),(n)->(n)',
  )
  return get_geo_masks(
      geos=np.array(stratified_random_geos),
      geo_strata=np.array(stratified_random_geos_strata),
      geo_conversions=np.array(geo_conversions),
      max_conversions=max_conversions,
  )


def _get_expanded_mask(reduced_mask, expanded_mask_size, filtered_geo_indices):
  expanded_mask = np.zeros(expanded_mask_size).astype(jnp.int32)
  expanded_mask[filtered_geo_indices] = reduced_mask
  return expanded_mask


def _should_apply_slope_filter(
    design_config: api.DesignConfig,
    selection_train_spend: Optional[jnp.ndarray],
) -> bool:
  """Checks if slope filter should be applied."""
  return (
      util.is_go_dark_or_heavy_up(design_config.experiment_types)
      and selection_train_spend is not None
  )


def get_stratified_sampling_candidates(
    selection_train: jnp.ndarray,
    filtered_data: pd.DataFrame,
    design_config: api.DesignConfig,
    constraints: api.Constraints,
    geo_stratum_labels: jnp.ndarray,
    key: jax.Array,
    selection_train_spend: Optional[jnp.ndarray] = None,
):
  """Generates stratified sampling study candidates subject to constraints.

  Args:
    selection_train: The pre-period conversions data.
    filtered_data: The filtered data used for design.
    design_config: The design config.
    constraints: The design constraints.
    geo_stratum_labels: An array representing the stratum label of each geo.
      geo_stratum_labels[i] is the stratum to which geo i belongs.
    key: The JAX PRNG key.
    selection_train_spend: The pre-period spend data.

  Returns:
    A boolean mask of treatment geos of shape (n_candidates, n_geos). For a
    fixed row (candidate), the value at index i is True if and only if geo i is
    a treatment geo. The geos are in increasing order of name.
  """
  conversions = selection_train
  geo_conversions = jnp.sum(conversions, axis=0)
  total_conversions = np.sum(geo_conversions)

  # Remove geos that are forced to be in control.
  geos = sorted(filtered_data['location'].unique())
  filtered_geo_indices = np.argwhere(
      ~np.isin(geos, list(constraints.included_control_geos))
  ).reshape(-1)
  strata, counts = np.unique(
      geo_stratum_labels[filtered_geo_indices], return_counts=True
  )
  sorted_strata_indices = np.argsort(strata)
  filtered_stratum_counts = np.take(counts, sorted_strata_indices)

  apply_slope_filter = _should_apply_slope_filter(
      design_config, selection_train_spend
  )
  n_designs = design_config.n_candidates
  max_retries = design_config.max_candidate_generation_retries
  valid_candidates_list = []
  current_count = 0

  # Split keys for the loop.
  loop_keys = jax.random.split(key, max_retries)

  for i in range(max_retries):
    if current_count >= n_designs:
      break

    # Generate a batch of candidates.
    filtered_treatment_geo_candidates = (
        get_unconstrained_stratified_sampling_candidates(
            design_config,
            filtered_stratum_counts,
            geo_stratum_labels[filtered_geo_indices],
            geo_conversions[filtered_geo_indices],
            float(constraints.max_conversions_percent * total_conversions),
            loop_keys[i],
        )
    )

    candidates_batch = np.apply_along_axis(
        _get_expanded_mask,
        axis=1,
        arr=filtered_treatment_geo_candidates,
        expanded_mask_size=len(geo_conversions),
        filtered_geo_indices=filtered_geo_indices,
    )
    candidates_batch = jnp.array(candidates_batch)

    # Ensure each candidate has at least 2 treated and 2 control geos.
    n_treated_batch = jnp.sum(candidates_batch, axis=1)
    n_control_batch = len(geo_conversions) - n_treated_batch
    valid_indices = (n_treated_batch >= 2) & (n_control_batch >= 2)

    if apply_slope_filter:
      slope_mask = tbr.check_slope_similarity(
          candidates_batch,
          selection_train,
          selection_train_spend,
          design_config.slope_tolerance,
      )
      valid_indices = jnp.logical_and(valid_indices, slope_mask)

    valid_batch = candidates_batch[valid_indices]

    if len(valid_batch) > 0:
      valid_candidates_list.append(valid_batch)
      current_count += len(valid_batch)

  if not valid_candidates_list:
    raise ValueError(
        'Could not find enough valid candidates satisfying design constraints'
        ' and/or slope similarity criteria. Consider relaxing constraints.'
    )

  # Concatenate all valid batches.
  candidates = jnp.concatenate(valid_candidates_list, axis=0)

  # Trim to the requested number of candidates.
  if len(candidates) > n_designs:
    candidates = candidates[:n_designs]
  elif len(candidates) < n_designs:
    logging.warning(
        'Found %d valid candidates after %d retries,'
        ' but requested %d. This is likely due to strict constraints and/or'
        ' slope similarity criteria. Proceeding with fewer candidates.',
        len(candidates),
        max_retries,
        n_designs,
    )

  return candidates


@functools.partial(jax.jit, static_argnames=['n_designs', 'n_geos', 'n_treated'])
def _generate_random_masks(
    n_designs: int,
    n_geos: int,
    n_treated: int,
    forced_control_mask: jnp.ndarray,
    key: jax.Array,
) -> jnp.ndarray:
  """Generates random treatment masks with fixed group sizes."""

  def _generate_single_mask(single_key):
    # Generate random scores for shuffling. "Scores" here refers to random
    # values assigned to each geo, which are then used to determine the rank of
    # each geo for random assignment. By sorting these scores, we can randomly
    # select the top N geos for the treatment group.
    scores = jax.random.uniform(single_key, shape=(n_geos,))

    # Force control geos to have high scores (2.0) so they are ranked last.
    # Uniform scores are in [0, 1).
    scores = jnp.where(forced_control_mask, 2.0, scores)

    # argsort(scores) gives indices that sort the scores.
    # argsort(argsort(scores)) gives the rank of each original score (0 to N-1).
    # This allows us to select the top `n_treated` random scores efficiently.
    ranks = jnp.argsort(jnp.argsort(scores))

    # Ranks 0 to n_treated-1 -> Treatment (1).
    # Ranks n_treated to n_geos-1 -> Control (0).
    mask = jnp.where(ranks < n_treated, 1, 0)
    return mask.astype(jnp.int32)

  keys = jax.random.split(key, n_designs)
  return jax.vmap(_generate_single_mask)(keys)


def get_random_candidates(
    filtered_data: pd.DataFrame,
    design_config: api.DesignConfig,
    constraints: api.Constraints,
    key: jax.Array,
    selection_train: jnp.ndarray,
    selection_train_spend: Optional[jnp.ndarray] = None,
) -> jnp.ndarray:
  """Generates random candidates for experiment design."""
  # Generate a batch of valid treatment masks (N_designs, Geos) satisfying
  # the design constraints and convert to jax array. 0 for control, 1 for
  # treatment1, 2 for treatment2, etc.
  # TODO: Add multicell support.

  # Determine the number of geos.
  geos = sorted(filtered_data['location'].unique())
  n_geos = len(geos)

  # Number of candidates to generate.
  n_designs = design_config.n_candidates
  max_conversions_percent = constraints.max_conversions_percent

  # Identify forced control geos.
  forced_control_mask = np.zeros(n_geos, dtype=bool)
  if constraints.included_control_geos:
    geo_to_idx = {g: i for i, g in enumerate(geos)}
    for g in constraints.included_control_geos:
      if g in geo_to_idx:
        forced_control_mask[geo_to_idx[g]] = True

  # Use the max_conversions_percent as a proxy to determine the number of
  # treated units.
  n_treated = int(n_geos * max_conversions_percent)

  # Check availability after forcing controls.
  n_available = n_geos - np.sum(forced_control_mask)
  if n_treated > n_available:
    logging.warning(
        'Requested %d treated units but only %d are '
        'available after excluding forced control geos. Reducing n_treated to '
        '%d.',
        n_treated,
        n_available,
        n_available,
    )
    n_treated = int(n_available)

  # Ensure at least 2 treatment geos.
  n_treated = max(2, n_treated)

  # Ensure at least 2 control units.
  if n_geos - n_treated < 2:
    raise ValueError('Not enough geos.')

  # Aggregate conversions per geo.
  geo_conversions = filtered_data.groupby('location')['conversions'].sum()
  # Reindex to ensure order matches 'geos'.
  geo_weights = jnp.array(geo_conversions.reindex(geos).fillna(0).values)
  total_volume = jnp.sum(geo_weights)

  apply_slope_filter = _should_apply_slope_filter(
      design_config, selection_train_spend
  )
  # Generate candidates in batches until we have enough.
  valid_candidates_list = []
  current_count = 0
  # Limit the number of retries to avoid infinite loops if constraints are too
  # tight.
  max_retries = design_config.max_candidate_generation_retries

  forced_control_mask_jax = jnp.array(forced_control_mask)

  # Split keys for the loop.
  loop_keys = jax.random.split(key, max_retries)

  for i in range(max_retries):
    if current_count >= n_designs:
      break

    # Generate a batch of candidates.
    candidates_batch = _generate_random_masks(
        n_designs,
        n_geos,
        n_treated,
        forced_control_mask_jax,
        loop_keys[i],
    )

    # Filter by volume constraint.
    treated_volumes = jnp.dot(candidates_batch, geo_weights)
    treated_pcts = treated_volumes / total_volume
    valid_indices = treated_pcts <= max_conversions_percent

    if apply_slope_filter:
      slope_mask = tbr.check_slope_similarity(
          candidates_batch,
          selection_train,
          selection_train_spend,
          design_config.slope_tolerance,
      )
      valid_indices = jnp.logical_and(valid_indices, slope_mask)

    valid_batch = candidates_batch[valid_indices]

    if len(valid_batch) > 0:
      valid_candidates_list.append(valid_batch)
      current_count += len(valid_batch)

  if not valid_candidates_list:
    raise ValueError(
        'Could not find enough valid candidates satisfying design constraints'
        ' and/or slope similarity criteria. Consider relaxing constraints.'
    )

  # Concatenate all valid batches.
  candidates = jnp.concatenate(valid_candidates_list, axis=0)

  # Trim to the requested number of candidates.
  if len(candidates) > n_designs:
    candidates = candidates[:n_designs]
  elif len(candidates) < n_designs:
    logging.warning(
        'Found %d valid candidates after %d retries,'
        ' but requested %d. This is likely due to strict constraints and/or'
        ' slope similarity criteria. Proceeding with fewer candidates.',
        len(candidates),
        max_retries,
        n_designs,
    )

  return candidates
