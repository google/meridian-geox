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

import datetime
import itertools
from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
from meridian_geox import api
from meridian_geox import generate_candidates
import numpy as np
import pandas as pd
from scipy import stats


def _reference_stratified_geo_sequence(stratum_seq, geos, geo_stratum_labels):
  """Reference lexsort implementation of get_stratified_geo_sequence."""
  sort_indices = np.lexsort([np.arange(len(stratum_seq)), stratum_seq])
  geo_strata = np.take(geo_stratum_labels, geos)
  geos_by_cluster = np.take(
      geos, np.lexsort([np.arange(len(geo_strata)), geo_strata])
  )
  return np.take(geos_by_cluster, np.argsort(sort_indices))


def _reference_mask_maximizing_conversions(
    geos, geo_strata, geo_conversions, max_conversions_per_cell, num_cells
):
  """Reference Python-loop implementation of the greedy multicell mask."""
  mask = np.zeros(len(geos), dtype=np.int32)
  n_geos = len(geos)
  for i in range(1, num_cells + 1):
    lb, ub = (i - 1) * n_geos // num_cells, i * n_geos // num_cells
    cell_geos, cell_strata = geos[lb:ub], geo_strata[lb:ub]
    geo_cluster_index, conversions = 0, 0.0
    for j, geo in enumerate(cell_geos):
      added_conversions = geo_conversions[geo]
      if (
          cell_strata[j] == cell_strata[geo_cluster_index]
          and conversions + added_conversions <= max_conversions_per_cell
      ):
        mask[geo] = i
        conversions += added_conversions
        geo_cluster_index += 1
  return mask


def _equal_conversions_data(n_geos):
  """Returns one day of data in which every geo has 10 conversions."""
  return pd.DataFrame([
      {
          'date': pd.Timestamp('2024-01-01'),
          'location': f'G{i:02d}',
          'conversions': 10.0,
      }
      for i in range(n_geos)
  ])


class GenerateCandidatesTest(parameterized.TestCase):

  def test_get_minimal_discrepancy_stratum_labels(self):
    stratum_counts = jnp.array([10, 20, 70])
    seq_length = 100
    pad_length = 20
    offset = 5
    sampler = stats.qmc.Sobol(d=1, scramble=True, rng=42)
    sobol_seq = jnp.ravel(sampler.random(seq_length + pad_length))
    labels = generate_candidates.get_minimal_discrepancy_stratum_labels(
        offset, stratum_counts, sobol_seq, seq_length
    )
    self.assertEqual(labels.shape, (seq_length,))
    unique_labels, counts = np.unique(labels, return_counts=True)
    self.assertTrue(np.all(unique_labels >= 0))
    self.assertTrue(np.all(unique_labels <= 2))
    # Check if counts are roughly proportional to stratum_counts
    proportions = counts / seq_length
    expected_proportions = stratum_counts / jnp.sum(stratum_counts)
    np.testing.assert_allclose(proportions, expected_proportions, atol=0.15)

    # check different offset gives different result
    offset2 = 6
    labels2 = generate_candidates.get_minimal_discrepancy_stratum_labels(
        offset2, stratum_counts, sobol_seq, seq_length
    )
    self.assertFalse(jnp.array_equal(labels, labels2))

  def test_get_stratified_geo_sequence(self):
    geo_stratum_labels = jnp.array([0, 0, 1, 1, 2, 2])
    stratum_seq = jnp.array([0, 1, 2, 0, 1, 2])
    geos = jnp.array([2, 5, 0, 1, 4, 3])
    stratified_geos = generate_candidates.get_stratified_geo_sequence(
        stratum_seq, geos, geo_stratum_labels
    )
    np.testing.assert_array_equal(
        geo_stratum_labels[stratified_geos], stratum_seq
    )
    # result must be a permutation of geos
    self.assertCountEqual(np.asarray(stratified_geos), np.asarray(geos))

  def test_get_intervals(self):
    stratum_counts = jnp.array([50, 25, 50, 75])
    intervals = generate_candidates._get_intervals(stratum_counts)
    np.testing.assert_allclose(intervals, jnp.array([0.25, 0.375, 0.625, 1.0]))

  def test_float_to_label(self):
    intervals = jnp.array([0.25, 0.375, 0.625, 1.0])
    floats = jnp.array([0.1, 0.3, 0.5, 0.7, 0.0, 0.249, 0.25, 0.374, 0.375])
    labels = jax.vmap(generate_candidates._float_to_label, in_axes=(0, None))(
        floats, intervals
    )
    np.testing.assert_array_equal(
        labels, jnp.array([0, 1, 2, 3, 0, 0, 1, 1, 2])
    )

  def test_compute_mask_maximizing_conversions(self):
    geos = np.array([0, 2, 4, 1, 3, 5])
    geo_strata = np.array([0, 1, 2, 0, 1, 2])
    geo_conversions = np.array([10, 1, 8, 2, 7, 3])
    max_conversions = 20.0
    mask = generate_candidates.compute_mask_maximizing_conversions(
        geos, geo_strata, geo_conversions, max_conversions
    )
    expected_mask = np.array([1, 0, 1, 0, 0, 0])
    np.testing.assert_array_equal(mask, expected_mask)

    max_conversions_high = 100.0
    mask_high = generate_candidates.compute_mask_maximizing_conversions(
        geos, geo_strata, geo_conversions, max_conversions_high
    )
    expected_mask_high = np.array([1, 1, 1, 1, 1, 1])
    np.testing.assert_array_equal(mask_high, expected_mask_high)

    # test max_conversions that forces skipping some geos
    max_conversions_low = 15.0
    mask_low = generate_candidates.compute_mask_maximizing_conversions(
        geos, geo_strata, geo_conversions, max_conversions_low
    )
    expected_mask_low = np.array([1, 0, 0, 1, 0, 1])
    np.testing.assert_array_equal(mask_low, expected_mask_low)

    # A geo that brings conversions exactly to the cap is still selected.
    mask_tie = generate_candidates.compute_mask_maximizing_conversions(
        np.array([3, 0, 2, 1]),
        np.array([0, 1, 0, 1]),
        np.array([10.0, 20.0, 30.0, 40.0]),
        50.0,
    )
    np.testing.assert_array_equal(mask_tie, np.array([1, 0, 0, 1]))

  def test_compute_mask_maximizing_conversions_multicell(self):
    geos = np.array([0, 2, 4, 1, 3, 5])
    geo_strata = np.array([0, 1, 2, 0, 1, 2])
    geo_conversions = np.array([10, 1, 8, 2, 7, 3])
    max_conversions = 20.0
    mask = generate_candidates.compute_mask_maximizing_conversions(
        geos, geo_strata, geo_conversions, max_conversions, num_cells=2
    )
    expected_mask = np.array([1, 2, 1, 2, 0, 2])
    np.testing.assert_array_equal(mask, expected_mask)

    max_conversions_high = 100.0
    mask_high = generate_candidates.compute_mask_maximizing_conversions(
        geos, geo_strata, geo_conversions, max_conversions_high, num_cells=2
    )
    expected_mask_high = np.array([1, 2, 1, 2, 1, 2])
    np.testing.assert_array_equal(mask_high, expected_mask_high)

  def test_get_unconstrained_stratified_sampling_candidates(self):
    design_config = api.DesignConfig(
        experiment_duration=datetime.timedelta(days=1),
        experiment_types=api.ExperimentType.HOLDBACK,
        n_candidates=10,
        seed=42,
    )
    stratum_counts = jnp.array([2, 2, 2])
    geo_stratum_labels = jnp.array([0, 0, 1, 1, 2, 2])
    geo_conversions = jnp.array([10, 2, 1, 7, 8, 3])
    max_conversions = 15.0
    key = jax.random.PRNGKey(0)
    sampler = stats.qmc.Sobol(d=1, scramble=True, rng=42)
    pad_length = max(10000, design_config.n_candidates)
    sobol_seq = jnp.ravel(sampler.random(6 + pad_length))
    candidates = (
        generate_candidates.get_unconstrained_stratified_sampling_candidates(
            design_config,
            stratum_counts,
            geo_stratum_labels,
            geo_conversions,
            max_conversions,
            key,
            sobol_seq,
            pad_length=pad_length,
        )
    )
    self.assertEqual(candidates.shape, (10, 6))
    for i in range(design_config.n_candidates):
      self.assertLessEqual(
          np.sum(np.array(geo_conversions)[candidates[i, :] == 1]),
          max_conversions,
      )

  def test_get_stratified_sampling_candidates(self):
    design_config = api.DesignConfig(
        experiment_duration=datetime.timedelta(days=1),
        experiment_types=api.ExperimentType.HOLDBACK,
        n_candidates=10,
        seed=42,
    )
    constraints = api.Constraints(
        max_conversions_percent=0.4,
    )
    geo_stratum_labels = jnp.array([0, 0, 1, 1, 2, 2])
    conversions_data = pd.DataFrame([
        {
            'date': pd.Timestamp('2024-01-01'),
            'location': 'G0',
            'conversions': 10.0,
        },
        {
            'date': pd.Timestamp('2024-01-01'),
            'location': 'G1',
            'conversions': 20.0,
        },
        {
            'date': pd.Timestamp('2024-01-01'),
            'location': 'G2',
            'conversions': 5.0,
        },
        {
            'date': pd.Timestamp('2024-01-01'),
            'location': 'G3',
            'conversions': 15.0,
        },
        {
            'date': pd.Timestamp('2024-01-01'),
            'location': 'G4',
            'conversions': 50.0,
        },
        {
            'date': pd.Timestamp('2024-01-01'),
            'location': 'G5',
            'conversions': 100.0,
        },
    ])
    # total conversions = 200. Max conversions = 0.4 * 200 = 80.
    key = jax.random.PRNGKey(0)
    selection_train = jnp.array([10.0, 20.0, 5.0, 15.0, 50.0, 100.0]).reshape(
        1, -1
    )
    candidates = generate_candidates.get_stratified_sampling_candidates(
        selection_train=selection_train,
        filtered_data=conversions_data,
        design_config=design_config,
        constraints=constraints,
        geo_stratum_labels=geo_stratum_labels,
        key=key,
    )
    self.assertEqual(candidates.shape[1], 6)
    self.assertLessEqual(len(candidates), design_config.n_candidates)
    self.assertLen(np.unique(candidates, axis=0), len(candidates))
    # total conversions = 200. Max conversions = 0.4 * 200 = 80.
    geo_conversions = np.array([10.0, 20.0, 5.0, 15.0, 50.0, 100.0])
    for candidate in np.asarray(candidates):
      self.assertLessEqual(np.sum(geo_conversions[candidate == 1]), 80.0)

  def test_get_stratified_sampling_candidates_min_geos_filter(self):
    design_config = api.DesignConfig(
        experiment_duration=datetime.timedelta(days=1),
        experiment_types=api.ExperimentType.HOLDBACK,
        n_candidates=10,
        seed=42,
    )
    # With 6 geos and max_conversions_percent=0.2, only 1 geo can be treated
    # (1/6 = 0.166... < 0.2 while 2/6 = 0.333... > 0.2).
    # Since we require at least 2 treated geos, all candidates should be
    # filtered.
    constraints = api.Constraints(
        max_conversions_percent=0.2,
    )
    geo_stratum_labels = jnp.array([0, 0, 1, 1, 2, 2])
    conversions_data = pd.DataFrame([
        {
            'date': pd.Timestamp('2024-01-01'),
            'location': 'G0',
            'conversions': 10.0,
        },
        {
            'date': pd.Timestamp('2024-01-01'),
            'location': 'G1',
            'conversions': 10.0,
        },
        {
            'date': pd.Timestamp('2024-01-01'),
            'location': 'G2',
            'conversions': 10.0,
        },
        {
            'date': pd.Timestamp('2024-01-01'),
            'location': 'G3',
            'conversions': 10.0,
        },
        {
            'date': pd.Timestamp('2024-01-01'),
            'location': 'G4',
            'conversions': 10.0,
        },
        {
            'date': pd.Timestamp('2024-01-01'),
            'location': 'G5',
            'conversions': 10.0,
        },
    ])

    key = jax.random.PRNGKey(0)
    selection_train = jnp.array([10.0] * 6).reshape(1, -1)

    with self.assertRaisesRegex(
        ValueError, 'Could not find enough valid candidates'
    ):
      generate_candidates.get_stratified_sampling_candidates(
          selection_train=selection_train,
          filtered_data=conversions_data,
          design_config=design_config,
          constraints=constraints,
          geo_stratum_labels=geo_stratum_labels,
          key=key,
      )

  def test_get_random_candidates_multicell(self):
    design_config = api.DesignConfig(
        experiment_duration=datetime.timedelta(days=1),
        cell_count=2,
        experiment_types={
            'cell_1': api.ExperimentType.HOLDBACK,
            'cell_2': api.ExperimentType.HOLDBACK,
        },
        n_candidates=10,
        seed=42,
    )
    constraints = api.Constraints(
        max_conversions_percent=0.5,
    )
    conversions_data = pd.DataFrame([
        {
            'date': pd.Timestamp('2024-01-01'),
            'location': f'G{i}',
            'conversions': 10.0,
        }
        for i in range(10)
    ])
    key = jax.random.PRNGKey(0)
    selection_train = jnp.zeros((1, 10))
    candidates = generate_candidates.get_random_candidates(
        filtered_data=conversions_data,
        design_config=design_config,
        constraints=constraints,
        key=key,
        selection_train=selection_train,
    )
    self.assertEqual(candidates.shape, (10, 10))

    # Verify that the sum of conversions of all treated cells (elements > 0)
    # is <= max_conversions_percent * total_conversions.
    geo_conversions = np.array([10.0] * 10)
    for i in range(design_config.n_candidates):
      treated_mask = candidates[i, :] > 0
      self.assertLessEqual(np.sum(geo_conversions[treated_mask]), 50.0)

  def test_distinct_rows_keeps_first_occurrences(self):
    masks = jnp.array([[0, 1], [1, 0], [0, 1], [1, 1], [1, 0]])
    np.testing.assert_array_equal(
        generate_candidates._distinct_rows(masks), [[0, 1], [1, 0], [1, 1]]
    )

  @parameterized.parameters((10, 5, 2), (10, 7, 3), (12, 6, 3), (9, 4, 1))
  def test_get_random_cell_sizes_matches_sampler(
      self, n_geos, n_treated, n_cells
  ):
    masks = generate_candidates._generate_random_masks(
        5,
        n_geos,
        n_treated,
        n_cells,
        jnp.zeros(n_geos, dtype=bool),
        jax.random.PRNGKey(0),
    )
    cell_sizes = generate_candidates._get_random_cell_sizes(
        n_geos, n_treated, n_cells
    )
    self.assertEqual(sum(cell_sizes), n_treated)
    for mask in np.asarray(masks):
      self.assertEqual(
          [int(np.sum(mask == cell)) for cell in range(1, n_cells + 1)],
          cell_sizes,
      )

  def test_get_random_candidates_enumerates_small_space(self):
    design_config = api.DesignConfig(
        experiment_duration=datetime.timedelta(days=1),
        experiment_types=api.ExperimentType.HOLDBACK,
        geo_assignment_rule=api.GeoAssignmentRule.RANDOM,
        n_candidates=100,
    )
    # 6 geos with max_conversions_percent=0.5 give 3 treated geos, and all
    # C(6, 3) = 20 masks satisfy the volume constraint.
    constraints = api.Constraints(max_conversions_percent=0.5)
    candidates = [
        np.asarray(
            generate_candidates.get_random_candidates(
                filtered_data=_equal_conversions_data(6),
                design_config=design_config,
                constraints=constraints,
                key=jax.random.PRNGKey(seed),
                selection_train=jnp.zeros((1, 6)),
            )
        )
        for seed in (0, 1)
    ]

    expected = set()
    for treated_geos in itertools.combinations(range(6), 3):
      mask = np.zeros(6, dtype=int)
      mask[list(treated_geos)] = 1
      expected.add(tuple(mask))
    self.assertLen(candidates[0], 20)
    self.assertEqual(
        {tuple(row.astype(int)) for row in candidates[0]}, expected
    )
    np.testing.assert_array_equal(candidates[0], candidates[1])

  def test_get_random_candidates_enumeration_excludes_forced_controls(self):
    design_config = api.DesignConfig(
        experiment_duration=datetime.timedelta(days=1),
        experiment_types=api.ExperimentType.HOLDBACK,
        geo_assignment_rule=api.GeoAssignmentRule.RANDOM,
        n_candidates=100,
    )
    constraints = api.Constraints(
        max_conversions_percent=0.5, included_control_geos={'G00'}
    )
    candidates = np.asarray(
        generate_candidates.get_random_candidates(
            filtered_data=_equal_conversions_data(6),
            design_config=design_config,
            constraints=constraints,
            key=jax.random.PRNGKey(0),
            selection_train=jnp.zeros((1, 6)),
        )
    )
    # 3 treated geos out of the 5 geos that are not forced to control.
    self.assertLen(candidates, 10)
    self.assertLen(np.unique(candidates, axis=0), 10)
    np.testing.assert_array_equal(candidates[:, 0], 0.0)

  def test_get_random_candidates_enumerates_multicell_space(self):
    design_config = api.DesignConfig(
        experiment_duration=datetime.timedelta(days=1),
        cell_count=2,
        experiment_types={
            'cell_1': api.ExperimentType.HOLDBACK,
            'cell_2': api.ExperimentType.HOLDBACK,
        },
        geo_assignment_rule=api.GeoAssignmentRule.RANDOM,
        n_candidates=1000,
    )
    # 8 geos with max_conversions_percent=0.5 give 2 treated geos per cell, so
    # there are C(8, 2) * C(6, 2) = 420 masks.
    constraints = api.Constraints(max_conversions_percent=0.5)
    candidates = np.asarray(
        generate_candidates.get_random_candidates(
            filtered_data=_equal_conversions_data(8),
            design_config=design_config,
            constraints=constraints,
            key=jax.random.PRNGKey(0),
            selection_train=jnp.zeros((1, 8)),
        )
    )
    self.assertLen(candidates, 420)
    self.assertLen(np.unique(candidates, axis=0), 420)
    np.testing.assert_array_equal(np.sum(candidates == 1, axis=1), 2)
    np.testing.assert_array_equal(np.sum(candidates == 2, axis=1), 2)

  def test_get_random_candidates_removes_duplicate_draws(self):
    design_config = api.DesignConfig(
        experiment_duration=datetime.timedelta(days=1),
        experiment_types=api.ExperimentType.HOLDBACK,
        geo_assignment_rule=api.GeoAssignmentRule.RANDOM,
        n_candidates=100,
    )
    # 10 geos with max_conversions_percent=0.35 give 3 treated geos. All
    # C(10, 3) = 120 masks are valid and exceed n_candidates, so masks are
    # sampled, and 100 draws from 120 masks contain duplicates.
    constraints = api.Constraints(max_conversions_percent=0.35)
    candidates = np.asarray(
        generate_candidates.get_random_candidates(
            filtered_data=_equal_conversions_data(10),
            design_config=design_config,
            constraints=constraints,
            key=jax.random.PRNGKey(0),
            selection_train=jnp.zeros((1, 10)),
        )
    )
    self.assertLess(len(candidates), design_config.n_candidates)
    self.assertLen(np.unique(candidates, axis=0), len(candidates))

  def test_get_stratified_sampling_candidates_removes_duplicate_draws(self):
    design_config = api.DesignConfig(
        experiment_duration=datetime.timedelta(days=1),
        experiment_types=api.ExperimentType.HOLDBACK,
        n_candidates=200,
        seed=42,
    )
    constraints = api.Constraints(max_conversions_percent=0.4)
    geo_conversions = [10.0, 20.0, 5.0, 15.0, 50.0, 100.0]
    conversions_data = pd.DataFrame([
        {
            'date': pd.Timestamp('2024-01-01'),
            'location': f'G{i}',
            'conversions': conversions,
        }
        for i, conversions in enumerate(geo_conversions)
    ])
    candidates = np.asarray(
        generate_candidates.get_stratified_sampling_candidates(
            selection_train=jnp.array([geo_conversions]),
            filtered_data=conversions_data,
            design_config=design_config,
            constraints=constraints,
            geo_stratum_labels=jnp.array([0, 0, 1, 1, 2, 2]),
            key=jax.random.PRNGKey(0),
        )
    )
    # 6 geos admit far fewer than n_candidates distinct stratified masks.
    self.assertLess(len(candidates), design_config.n_candidates)
    self.assertLen(np.unique(candidates, axis=0), len(candidates))

  @parameterized.parameters(0, 1, 2)
  def test_get_stratified_geo_sequence_matches_reference(self, seed):
    rng = np.random.default_rng(seed)
    n_geos, num_strata = 50, 4
    geo_stratum_labels = rng.integers(0, num_strata, n_geos)
    stratum_seq = rng.permutation(geo_stratum_labels)
    geos = rng.permutation(n_geos)
    stratified_geos = generate_candidates.get_stratified_geo_sequence(
        jnp.array(stratum_seq), jnp.array(geos), jnp.array(geo_stratum_labels)
    )
    self.assertEqual(stratified_geos.shape, (n_geos,))
    np.testing.assert_array_equal(
        stratified_geos,
        _reference_stratified_geo_sequence(
            stratum_seq, geos, geo_stratum_labels
        ),
    )

  @parameterized.product(num_cells=(1, 2, 3), seed=(0, 1, 2))
  def test_compute_mask_maximizing_conversions_matches_reference(
      self, num_cells, seed
  ):
    rng = np.random.default_rng(seed)
    n_geos, num_strata = 50, 4
    geos = rng.permutation(n_geos)
    geo_strata = rng.integers(0, num_strata, n_geos)
    geo_conversions = rng.lognormal(5.0, 1.0, n_geos).astype(np.float32)
    max_conversions_per_cell = float(0.3 * geo_conversions.sum()) / num_cells
    mask = generate_candidates.compute_mask_maximizing_conversions(
        jnp.array(geos),
        jnp.array(geo_strata),
        jnp.array(geo_conversions),
        max_conversions_per_cell,
        num_cells=num_cells,
    )
    self.assertEqual(mask.shape, (n_geos,))
    np.testing.assert_array_equal(
        mask,
        _reference_mask_maximizing_conversions(
            geos,
            geo_strata,
            geo_conversions,
            max_conversions_per_cell,
            num_cells,
        ),
    )


if __name__ == '__main__':
  absltest.main()
