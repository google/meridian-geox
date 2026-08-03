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
from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
from meridian_geox import api
from meridian_geox import generate_candidates
import numpy as np
import pandas as pd
from scipy import stats


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
        pad_length=20,
    )
    stratum_counts = jnp.array([2, 2, 2])
    geo_stratum_labels = jnp.array([0, 0, 1, 1, 2, 2])
    geo_conversions = jnp.array([10, 2, 1, 7, 8, 3])
    max_conversions = 15.0
    key = jax.random.PRNGKey(0)
    sampler = stats.qmc.Sobol(d=1, scramble=True, rng=42)
    sobol_seq = jnp.ravel(sampler.random(6 + design_config.pad_length))
    candidates = (
        generate_candidates.get_unconstrained_stratified_sampling_candidates(
            design_config,
            stratum_counts,
            geo_stratum_labels,
            geo_conversions,
            max_conversions,
            key,
            sobol_seq,
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
        pad_length=20,
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
    self.assertEqual(candidates.shape, (10, 6))
    # total conversions = 200. Max conversions = 0.4 * 200 = 80.
    geo_conversions = np.array([10.0, 20.0, 5.0, 15.0, 50.0, 100.0])
    for i in range(design_config.n_candidates):
      self.assertLessEqual(np.sum(geo_conversions[candidates[i, :] == 1]), 80.0)

  def test_get_stratified_sampling_candidates_min_geos_filter(self):
    design_config = api.DesignConfig(
        experiment_duration=datetime.timedelta(days=1),
        experiment_types=api.ExperimentType.HOLDBACK,
        n_candidates=10,
        seed=42,
        pad_length=20,
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


if __name__ == '__main__':
  absltest.main()
