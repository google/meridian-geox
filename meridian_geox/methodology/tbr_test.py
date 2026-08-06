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

from absl.testing import parameterized
import jax
import jax.numpy as jnp
from jax.scipy import stats
from meridian_geox import api
from meridian_geox.methodology import tbr
from meridian_geox.methodology import util as methodology_util
import numpy as np
import pandas as pd

from absl.testing import absltest


class TbrTest(parameterized.TestCase):

  def test_fit_linear_regression(self):
    # y = 2 + 3x.
    x = jnp.array([0.0, 1.0, 2.0, 3.0, 4.0])
    y = 2.0 + 3.0 * x
    alpha, beta = tbr._fit_linear_regression(x, y)
    np.testing.assert_allclose(alpha, 2.0, rtol=5e-5, atol=5e-5)
    np.testing.assert_allclose(beta, 3.0, rtol=5e-5, atol=5e-5)

  def test_fit_linear_regression_constant_x(self):
    # x is constant. Denominator in slope calculation is 0.
    x = jnp.array([100.0, 100.0, 100.0])
    y = jnp.array([110.0, 120.0, 130.0])
    alpha, beta = tbr._fit_linear_regression(x, y)
    # Slope should be 0.0, intercept should be mean(y) = 120.0.
    np.testing.assert_allclose(beta, 0.0, rtol=5e-5, atol=5e-5)
    np.testing.assert_allclose(alpha, 120.0, rtol=5e-5, atol=5e-5)

  def test_compute_group_mean(self):
    # 2 time points, 4 geos.
    data = jnp.array([[10.0, 20.0, 30.0, 40.0], [15.0, 25.0, 35.0, 45.0]])
    # Geos 0 and 2 are treated (indices 0, 2).
    mask = jnp.array([1.0, 0.0, 1.0, 0.0])

    y_mean = tbr._compute_group_mean(data, mask, 1.0)
    x_mean = tbr._compute_group_mean(data, mask, 0.0)

    # Treated: (10+30)/2=20, (15+35)/2=25.
    expected_y_mean = jnp.array([20.0, 25.0])
    # Control: (20+40)/2=30, (25+45)/2=35.
    expected_x_mean = jnp.array([30.0, 35.0])

    np.testing.assert_allclose(y_mean, expected_y_mean, rtol=5e-5, atol=5e-5)
    np.testing.assert_allclose(x_mean, expected_x_mean, rtol=5e-5, atol=5e-5)

  def test_get_r2_perfect_correlation(self):
    # Create data where treated is exactly 2 * control + 1.
    t = 10
    n_geos = 4
    # Control data.
    control_data = jax.random.normal(jax.random.PRNGKey(0), (t, n_geos // 2))
    # Treated data perfectly correlated.
    treated_data = 2.0 * control_data + 1.0

    data_pre = jnp.concatenate([treated_data, control_data], axis=1)
    data_val = data_pre  # validation same as pre for this test.

    # First half is treated, second half is control.
    mask = jnp.concatenate([jnp.ones(n_geos // 2), jnp.zeros(n_geos // 2)])
    treatment_masks = mask[None, :]  # Batch of size 1.

    r2 = tbr.get_r2(data_pre, data_val, treatment_masks, jnp.array([1.0]))

    # Verify that R2 is close to 1.0 due to perfect linear relationship.
    np.testing.assert_allclose(r2, 1.0, rtol=5e-5, atol=5e-5)

  def test_get_r2_negative_correlation_clamped_to_zero(self):
    # Create data where treated is negatively correlated with control
    # (-2 * control + 1).
    t = 10
    n_geos = 4
    control_data = jax.random.normal(jax.random.PRNGKey(0), (t, n_geos // 2))
    treated_data = -2.0 * control_data + 1.0

    data_pre = jnp.concatenate([treated_data, control_data], axis=1)
    data_val = data_pre

    mask = jnp.concatenate([jnp.ones(n_geos // 2), jnp.zeros(n_geos // 2)])
    treatment_masks = mask[None, :]

    r2 = tbr.get_r2(data_pre, data_val, treatment_masks, jnp.array([1.0]))

    # Because negative slope is clamped to 0, R2 is <= 0.0.
    self.assertLessEqual(float(r2[0, 0]), 0.0)

  def test_fit_linear_regression_negative_slope_clamped(self):
    x = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y = jnp.array([5.0, 4.0, 3.0, 2.0, 1.0])  # negative slope -1.0
    intercept, slope = tbr._fit_linear_regression(x, y)
    self.assertAlmostEqual(float(slope), 0.0)
    self.assertAlmostEqual(float(intercept), 3.0)

  def test_get_r2_multicell(self):
    t = 10
    # Control data.
    control_data = jax.random.normal(jax.random.PRNGKey(0), (t, 2))
    # Cell 1 data perfectly correlated.
    cell1_data = 2.0 * control_data + 1.0
    # Cell 2 data perfectly correlated with different slope/intercept.
    cell2_data = 0.5 * control_data - 1.0

    data_pre = jnp.concatenate([cell1_data, cell2_data, control_data], axis=1)
    data_val = data_pre  # validation same as pre for this test.

    # Geos 0,1 are Cell 1; Geos 2,3 are Cell 2; Geos 4,5 are Control.
    mask = jnp.array([1.0, 1.0, 2.0, 2.0, 0.0, 0.0])
    treatment_masks = mask[None, :]  # Batch of size 1.

    cell_ids = jnp.array([1.0, 2.0])
    r2 = tbr.get_r2(data_pre, data_val, treatment_masks, cell_ids)

    # Verify that R2 is close to 1.0 for both cells due to perfect linear
    # relationship. Shape should be (1, 2)
    self.assertEqual(r2.shape, (1, 2))
    np.testing.assert_allclose(r2, 1.0, rtol=5e-5, atol=5e-5)

  def test_get_r2_invalid_design(self):
    t = 10
    n_geos = 4
    data_pre = jnp.zeros((t, n_geos))
    data_val = jnp.zeros((t, n_geos))

    # No treated geos in these designs.
    mask_zeros = jnp.zeros(n_geos)
    # No control geos in these designs.
    mask_ones = jnp.ones(n_geos)

    treatment_masks = jnp.stack([mask_zeros, mask_ones])

    r2 = tbr.get_r2(data_pre, data_val, treatment_masks, jnp.array([1.0]))

    self.assertTrue(jnp.isnan(r2[0, 0]))
    self.assertTrue(jnp.isnan(r2[1, 0]))

  def test_get_mde_identical_geos_simplified_design_aware_placebo(self):
    t_pre = 10
    t_val = 5
    n_geos = 3
    n_treated = 1

    # Create identical data for all geos.
    # Create a time series like 0, 1, 2, ...
    time_series = jnp.arange(t_pre + t_val, dtype=jnp.float32)
    # Replicate for all geos: (T, N).
    data = jnp.tile(time_series[:, None], (1, n_geos))

    data_pre = data[:t_pre, :]
    data_val = data[t_pre:, :]

    masks = jnp.zeros((1, n_geos))
    masks = masks.at[0, :n_treated].set(1.0)

    # Dummy keys for simplified design aware placebo test.
    key = jax.random.PRNGKey(0)
    random_keys = jax.random.split(key, 1)

    results = tbr.get_mde(
        data_pre=data_pre,
        data_val=data_val,
        treatment_masks=masks,
        random_keys=random_keys,
        n_permutations=100,
        z_score_sum=1.96,
        se_method=api.SeMethod.SIMPLIFIED_DESIGN_AWARE_PLACEBO,
    )

    # MDE should be 0 because all effects are exactly 0 (SE=0).
    np.testing.assert_allclose(results.mde_abs, 0.0, rtol=5e-5, atol=5e-5)
    np.testing.assert_allclose(results.mde_pct, 0.0, rtol=5e-5, atol=5e-5)

  @parameterized.named_parameters(
      dict(
          testcase_name='two_sided',
          test_type=api.TestType.TWO_SIDED,
      ),
      dict(
          testcase_name='one_sided',
          test_type=api.TestType.ONE_SIDED,
      ),
  )
  def test_get_mde_simplified_design_aware_placebo(self, test_type):
    t_pre = 10
    t_val = 5
    n_geos = 10
    n_treated = 4
    n_designs = 3

    key = jax.random.PRNGKey(42)
    # Generate random data.
    data_pre = jax.random.normal(key, (t_pre, n_geos))
    data_val = jax.random.normal(key, (t_val, n_geos))

    # Create distinct masks.
    masks = jnp.zeros((n_designs, n_geos))
    for i in range(n_designs):
      # Treat units i to i+n_treated.
      start = i
      end = start + n_treated
      masks = masks.at[i, start:end].set(1.0)

    z_score_sum = 1.96

    # Dummy keys for simplified design aware placebo test.
    random_keys = jax.random.split(key, n_designs)

    results = tbr.get_mde(
        data_pre=data_pre,
        data_val=data_val,
        treatment_masks=masks,
        random_keys=random_keys,
        n_permutations=100,
        z_score_sum=z_score_sum,
        test_type=test_type,
        se_method=api.SeMethod.SIMPLIFIED_DESIGN_AWARE_PLACEBO,
    )

    # Squeeze the results to remove the extra dimension since there is only 1
    # treatment cell.
    results.mde_abs = results.mde_abs.squeeze()
    results.mde_pct = results.mde_pct.squeeze()
    results.p_value = results.p_value.squeeze()
    results.observed_conversions = results.observed_conversions.squeeze()
    results.counterfactual_conversions = (
        results.counterfactual_conversions.squeeze()
    )

    self.assertEqual(results.mde_abs.shape, (n_designs,))
    self.assertEqual(results.mde_pct.shape, (n_designs,))
    self.assertEqual(results.p_value.shape, (n_designs,))
    self.assertEqual(
        results.observed_conversions.shape, (n_designs, t_pre + t_val)
    )
    self.assertEqual(
        results.counterfactual_conversions.shape, (n_designs, t_pre + t_val)
    )

    # mde_abs should be constant across designs because SE is derived from the
    # set check that values are close.
    first_mde = results.mde_abs[0]
    np.testing.assert_allclose(results.mde_abs, first_mde, atol=1e-6)

    # Calculate effects manually to verify results.
    effects = []
    baselines = []
    for i in range(n_designs):
      mask = masks[i]
      y_pre = tbr._compute_group_mean(data_pre, mask, 1.0)
      x_pre = tbr._compute_group_mean(data_pre, mask, 0.0)
      alpha, beta = tbr._fit_linear_regression(x_pre, y_pre)
      y_val = tbr._compute_group_mean(data_val, mask, 1.0)
      x_val = tbr._compute_group_mean(data_val, mask, 0.0)
      y_pred = alpha + beta * x_val
      real_effect = jnp.mean(y_val - y_pred)
      effects.append(real_effect)
      baselines.append(jnp.mean(y_val))

    effects = jnp.array(effects)
    baselines = jnp.array(baselines)

    # 1. Verify MDE Abs.
    expected_se = jnp.std(effects)
    expected_mde_abs = expected_se * z_score_sum
    np.testing.assert_allclose(
        results.mde_abs[0], expected_mde_abs, rtol=5e-5, atol=5e-5
    )

    # 2. Verify MDE Pct.
    expected_mde_pct = expected_mde_abs / baselines
    np.testing.assert_allclose(
        results.mde_pct, expected_mde_pct, rtol=5e-5, atol=5e-5
    )

    # 3. Verify P-values.
    # z_scores = effects / se.
    z_scores = effects / expected_se
    if test_type == api.TestType.TWO_SIDED:
      # p = 2 * (1 - cdf(|z|)).
      expected_p_values = 2.0 * (1.0 - stats.norm.cdf(jnp.abs(z_scores)))
    else:
      # p = 1 - cdf(z).
      expected_p_values = 1.0 - stats.norm.cdf(z_scores)

    np.testing.assert_allclose(
        results.p_value, expected_p_values, rtol=5e-5, atol=5e-5
    )

  def test_get_mde_single_design_simplified_design_aware_placebo(self):
    t_pre = 10
    t_val = 5
    n_geos = 5
    n_treated = 2
    n_designs = 1

    key = jax.random.PRNGKey(123)
    data_pre = jax.random.normal(key, (t_pre, n_geos))
    data_val = jax.random.normal(key, (t_val, n_geos)) + 10.0

    masks = jnp.zeros((n_designs, n_geos))
    masks = masks.at[0, :n_treated].set(1.0)

    # Dummy keys for simplified design aware placebo test.
    random_keys = jax.random.split(key, n_designs)

    results = tbr.get_mde(
        data_pre=data_pre,
        data_val=data_val,
        treatment_masks=masks,
        random_keys=random_keys,
        n_permutations=100,
        z_score_sum=1.96,
        se_method=api.SeMethod.SIMPLIFIED_DESIGN_AWARE_PLACEBO,
    )

    # With only 1 design, std(effects) is 0.
    np.testing.assert_allclose(results.mde_abs, 0.0, atol=1e-6)
    np.testing.assert_allclose(results.mde_pct, 0.0, atol=1e-6)
    # When SE is 0, p-value should default to 1.0.
    np.testing.assert_allclose(results.p_value, 1.0, atol=1e-6)

  @parameterized.named_parameters(
      dict(
          testcase_name='two_sided',
          test_type=api.TestType.TWO_SIDED,
      ),
      dict(
          testcase_name='one_sided',
          test_type=api.TestType.ONE_SIDED,
      ),
  )
  def test_get_mde_multicell_simplified_design_aware_placebo(self, test_type):
    t_pre = 10
    t_val = 5
    n_geos = 12
    n_designs = 3
    cell_ids = jnp.array([1.0, 2.0])
    k_cells = len(cell_ids)

    key = jax.random.PRNGKey(42)
    # Generate random data.
    data_pre = jax.random.normal(key, (t_pre, n_geos)) + 10.0
    data_val = jax.random.normal(key, (t_val, n_geos)) + 10.0

    # Create distinct masks.
    masks = jnp.zeros((n_designs, n_geos))
    for i in range(n_designs):
      masks = masks.at[i, (i) % n_geos].set(1.0)
      masks = masks.at[i, (i + 1) % n_geos].set(1.0)
      masks = masks.at[i, (i + 2) % n_geos].set(2.0)
      masks = masks.at[i, (i + 3) % n_geos].set(2.0)

    z_score_sum = 1.96
    random_keys = jax.random.split(key, n_designs)

    results = tbr.get_mde(
        data_pre=data_pre,
        data_val=data_val,
        treatment_masks=masks,
        random_keys=random_keys,
        n_permutations=100,
        z_score_sum=z_score_sum,
        test_type=test_type,
        se_method=api.SeMethod.SIMPLIFIED_DESIGN_AWARE_PLACEBO,
        cell_ids=cell_ids,
    )

    self.assertEqual(results.mde_abs.shape, (n_designs, k_cells))
    self.assertEqual(results.mde_pct.shape, (n_designs, k_cells))
    self.assertEqual(results.p_value.shape, (n_designs, k_cells))
    self.assertEqual(
        results.observed_conversions.shape, (n_designs, k_cells, t_pre + t_val)
    )
    self.assertEqual(
        results.counterfactual_conversions.shape,
        (n_designs, k_cells, t_pre + t_val),
    )

    # Verify that MDE abs is constant across designs within each cell (but can
    # differ between cells).
    for c in range(k_cells):
      first_mde = results.mde_abs[0, c]
      np.testing.assert_allclose(results.mde_abs[:, c], first_mde, atol=1e-6)

    # Verify that standard calculation matches.
    for c in range(k_cells):
      cell_id = cell_ids[c]
      effects = []
      baselines = []
      for i in range(n_designs):
        mask = masks[i]
        y_pre = tbr._compute_group_mean(data_pre, mask, cell_id)  # pyrefly: ignore[bad-argument-type]
        x_pre = tbr._compute_group_mean(data_pre, mask, 0.0)
        alpha, beta = tbr._fit_linear_regression(x_pre, y_pre)
        y_val = tbr._compute_group_mean(data_val, mask, cell_id)  # pyrefly: ignore[bad-argument-type]
        x_val = tbr._compute_group_mean(data_val, mask, 0.0)
        y_pred = alpha + beta * x_val
        real_effect = jnp.mean(y_val - y_pred)
        effects.append(real_effect)
        baselines.append(jnp.mean(y_val))

      effects = jnp.array(effects)
      baselines = jnp.array(baselines)

      expected_se = jnp.std(effects)
      expected_mde_abs = expected_se * z_score_sum
      np.testing.assert_allclose(
          results.mde_abs[:, c], expected_mde_abs, rtol=5e-5, atol=5e-5
      )

      expected_mde_pct = expected_mde_abs / baselines
      np.testing.assert_allclose(
          results.mde_pct[:, c], expected_mde_pct, rtol=5e-5, atol=5e-5
      )

      z_scores = effects / expected_se
      if test_type == api.TestType.TWO_SIDED:
        expected_p_values = 2.0 * (1.0 - stats.norm.cdf(jnp.abs(z_scores)))
      else:
        expected_p_values = 1.0 - stats.norm.cdf(z_scores)
      np.testing.assert_allclose(
          results.p_value[:, c], expected_p_values, rtol=5e-5, atol=5e-5
      )

  def test_get_mde_multicell_identical_geos_simplified_design_aware_placebo(
      self,
  ):
    t_pre = 10
    t_val = 5
    n_geos = 6
    cell_ids = jnp.array([1.0, 2.0])

    # Create identical data for all geos.
    time_series = jnp.arange(t_pre + t_val, dtype=jnp.float32)
    data = jnp.tile(time_series[:, None], (1, n_geos))

    data_pre = data[:t_pre, :]
    data_val = data[t_pre:, :]

    masks = jnp.array([[1.0, 1.0, 2.0, 2.0, 0.0, 0.0]])

    key = jax.random.PRNGKey(0)
    random_keys = jax.random.split(key, 1)

    results = tbr.get_mde(
        data_pre=data_pre,
        data_val=data_val,
        treatment_masks=masks,
        random_keys=random_keys,
        n_permutations=100,
        z_score_sum=1.96,
        se_method=api.SeMethod.SIMPLIFIED_DESIGN_AWARE_PLACEBO,
        cell_ids=cell_ids,
    )

    # MDE should be 0 because all effects are exactly 0 (SE=0).
    np.testing.assert_allclose(results.mde_abs, 0.0, rtol=5e-5, atol=5e-5)
    np.testing.assert_allclose(results.mde_pct, 0.0, rtol=5e-5, atol=5e-5)

  def test_get_mde_multicell_placebo_raises_not_implemented(self):
    t_pre = 10
    t_val = 5
    n_geos = 6
    cell_ids = jnp.array([1.0, 2.0])

    data_pre = jnp.zeros((t_pre, n_geos))
    data_val = jnp.zeros((t_val, n_geos))
    masks = jnp.array([[1.0, 1.0, 2.0, 2.0, 0.0, 0.0]])

    key = jax.random.PRNGKey(0)
    random_keys = jax.random.split(key, 1)

    with self.assertRaisesRegex(
        NotImplementedError,
        'Multicell support is not implemented for _get_mde_placebo.*',
    ):
      tbr.get_mde(
          data_pre=data_pre,
          data_val=data_val,
          treatment_masks=masks,
          random_keys=random_keys,
          n_permutations=20,
          z_score_sum=1.96,
          se_method=api.SeMethod.PLACEBO,
          cell_ids=cell_ids,
      )

  @parameterized.named_parameters(
      dict(
          testcase_name='two_sided',
          test_type=api.TestType.TWO_SIDED,
      ),
      dict(
          testcase_name='one_sided',
          test_type=api.TestType.ONE_SIDED,
      ),
  )
  def test_get_mde_placebo(self, test_type):
    t_pre = 10
    t_val = 5
    n_geos = 10
    n_treated = 4
    n_designs = 2

    key = jax.random.PRNGKey(42)
    # Generate random data.
    data_pre = jax.random.normal(key, (t_pre, n_geos))
    data_val = jax.random.normal(key, (t_val, n_geos))

    # Create distinct masks.
    masks = jnp.zeros((n_designs, n_geos))
    for i in range(n_designs):
      # Treat units i to i+n_treated.
      start = i
      end = start + n_treated
      masks = masks.at[i, start:end].set(1.0)

    z_score_sum = 1.96
    random_keys = jax.random.split(key, n_designs)
    n_permutations = 50

    results = tbr.get_mde(
        data_pre=data_pre,
        data_val=data_val,
        treatment_masks=masks,
        random_keys=random_keys,
        n_permutations=n_permutations,
        z_score_sum=z_score_sum,
        test_type=test_type,
        se_method=api.SeMethod.PLACEBO,
    )

    # Squeeze the results to remove the extra dimension since there is only 1
    # treatment cell.
    results.mde_abs = results.mde_abs.squeeze()
    results.mde_pct = results.mde_pct.squeeze()
    results.p_value = results.p_value.squeeze()
    results.observed_conversions = results.observed_conversions.squeeze()
    results.counterfactual_conversions = (
        results.counterfactual_conversions.squeeze()
    )

    self.assertEqual(results.mde_abs.shape, (n_designs,))
    self.assertEqual(results.mde_pct.shape, (n_designs,))
    self.assertEqual(results.p_value.shape, (n_designs,))
    self.assertEqual(
        results.observed_conversions.shape, (n_designs, t_pre + t_val)
    )
    self.assertEqual(
        results.counterfactual_conversions.shape, (n_designs, t_pre + t_val)
    )

    # With random data, MDE should be > 0.
    self.assertTrue(jnp.all(results.mde_abs > 0.0))
    self.assertTrue(jnp.all(results.mde_pct > 0.0))

    # P-values should be in [0, 1].
    self.assertTrue(jnp.all(results.p_value >= 0.0))
    self.assertTrue(jnp.all(results.p_value <= 1.0))

  def test_get_mde_identical_geos_placebo(self):
    t_pre = 10
    t_val = 5
    n_geos = 10
    n_treated = 2

    # Create identical data for all geos.
    time_series = jnp.arange(t_pre + t_val, dtype=jnp.float32)
    data = jnp.tile(time_series[:, None], (1, n_geos))
    data_pre = data[:t_pre, :]
    data_val = data[t_pre:, :]

    masks = jnp.zeros((1, n_geos))
    masks = masks.at[0, :n_treated].set(1.0)

    key = jax.random.PRNGKey(0)
    random_keys = jax.random.split(key, 1)

    results = tbr.get_mde(
        data_pre=data_pre,
        data_val=data_val,
        treatment_masks=masks,
        random_keys=random_keys,
        n_permutations=20,
        z_score_sum=1.96,
        se_method=api.SeMethod.PLACEBO,
    )

    # Placebo effects should all be 0 because any control group fits perfectly.
    # Therefore std(placebo_effects) = 0 -> MDE = 0.
    np.testing.assert_allclose(results.mde_abs, 0.0, rtol=5e-5, atol=5e-5)
    np.testing.assert_allclose(results.mde_pct, 0.0, rtol=5e-5, atol=5e-5)

  def test_get_mde_placebo_reproducibility(self):
    t_pre = 10
    t_val = 5
    n_geos = 10
    n_treated = 4
    n_designs = 2

    key = jax.random.PRNGKey(42)
    data_pre = jax.random.normal(key, (t_pre, n_geos))
    data_val = jax.random.normal(key, (t_val, n_geos))

    masks = jnp.zeros((n_designs, n_geos))
    for i in range(n_designs):
      masks = masks.at[i, :n_treated].set(1.0)

    z_score_sum = 1.96
    random_keys = jax.random.split(key, n_designs)
    n_permutations = 50

    results1 = tbr.get_mde(
        data_pre=data_pre,
        data_val=data_val,
        treatment_masks=masks,
        random_keys=random_keys,
        n_permutations=n_permutations,
        z_score_sum=z_score_sum,
        se_method=api.SeMethod.PLACEBO,
    )

    results2 = tbr.get_mde(
        data_pre=data_pre,
        data_val=data_val,
        treatment_masks=masks,
        random_keys=random_keys,
        n_permutations=n_permutations,
        z_score_sum=z_score_sum,
        se_method=api.SeMethod.PLACEBO,
    )

    np.testing.assert_allclose(results1.mde_abs, results2.mde_abs, atol=1e-6)
    np.testing.assert_allclose(results1.mde_pct, results2.mde_pct, atol=1e-6)
    np.testing.assert_allclose(results1.p_value, results2.p_value, atol=1e-6)

  def test_generate_analysis_holdback(self):
    # Setup 6 geos: 2 treatment, 4 control
    treatment_geos = {'cell_1': {'G0', 'G1'}}
    geos = sorted(['G0', 'G1', 'G2', 'G3', 'G4', 'G5'])

    # Setup data: 10 days pre, 5 days test
    dates = pd.date_range('2024-01-01', periods=15)
    data_list = []
    for date in dates:
      for geo in geos:
        val = 100.0
        spend = 10.0
        if (
            date >= pd.Timestamp('2024-01-11')
            and geo in treatment_geos['cell_1']
        ):
          val += 10.0  # 10% lift
          spend = 20.0
        data_list.append({
            'date': date,
            'location': geo,
            'conversions': val,
            'spend': spend,
        })

    conversions_data = pd.DataFrame(data_list)

    treatment_mask = jnp.array([1, 1, 0, 0, 0, 0])  # G0, G1 are first
    # Generate some dummy placebo masks (simulating random designs)
    # They should have 0s for G0, G1 (original treatment)
    # and some 1s for others
    placebo_masks = jnp.array([
        [0, 0, 1, 1, 0, 0],
        [0, 0, 0, 0, 1, 1],
        [0, 0, 1, 0, 1, 0],
        [0, 0, 0, 1, 0, 1],
        [0, 0, 1, 0, 0, 1],
        [0, 0, 0, 1, 1, 0],
        [0, 0, 1, 1, 0, 0],
        [0, 0, 0, 0, 1, 1],
        [0, 0, 1, 0, 1, 0],
        [0, 0, 0, 1, 0, 1],
    ])

    pivoted_conversions = (
        conversions_data.pivot_table(
            index='date',
            columns='location',
            values='conversions',
            aggfunc='sum',
        )
        .sort_index()
        .reindex(sorted(geos), axis=1)
    )
    pretest_conversions = jnp.array(
        pivoted_conversions[
            pivoted_conversions.index < pd.Timestamp('2024-01-11')
        ].values
    )
    test_conversions = jnp.array(
        pivoted_conversions[
            (pivoted_conversions.index >= pd.Timestamp('2024-01-11'))
            & (pivoted_conversions.index <= pd.Timestamp('2024-01-15'))
        ].values
    )

    pivoted_spend = (
        conversions_data.pivot_table(
            index='date', columns='location', values='spend', aggfunc='sum'
        )
        .sort_index()
        .reindex(sorted(geos), axis=1)
    )
    pretest_spend = jnp.array(
        pivoted_spend[pivoted_spend.index < pd.Timestamp('2024-01-11')].values
    )
    test_spend = jnp.array(
        pivoted_spend[
            (pivoted_spend.index >= pd.Timestamp('2024-01-11'))
            & (pivoted_spend.index <= pd.Timestamp('2024-01-15'))
        ].values
    )

    pretest_train_conversions = pretest_conversions[:-5]
    pretest_val_conversions = pretest_conversions[-5:]

    tbr_result = tbr.analyze(
        pretest_train_conversions=pretest_train_conversions,
        pretest_val_conversions=pretest_val_conversions,
        test_conversions=test_conversions,
        treatment_mask=treatment_mask,
        placebo_masks=placebo_masks,
        alpha=0.1,
        experiment_type=api.ExperimentType.HOLDBACK,
        treatment_cell_id=1.0,
        pretest_spend=pretest_spend,
        test_spend=test_spend,
    )

    metrics = api.AnalysisMetrics(
        lift=tbr_result.lift,
        percent_lift=tbr_result.percent_lift,
        cumulative_lift=pd.DataFrame(
            data=tbr_result.cumulative_lift_with_cis,
            index=pd.date_range('2024-01-11', periods=5),
            columns=['lift', 'lower_bound', 'upper_bound'],
        ),
        counterfactual_conversions=pd.DataFrame(
            data=tbr_result.counterfactual_conversions_with_cis,
            index=pd.date_range('2024-01-01', periods=15),
            columns=[
                'observed',
                'counterfactual',
                'lower_bound',
                'upper_bound',
            ],
        ),
        pointwise_difference=pd.DataFrame(
            data=tbr_result.pointwise_difference_with_cis,
            index=pd.date_range('2024-01-01', periods=15),
            columns=['difference', 'lower_bound', 'upper_bound'],
        ),
        icpd=tbr_result.icpd,
        cumulative_icpd=pd.DataFrame(
            data=tbr_result.cumulative_icpd_with_cis,
            index=pd.date_range('2024-01-11', periods=5),
            columns=['icpd', 'lower_bound', 'upper_bound'],
        ),
    )

    self.assertIsInstance(metrics, api.AnalysisMetrics)
    self.assertLen(metrics.counterfactual_conversions, 15)
    self.assertIsNotNone(tbr_result.counterfactual_spend)
    assert tbr_result.counterfactual_spend is not None
    self.assertEqual(tbr_result.counterfactual_spend.shape, (5,))
    self.assertIn('observed', metrics.counterfactual_conversions.columns)
    self.assertLen(metrics.pointwise_difference, 15)
    self.assertIn('difference', metrics.pointwise_difference.columns)
    # 2 treatment geos * 10 lift/day * 5 days = 100 total lift
    self.assertAlmostEqual(metrics.lift.point_estimate, 100.0, delta=1.0)
    # Total conversions in treatment: 2 * (110 * 5) = 1100
    # Percent lift: 100 / 1100 = ~0.0909
    self.assertAlmostEqual(
        metrics.percent_lift.point_estimate, 100.0 / 1100.0, delta=0.01
    )
    # Since we added a clear lift, p-value should be small
    self.assertLess(metrics.lift.p_value, 0.5)

    self.assertLen(metrics.cumulative_lift, 5)
    self.assertIn('lift', metrics.cumulative_lift.columns)

    # Total lift per day = 2 geos * 10 = 20
    # Total incremental spend per day = 2 geos * (20 - 10) = 20
    # ICPD = 20 / 20 = 1.0
    assert metrics.icpd is not None
    self.assertAlmostEqual(metrics.icpd.point_estimate, 1.0, delta=0.01)
    # p-value should be the same as lift p-value
    self.assertEqual(metrics.icpd.p_value, metrics.lift.p_value)

    assert metrics.cumulative_icpd is not None
    self.assertLen(metrics.cumulative_icpd, 5)
    self.assertIn('icpd', metrics.cumulative_icpd.columns)
    self.assertIn('lower_bound', metrics.cumulative_icpd.columns)
    self.assertIn('upper_bound', metrics.cumulative_icpd.columns)

  def test_generate_analysis_go_dark(self):
    # Setup 6 geos: 2 treatment, 4 control
    treatment_geos = {'cell_1': {'G0', 'G1'}}
    geos = sorted(['G0', 'G1', 'G2', 'G3', 'G4', 'G5'])

    # Setup data: 10 days pre, 5 days test
    # In GO_DARK, we expect conversions and spend to DROP.
    dates = pd.date_range('2024-01-01', periods=15)
    data_list = []
    for date in dates:
      for geo in geos:
        val = 100.0
        spend = 10.0
        if (
            date >= pd.Timestamp('2024-01-11')
            and geo in treatment_geos['cell_1']
        ):
          val -= 10.0  # 10% drop
          spend = 0.0  # Stop spending
        data_list.append({
            'date': date,
            'location': geo,
            'conversions': val,
            'spend': spend,
        })

    conversions_data = pd.DataFrame(data_list)

    treatment_mask = jnp.array([1, 1, 0, 0, 0, 0])
    placebo_masks = jnp.array([
        [0, 0, 1, 1, 0, 0],
        [0, 0, 0, 0, 1, 1],
        [0, 0, 1, 0, 1, 0],
        [0, 0, 0, 1, 0, 1],
        [0, 0, 1, 0, 0, 1],
        [0, 0, 0, 1, 1, 0],
        [0, 0, 1, 1, 0, 0],
        [0, 0, 0, 0, 1, 1],
        [0, 0, 1, 0, 1, 0],
        [0, 0, 0, 1, 0, 1],
    ])

    pivoted_conversions = (
        conversions_data.pivot_table(
            index='date',
            columns='location',
            values='conversions',
            aggfunc='sum',
        )
        .sort_index()
        .reindex(sorted(geos), axis=1)
    )
    pretest_conversions = jnp.array(
        pivoted_conversions[
            pivoted_conversions.index < pd.Timestamp('2024-01-11')
        ].values
    )
    test_conversions = jnp.array(
        pivoted_conversions[
            (pivoted_conversions.index >= pd.Timestamp('2024-01-11'))
            & (pivoted_conversions.index <= pd.Timestamp('2024-01-15'))
        ].values
    )

    pivoted_spend = (
        conversions_data.pivot_table(
            index='date', columns='location', values='spend', aggfunc='sum'
        )
        .sort_index()
        .reindex(sorted(geos), axis=1)
    )
    pretest_spend = jnp.array(
        pivoted_spend[pivoted_spend.index < pd.Timestamp('2024-01-11')].values
    )
    test_spend = jnp.array(
        pivoted_spend[
            (pivoted_spend.index >= pd.Timestamp('2024-01-11'))
            & (pivoted_spend.index <= pd.Timestamp('2024-01-15'))
        ].values
    )

    pretest_train_conversions = pretest_conversions[:-5]
    pretest_val_conversions = pretest_conversions[-5:]

    tbr_result = tbr.analyze(
        pretest_train_conversions=pretest_train_conversions,
        pretest_val_conversions=pretest_val_conversions,
        test_conversions=test_conversions,
        treatment_mask=treatment_mask,
        placebo_masks=placebo_masks,
        alpha=0.1,
        experiment_type=api.ExperimentType.GO_DARK,
        treatment_cell_id=1.0,
        pretest_spend=pretest_spend,
        test_spend=test_spend,
    )

    metrics = api.AnalysisMetrics(
        lift=tbr_result.lift,
        percent_lift=tbr_result.percent_lift,
        cumulative_lift=pd.DataFrame(
            data=tbr_result.cumulative_lift_with_cis,
            index=pd.date_range('2024-01-11', periods=5),
            columns=['lift', 'lower_bound', 'upper_bound'],
        ),
        counterfactual_conversions=pd.DataFrame(
            data=tbr_result.counterfactual_conversions_with_cis,
            index=pd.date_range('2024-01-01', periods=15),
            columns=[
                'observed',
                'counterfactual',
                'lower_bound',
                'upper_bound',
            ],
        ),
        pointwise_difference=pd.DataFrame(
            data=tbr_result.pointwise_difference_with_cis,
            index=pd.date_range('2024-01-01', periods=15),
            columns=['difference', 'lower_bound', 'upper_bound'],
        ),
        icpd=tbr_result.icpd,
        cumulative_icpd=pd.DataFrame(
            data=tbr_result.cumulative_icpd_with_cis,
            index=pd.date_range('2024-01-11', periods=5),
            columns=['icpd', 'lower_bound', 'upper_bound'],
        ),
    )

    self.assertIsInstance(metrics, api.AnalysisMetrics)
    self.assertLen(metrics.counterfactual_conversions, 15)
    self.assertIn('observed', metrics.counterfactual_conversions.columns)
    self.assertLen(metrics.pointwise_difference, 15)
    self.assertIn('difference', metrics.pointwise_difference.columns)
    # Lift should be POSITIVE because we negated the negative drop.
    # 2 treatment geos * 10 drop/day * 5 days = 100 total absolute drop.
    # Negating it gives 100.0 lift.
    self.assertAlmostEqual(metrics.lift.point_estimate, 100.0, delta=1.0)

    # Incremental spend: stop spending 10.0 per day for 2 geos for
    # 5 days = 100.0.
    # Negating -100.0 gives 100.0.
    # ICPD = 100.0 / 100.0 = 1.0.
    assert metrics.icpd is not None
    self.assertAlmostEqual(metrics.icpd.point_estimate, 1.0, delta=0.01)

  def test_generate_analysis_with_treatment_cell_id(self):
    # Setup 6 geos: 1 treatment in cell 1, 1 treatment in cell 2, 4 control
    treatment_geos = {'cell_1': {'G0'}, 'cell_2': {'G1'}}
    geos = sorted(['G0', 'G1', 'G2', 'G3', 'G4', 'G5'])

    # Setup data: 10 days pre, 5 days test. G1 (cell 2) gets lift.
    dates = pd.date_range('2024-01-01', periods=15)
    data_list = []
    for date in dates:
      for geo in geos:
        val = 100.0
        spend = 10.0
        if (
            date >= pd.Timestamp('2024-01-11')
            and geo in treatment_geos['cell_2']
        ):
          val += 10.0  # 10% lift
          spend = 20.0
        elif (
            date >= pd.Timestamp('2024-01-11')
            and geo in treatment_geos['cell_1']
        ):
          val += 5.0  # some lift in cell 1, ignored by cell 2 analysis
          spend = 15.0
        data_list.append({
            'date': date,
            'location': geo,
            'conversions': val,
            'spend': spend,
        })

    conversions_data = pd.DataFrame(data_list)

    # Use 1.0 for cell 1, 2.0 for cell 2, 0.0 for control
    treatment_mask = jnp.array([1, 2, 0, 0, 0, 0])
    placebo_masks = jnp.array([
        [0, 0, 1, 2, 0, 0],
        [0, 0, 0, 0, 1, 2],
        [0, 0, 1, 0, 2, 0],
        [0, 0, 0, 1, 0, 2],
        [0, 0, 1, 0, 0, 2],
        [0, 0, 0, 1, 2, 0],
        [0, 0, 1, 2, 0, 0],
        [0, 0, 0, 0, 1, 2],
        [0, 0, 1, 0, 2, 0],
        [0, 0, 0, 1, 0, 2],
    ])

    pivoted_conversions = (
        conversions_data.pivot_table(
            index='date',
            columns='location',
            values='conversions',
            aggfunc='sum',
        )
        .sort_index()
        .reindex(sorted(geos), axis=1)
    )
    pretest_conversions = jnp.array(
        pivoted_conversions[
            pivoted_conversions.index < pd.Timestamp('2024-01-11')
        ].values
    )
    test_conversions = jnp.array(
        pivoted_conversions[
            (pivoted_conversions.index >= pd.Timestamp('2024-01-11'))
            & (pivoted_conversions.index <= pd.Timestamp('2024-01-15'))
        ].values
    )

    pivoted_spend = (
        conversions_data.pivot_table(
            index='date', columns='location', values='spend', aggfunc='sum'
        )
        .sort_index()
        .reindex(sorted(geos), axis=1)
    )
    pretest_spend = jnp.array(
        pivoted_spend[pivoted_spend.index < pd.Timestamp('2024-01-11')].values
    )
    test_spend = jnp.array(
        pivoted_spend[
            (pivoted_spend.index >= pd.Timestamp('2024-01-11'))
            & (pivoted_spend.index <= pd.Timestamp('2024-01-15'))
        ].values
    )

    pretest_train_conversions = pretest_conversions[:-5]
    pretest_val_conversions = pretest_conversions[-5:]

    tbr_result = tbr.analyze(
        pretest_train_conversions=pretest_train_conversions,
        pretest_val_conversions=pretest_val_conversions,
        test_conversions=test_conversions,
        treatment_mask=treatment_mask,
        placebo_masks=placebo_masks,
        alpha=0.1,
        experiment_type=api.ExperimentType.HOLDBACK,
        treatment_cell_id=2.0,
        pretest_spend=pretest_spend,
        test_spend=test_spend,
    )

    # G1 is only 1 treated geo for cell 2:
    # 1 geo * 10 lift/day * 5 days = 50.0 total lift
    self.assertAlmostEqual(tbr_result.lift.point_estimate, 50.0, delta=1.0)
    self.assertLess(tbr_result.lift.p_value, 0.5)
    self.assertIsNotNone(tbr_result.icpd)
    icpd = tbr_result.icpd
    assert icpd is not None
    self.assertAlmostEqual(icpd.point_estimate, 1.0, delta=0.01)

  def test_compute_studentized_p_value(self):
    p_val = methodology_util.compute_studentized_p_value(
        estimate=10.0,
        rmse=1.0,
        t_placebo=jnp.array([0.0, 5.0, 10.0, 15.0]),
        test_type=api.TestType.TWO_SIDED,
    )
    # n_extreme_left = sum([0, 5, 10, 15] <= 10.0) = 3
    # n_extreme_right = sum([0, 5, 10, 15] >= 10.0) = 2
    # n_extreme = 2 * min(3, 2) = 4
    # p_val = (1 + 4) / (1 + 4) = 1.0
    self.assertAlmostEqual(p_val, 1.0)

  def test_compute_se(self):
    rmse = 2.0
    t_placebo = jnp.array([-1.0, 0.0, 1.0])
    se = methodology_util.compute_se(rmse, t_placebo)
    # std([-1, 0, 1]) = sqrt(2/3)
    expected_se = jnp.std(t_placebo) * rmse
    self.assertAlmostEqual(se, expected_se)

  def test_compute_cis_two_sided(self):
    estimate = 100.0
    rmse = 10.0
    t_placebo = jnp.array([-1.0, 0.0, 1.0])
    alpha = 0.1
    lower, upper = methodology_util.compute_cis(
        estimate, rmse, t_placebo, alpha, api.TestType.TWO_SIDED
    )
    # quantile(t_placebo, 1-0.05) = quantile([-1, 0, 1], 0.95)
    # quantile(t_placebo, 0.05) = quantile([-1, 0, 1], 0.05)
    expected_lower = estimate - jnp.quantile(t_placebo, 0.95) * rmse
    expected_upper = estimate - jnp.quantile(t_placebo, 0.05) * rmse
    self.assertAlmostEqual(lower, expected_lower)
    self.assertAlmostEqual(upper, expected_upper)

  def test_compute_cis_one_sided(self):
    estimate = 100.0
    rmse = 10.0
    t_placebo = jnp.array([-1.0, 0.0, 1.0])
    alpha = 0.1
    lower, upper = methodology_util.compute_cis(
        estimate, rmse, t_placebo, alpha, api.TestType.ONE_SIDED
    )
    expected_lower = estimate - jnp.quantile(t_placebo, 0.9) * rmse
    expected_upper = jnp.inf
    self.assertAlmostEqual(lower, expected_lower)
    self.assertEqual(upper, expected_upper)

  @parameterized.named_parameters(
      dict(
          testcase_name='holdback_two_sided',
          experiment_type=api.ExperimentType.HOLDBACK,
          test_type=api.TestType.TWO_SIDED,
          expected_estimate=0.1,  # (1.1 / 1.0) - 1
      ),
      dict(
          testcase_name='holdback_one_sided',
          experiment_type=api.ExperimentType.HOLDBACK,
          test_type=api.TestType.ONE_SIDED,
          expected_estimate=0.1,
      ),
      dict(
          testcase_name='go_dark_two_sided',
          experiment_type=api.ExperimentType.GO_DARK,
          test_type=api.TestType.TWO_SIDED,
          expected_estimate=-(0.9 / 1.0 - 1.0),  # -(0.9 - 1) = 0.1
      ),
      dict(
          testcase_name='go_dark_one_sided',
          experiment_type=api.ExperimentType.GO_DARK,
          test_type=api.TestType.ONE_SIDED,
          expected_estimate=0.1,
      ),
  )
  def test_get_percent_lift(
      self, experiment_type, test_type, expected_estimate
  ):
    # Setup values.
    # For HOLDBACK: y_test = 1.1, y_pred = 1.0. log(1.1/1.0) = ~0.0953.
    # For GO_DARK: y_test = 0.9, y_pred = 1.0. log(0.9/1.0) = ~-0.1053.
    if experiment_type == api.ExperimentType.HOLDBACK:
      y_test = 1.1
    else:
      y_test = 0.9
    y_pred = jnp.array(1.0)
    log_rmse = jnp.array(0.1)
    y_test_placebos = jnp.array([1.0, 1.05, 0.95])
    y_pred_placebos = jnp.array([1.0, 1.0, 1.0])
    placebo_log_rmses = jnp.array([0.1, 0.1, 0.1])
    alpha = 0.1

    estimate = methodology_util.get_percent_lift(
        y_test=jnp.array(y_test),
        y_pred=y_pred,
        log_rmse=log_rmse,
        y_test_placebos=y_test_placebos,
        y_pred_placebos=y_pred_placebos,
        placebo_log_rmses=placebo_log_rmses,
        alpha=alpha,
        experiment_type=experiment_type,
        test_type=test_type,
    )

    self.assertIsInstance(estimate, api.Estimate)
    self.assertIsInstance(estimate.point_estimate, float)
    self.assertAlmostEqual(
        estimate.point_estimate, expected_estimate, delta=1e-5
    )
    self.assertIsInstance(estimate.lower_bound, float)
    self.assertIsInstance(estimate.upper_bound, float)
    self.assertIsInstance(estimate.standard_deviation, float)
    self.assertIsInstance(estimate.p_value, float)

  def test_compute_placebo_effect_from_mask(self):
    # Setup simple data where linear fit is perfect
    # Geos: 0 treated (original), 1 placebo treated, 2 control
    data_pretest = jnp.array([
        [10.0, 10.0, 10.0],
        [20.0, 20.0, 20.0],
        [30.0, 30.0, 30.0],
    ])
    data_test = jnp.array([
        [40.0, 40.0, 40.0],
    ])
    mask = jnp.array([1, 0, 0])
    p_mask = jnp.array([0, 1, 0])

    data_pretest_train = data_pretest[:-1]
    data_pretest_val = data_pretest[-1:]

    (
        y_test,
        y_pred,
        rmse,
        log_rmse,
    ) = tbr._compute_placebo_effect_from_mask(
        data_pretest,
        data_pretest_train,
        data_pretest_val,
        data_test,
        mask,
        p_mask,
        1.0,
    )
    effect = y_test - y_pred
    # p_mask=1 selects G1. valid_control_mask selects G2.
    # pre T0+T1: py = [10, 20], px = [10, 20]. alpha=0, beta=1.
    # pre T2: py_val = [30], px_val = [30]. py_pred_val = 30. rmse = 0.
    # test: py_test = [40], px_test = [40].
    # py_pred = 0 + 1 * 40 = 40.
    # effect = 40 - 40 = 0.
    self.assertAlmostEqual(float(rmse), 0.0, delta=1e-4)
    self.assertAlmostEqual(float(effect[0]), 0.0, delta=1e-4)
    self.assertAlmostEqual(float(log_rmse), 0.0, delta=1e-4)

  def test_compute_placebo_effect_from_mask_multicell(self):
    # Setup simple data with 5 geos
    data_pretest = jnp.array([
        [10.0, 10.0, 10.0, 10.0, 10.0],
        [20.0, 20.0, 20.0, 20.0, 20.0],
        [30.0, 30.0, 30.0, 30.0, 30.0],
    ])
    data_test = jnp.array([
        [30.0, 30.0, 50.0, 40.0, 30.0],
    ])
    # G0 is cell 1, G1 is cell 2, G2-G4 are control
    mask = jnp.array([1, 2, 0, 0, 0])
    # G2 is placebo cell 2, G3 is placebo cell 1, G4 is control
    # Note that non-zero placebo mask entries are strictly restricted to
    # indices originally assigned to control (where mask == 0).
    p_mask = jnp.array([0, 0, 2, 1, 0])

    data_pretest_train = data_pretest[:-1]
    data_pretest_val = data_pretest[-1:]

    (
        y_test,
        y_pred,
        rmse,
        log_rmse,
    ) = tbr._compute_placebo_effect_from_mask(
        data_pretest,
        data_pretest_train,
        data_pretest_val,
        data_test,
        mask,
        p_mask,
        2.0,
    )
    effect = y_test - y_pred
    self.assertAlmostEqual(float(rmse), 0.0, delta=1e-4)
    self.assertAlmostEqual(float(y_test[0]), 50.0, delta=1e-4)
    self.assertAlmostEqual(float(y_pred[0]), 30.0, delta=1e-4)
    self.assertAlmostEqual(float(effect[0]), 20.0, delta=1e-4)
    self.assertAlmostEqual(float(log_rmse), 0.0, delta=1e-4)


if __name__ == '__main__':
  absltest.main()
