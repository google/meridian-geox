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
import jax.numpy as jnp
from meridian_geox import api
from meridian_geox.methodology import base
import numpy as np


class _MeanDiffMethodology(base.CounterfactualMethodology):
  """Minimal Difference-in-Differences (slope=1) methodology for base tests."""

  def _fit_and_predict_single(
      self,
      data_train: jnp.ndarray,
      data_eval: jnp.ndarray,
      treatment_group_mask: jnp.ndarray,
      control_group_mask: jnp.ndarray,
  ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    y_train = base.compute_binary_group_mean(data_train, treatment_group_mask)
    x_train = base.compute_binary_group_mean(data_train, control_group_mask)
    intercept = jnp.mean(y_train - x_train)
    y_pred_train = x_train + intercept

    y_eval = base.compute_binary_group_mean(data_eval, treatment_group_mask)
    x_eval = base.compute_binary_group_mean(data_eval, control_group_mask)
    y_pred_eval = x_eval + intercept
    return y_train, y_pred_train, y_eval, y_pred_eval


class BaseMethodologyTest(parameterized.TestCase):

  def test_compute_group_mean(self):
    data = jnp.array([[10.0, 20.0, 30.0, 40.0], [15.0, 25.0, 35.0, 45.0]])
    mask = jnp.array([1.0, 0.0, 1.0, 0.0])

    y_mean = base.compute_group_mean(data, mask, 1.0)
    x_mean = base.compute_group_mean(data, mask, 0.0)

    self.assertEqual(y_mean.shape, (2,))
    self.assertEqual(x_mean.shape, (2,))
    np.testing.assert_allclose(
        y_mean, jnp.array([20.0, 25.0]), rtol=5e-5, atol=5e-5
    )
    np.testing.assert_allclose(
        x_mean, jnp.array([30.0, 35.0]), rtol=5e-5, atol=5e-5
    )

  def test_compute_group_mean_empty_mask_returns_zeros(self):
    data = jnp.array([[10.0, 20.0], [15.0, 25.0], [30.0, 40.0]])
    mask = jnp.array([0.0, 0.0])

    empty_cell_mean = base.compute_group_mean(data, mask, 1.0)
    empty_binary_mean = base.compute_binary_group_mean(data, mask)

    self.assertEqual(empty_cell_mean.shape, (3,))
    self.assertEqual(empty_binary_mean.shape, (3,))
    np.testing.assert_allclose(
        empty_cell_mean, jnp.zeros((3,)), rtol=5e-5, atol=5e-5
    )
    np.testing.assert_allclose(
        empty_binary_mean, jnp.zeros((3,)), rtol=5e-5, atol=5e-5
    )

  def test_check_candidate_compatibility_default_true(self):
    methodology = _MeanDiffMethodology()
    candidates = jnp.array([[1.0, 0.0], [0.0, 1.0]])
    data = jnp.ones((5, 2))
    design_config = api.DesignConfig(
        experiment_duration=datetime.timedelta(days=2),
    )
    compat = methodology.check_candidate_compatibility(
        candidates=candidates,
        conversion_data=data,
        spend_data=data,
        design_config=design_config,
        cell_id=1.0,
    )
    self.assertEqual(compat.shape, (2,))
    np.testing.assert_array_equal(compat, jnp.array([True, True]))

  @parameterized.named_parameters(
      dict(
          testcase_name='two_sided',
          test_type=api.TestType.TWO_SIDED,
          expected_p_values=[1.0, 0.04550026],
      ),
      dict(
          testcase_name='one_sided',
          test_type=api.TestType.ONE_SIDED,
          expected_p_values=[0.5, 0.02275013],
      ),
  )
  def test_get_r2_and_get_mde_via_base_template(
      self, test_type, expected_p_values
  ):
    methodology = _MeanDiffMethodology()
    # 4 geos:
    # - Geo 0: control + 5.0 in pre, control + 5.0 in val (0 residual)
    # - Geo 1: control + 5.0 in pre, control + 5.0 + [2.0, 2.0] in val (+2 lift)
    # - Geo 2, 3: control geos ([10, 20, 30, 40] in pre, [50, 70] in val)
    data_pre = jnp.array([
        [15.0, 15.0, 10.0, 10.0],
        [25.0, 25.0, 20.0, 20.0],
        [35.0, 35.0, 30.0, 30.0],
        [45.0, 45.0, 40.0, 40.0],
    ])
    data_val = jnp.array([
        [55.0, 57.0, 50.0, 50.0],
        [75.0, 77.0, 70.0, 70.0],
    ])
    masks = jnp.array([
        [1.0, 2.0, 0.0, 0.0],
        [2.0, 1.0, 0.0, 0.0],
    ])

    r2 = methodology.get_r2(data_pre, data_val, masks, jnp.array([1.0]))
    # Candidate 0: mse = 0, var_y = 100 -> r2 = 1.0
    # Candidate 1: residuals = [2, 2], mse = 4, var_y = 100 -> r2 = 0.96
    self.assertEqual(r2.shape, (2, 1))
    np.testing.assert_allclose(
        r2, jnp.array([[1.0], [0.96]]), rtol=5e-5, atol=5e-5
    )

    mde_results = methodology.get_mde(
        data_pre=data_pre,
        data_val=data_val,
        treatment_masks=masks,
        z_score_sum=1.96,
        test_type=test_type,
    )
    self.assertEqual(mde_results.mde_abs.shape, (2, 1))
    self.assertEqual(mde_results.mde_pct.shape, (2, 1))
    self.assertEqual(mde_results.p_value.shape, (2, 1))
    self.assertEqual(mde_results.observed_conversions.shape, (2, 1, 6))
    self.assertEqual(mde_results.counterfactual_conversions.shape, (2, 1, 6))
    # Candidate effects: [0.0, 2.0] -> se = 1.0 -> mde_abs = 1.96
    # Candidate baselines: [65.0, 67.0] -> mde_pct = [1.96 / 65, 1.96 / 67]
    np.testing.assert_allclose(
        mde_results.mde_abs,
        jnp.array([[1.96], [1.96]]),
        rtol=5e-5,
        atol=5e-5,
    )
    np.testing.assert_allclose(
        mde_results.mde_pct,
        jnp.array([[1.96 / 65.0], [1.96 / 67.0]]),
        rtol=5e-5,
        atol=5e-5,
    )
    np.testing.assert_allclose(
        mde_results.p_value,
        jnp.array([[expected_p_values[0]], [expected_p_values[1]]]),
        rtol=5e-5,
        atol=5e-5,
    )
    np.testing.assert_allclose(
        mde_results.observed_conversions,
        jnp.array([
            [[15.0, 25.0, 35.0, 45.0, 55.0, 75.0]],
            [[15.0, 25.0, 35.0, 45.0, 57.0, 77.0]],
        ]),
        rtol=5e-5,
        atol=5e-5,
    )
    np.testing.assert_allclose(
        mde_results.counterfactual_conversions,
        jnp.array([
            [[15.0, 25.0, 35.0, 45.0, 55.0, 75.0]],
            [[15.0, 25.0, 35.0, 45.0, 55.0, 75.0]],
        ]),
        rtol=5e-5,
        atol=5e-5,
    )

  def test_get_r2_and_get_mde_multicell_and_cell_total_scaling(self):
    methodology = _MeanDiffMethodology()
    # 5 geos across 4 pre periods and 2 val periods:
    # - Geos 0, 1: control + 5.0 in pre, control + 5.0 in val (0 residual)
    # - Geo 2: control + 5.0 in pre, control + 7.0 in val (+2 lift)
    # - Geo 3: control + 10.0 in pre, control + 10.0 in val (0 residual)
    # - Geo 4: control ([10, 20, 30, 40] in pre, [50, 70] in val)
    data_pre = jnp.array([
        [15.0, 15.0, 15.0, 20.0, 10.0],
        [25.0, 25.0, 25.0, 30.0, 20.0],
        [35.0, 35.0, 35.0, 40.0, 30.0],
        [45.0, 45.0, 45.0, 50.0, 40.0],
    ])
    data_val = jnp.array([
        [55.0, 55.0, 57.0, 60.0, 50.0],
        [75.0, 75.0, 77.0, 80.0, 70.0],
    ])
    # 3 designs x 2 cells (N_designs=3 != k_cells=2), with n_treated=2 in cell 1
    masks = jnp.array([
        [1.0, 1.0, 0.0, 2.0, 0.0],
        [1.0, 0.0, 1.0, 2.0, 0.0],
        [0.0, 1.0, 1.0, 2.0, 0.0],
    ])
    cell_ids = jnp.array([1.0, 2.0])

    r2 = methodology.get_r2(data_pre, data_val, masks, cell_ids)
    self.assertEqual(r2.shape, (3, 2))
    # Candidate 0: Control {2,4} has val +1 -> res=-1, mse=1, var=100 -> r2=0.99
    # Candidate 1: Cell 1 {0,2} has val +1 -> r2=0.99; Cell 2 {3} -> r2=1.0
    np.testing.assert_allclose(
        r2,
        jnp.array([[0.99, 0.99], [0.99, 1.0], [0.99, 1.0]]),
        rtol=5e-5,
        atol=5e-5,
    )

    mde_results = methodology.get_mde(
        data_pre=data_pre,
        data_val=data_val,
        treatment_masks=masks,
        z_score_sum=1.96,
        cell_ids=cell_ids,
    )
    self.assertEqual(mde_results.mde_abs.shape, (3, 2))
    self.assertEqual(mde_results.mde_pct.shape, (3, 2))
    self.assertEqual(mde_results.p_value.shape, (3, 2))
    self.assertEqual(mde_results.observed_conversions.shape, (3, 2, 6))
    self.assertEqual(mde_results.counterfactual_conversions.shape, (3, 2, 6))

    # Cell 1: effects = [-1, 1, 1] -> se = sqrt(8/9), baselines = [65, 66, 66]
    # Cell 2: effects = [-1, 0, 0] -> se = sqrt(2/9), baselines = [70, 70, 70]
    se_c1 = jnp.sqrt(8.0 / 9.0)
    se_c2 = jnp.sqrt(2.0 / 9.0)
    expected_mde_abs_c1 = 1.96 * se_c1
    expected_mde_abs_c2 = 1.96 * se_c2
    expected_mde_abs = jnp.array([
        [expected_mde_abs_c1, expected_mde_abs_c2],
        [expected_mde_abs_c1, expected_mde_abs_c2],
        [expected_mde_abs_c1, expected_mde_abs_c2],
    ])
    expected_baselines = jnp.array([
        [65.0, 70.0],
        [66.0, 70.0],
        [66.0, 70.0],
    ])
    # Two-sided normal p-values for z = [[-1/se_c1, -1/se_c2], [1/se_c1, 0],
    # [1/se_c1, 0]] = [[-3/(2*sqrt(2)), -3/sqrt(2)], [3/(2*sqrt(2)), 0], ...].
    expected_p_value = jnp.array([
        [0.28884437, 0.03389485],
        [0.28884437, 1.0],
        [0.28884437, 1.0],
    ])
    np.testing.assert_allclose(
        mde_results.mde_abs, expected_mde_abs, rtol=5e-5, atol=5e-5
    )
    np.testing.assert_allclose(
        mde_results.mde_pct,
        expected_mde_abs / expected_baselines,
        rtol=5e-5,
        atol=5e-5,
    )
    np.testing.assert_allclose(
        mde_results.p_value, expected_p_value, rtol=5e-5, atol=5e-5
    )

    # Candidate 0, Cell 1 has n_treated=2 geos (Geos 0 and 1), so observed
    # conversions must equal 2 * [15, 25, 35, 45, 55, 75] and counterfactual
    # conversions must equal 2 * [15, 25, 35, 45, 56, 76].
    np.testing.assert_allclose(
        mde_results.observed_conversions[0, 0],
        jnp.array([30.0, 50.0, 70.0, 90.0, 110.0, 150.0]),
        rtol=5e-5,
        atol=5e-5,
    )
    np.testing.assert_allclose(
        mde_results.counterfactual_conversions[0, 0],
        jnp.array([30.0, 50.0, 70.0, 90.0, 112.0, 152.0]),
        rtol=5e-5,
        atol=5e-5,
    )
    # Candidate 0, Cell 2 has n_treated=1 geo (Geo 3), so observed conversions
    # equal 1 * [20, 30, 40, 50, 60, 80].
    np.testing.assert_allclose(
        mde_results.observed_conversions[0, 1],
        jnp.array([20.0, 30.0, 40.0, 50.0, 60.0, 80.0]),
        rtol=5e-5,
        atol=5e-5,
    )
    np.testing.assert_allclose(
        mde_results.counterfactual_conversions[0, 1],
        jnp.array([20.0, 30.0, 40.0, 50.0, 61.0, 81.0]),
        rtol=5e-5,
        atol=5e-5,
    )

  def test_get_r2_invalid_masks_and_zero_variance_guards(self):
    methodology = _MeanDiffMethodology()
    # Constant pre-period and validation-period data -> mse == 0.0 and
    # var_y == 0.0, exercising the 0.0 / 0.0 NaN guard (safe_var_y).
    data_pre = jnp.full((2, 3), 50.0)
    data_val_const = jnp.full((2, 3), 50.0)
    masks = jnp.array([
        [1.0, 0.0, 0.0],  # valid mask, but var_y == 0.0 -> r2 == 0.0
        [1.0, 1.0, 1.0],  # 0 control geos -> r2 == nan
        [0.0, 0.0, 0.0],  # 0 treated geos in cell 1 -> r2 == nan
    ])

    r2 = methodology.get_r2(data_pre, data_val_const, masks, jnp.array([1.0]))
    self.assertEqual(r2.shape, (3, 1))
    np.testing.assert_allclose(r2[0, 0], 0.0, rtol=5e-5, atol=5e-5)
    self.assertTrue(bool(jnp.isnan(r2[1, 0])))
    self.assertTrue(bool(jnp.isnan(r2[2, 0])))

  def test_get_mde_zero_se_and_zero_baseline_guards(self):
    methodology = _MeanDiffMethodology()
    data_pre = jnp.zeros((3, 3))
    data_val = jnp.zeros((2, 3))
    masks = jnp.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ])

    mde_results = methodology.get_mde(
        data_pre=data_pre,
        data_val=data_val,
        treatment_masks=masks,
        z_score_sum=1.96,
    )
    self.assertEqual(mde_results.mde_abs.shape, (2, 1))
    self.assertEqual(mde_results.mde_pct.shape, (2, 1))
    self.assertEqual(mde_results.p_value.shape, (2, 1))
    np.testing.assert_allclose(
        mde_results.mde_abs, jnp.zeros((2, 1)), rtol=5e-5, atol=5e-5
    )
    self.assertTrue(bool(jnp.all(jnp.isnan(mde_results.mde_pct))))
    np.testing.assert_allclose(
        mde_results.p_value, jnp.ones((2, 1)), rtol=5e-5, atol=5e-5
    )


if __name__ == '__main__':
  absltest.main()
