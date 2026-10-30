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
from jax.scipy import stats
from meridian_geox import api
from meridian_geox.methodology import base
from meridian_geox.methodology import util as methodology_util
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
    expected_z_scores = jnp.array([
        [-1.0 / se_c1, -1.0 / se_c2],
        [1.0 / se_c1, 0.0],
        [1.0 / se_c1, 0.0],
    ])
    expected_p_value = 2.0 * (1.0 - stats.norm.cdf(jnp.abs(expected_z_scores)))
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
    self.assertAlmostEqual(float(r2[0, 0]), 0.0, places=5)
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

  def test_compute_placebo_effect_from_mask_excludes_original_treated_geos(
      self,
  ):
    methodology = _MeanDiffMethodology()
    # 5 geos:
    # - Geo 0: originally treated (mask == 1.0), with huge offset +1000 so if it
    #   leaks into valid_control_mask the prediction will be wildly wrong.
    # - Geo 1: placebo treated for cell 2 (p_mask == 2.0), control + 5 in train,
    #   [0.0, 164.0] in val (so scale_val = 82.0 clips py_val[0] = 0.0 to
    #   0.082 in compute_regularized_log_ratio, and combined 4-row pretest
    #   intercept is (5 + 5 - 70 + 84) / 4 = 6.0 != train intercept 5.0), and
    #   [106.0, 126.0] in test.
    # - Geos 2, 3: valid controls (mask == 0.0 and p_mask == 0.0) with base 50.
    # - Geo 4: untreated other cell 3 (mask == 3.0, p_mask == 3.0) with huge
    #   offset +2000 so mutating p_mask == treatment_cell_id (2.0) to >= fails.
    data_pretest_train = jnp.array([
        [1050.0, 55.0, 50.0, 50.0, 2050.0],
        [1060.0, 65.0, 60.0, 60.0, 2060.0],
    ])
    data_pretest_val = jnp.array([
        [1070.0, 0.0, 70.0, 70.0, 2070.0],
        [1080.0, 164.0, 80.0, 80.0, 2080.0],
    ])
    data_pretest = jnp.concatenate(
        [data_pretest_train, data_pretest_val], axis=0
    )
    data_test = jnp.array([
        [1090.0, 106.0, 90.0, 90.0, 2090.0],
        [1100.0, 126.0, 100.0, 100.0, 2100.0],
    ])
    mask = jnp.array([1.0, 0.0, 0.0, 0.0, 3.0])
    p_mask = jnp.array([0.0, 2.0, 0.0, 0.0, 3.0])

    py_test, py_pred, p_rmse, p_log_rmse = (
        methodology._compute_placebo_effect_from_mask(
            data_pretest=data_pretest,
            data_pretest_train=data_pretest_train,
            data_pretest_val=data_pretest_val,
            data_test=data_test,
            mask=mask,
            p_mask=p_mask,
            treatment_cell_id=2.0,
        )
    )
    # Train intercept = 5.0 -> py_pred_val = [75, 85], py_val = [0, 164] ->
    # p_rmse = sqrt(((-75)^2 + 79^2) / 2) = sqrt(5933.0).
    # Combined pretest intercept = mean([5, 5, -70, 84]) = 6.0 ->
    # py_pred = [96, 106], py_test = [106, 126].
    np.testing.assert_allclose(
        py_test, jnp.array([106.0, 126.0]), rtol=5e-5, atol=5e-5
    )
    np.testing.assert_allclose(
        py_pred, jnp.array([96.0, 106.0]), rtol=5e-5, atol=5e-5
    )
    self.assertAlmostEqual(float(p_rmse), float(jnp.sqrt(5933.0)), places=4)
    expected_log_err = methodology_util.compute_regularized_log_ratio(
        jnp.array([0.0, 164.0]),
        jnp.array([75.0, 85.0]),
        scale=82.0,
    )
    expected_p_log_rmse = jnp.sqrt(jnp.mean(jnp.square(expected_log_err)))
    self.assertAlmostEqual(
        float(p_log_rmse), float(expected_p_log_rmse), places=5
    )

  @parameterized.named_parameters(
      dict(testcase_name='positive_spend', incremental_spend_sign=1.0),
      dict(testcase_name='go_dark_spend', incremental_spend_sign=-1.0),
  )
  def test_compute_icpd_and_pytree_registration(self, incremental_spend_sign):
    methodology = _MeanDiffMethodology()
    # 5 geos: Geo 0 in cell 1 (500.0), Geos 1, 2 in cell 2 (n_treatment_geos=2),
    # Geo 3 is control (0.0), Geo 4 in cell 3 (1000.0).
    # Pretest spend: treated = control + 10.
    # Test spend (3 periods): control = [20, 30, 40], pred = [30, 40, 50],
    # treated = pred + sign * [2, 6, 5] -> signed total incremental spend =
    # 2 * [2, 6, 5] = [4, 12, 10] -> cumulative_incremental_spend = [4, 16, 26],
    # while counterfactual_spend = 2 * [30, 40, 50] = [60, 80, 100] always.
    pretest_spend = jnp.array([
        [500.0, 20.0, 20.0, 10.0, 1000.0],
        [500.0, 30.0, 30.0, 20.0, 1000.0],
    ])
    spend_ctrl_test = jnp.array([20.0, 30.0, 40.0])
    spend_treat_test = jnp.array(
        [30.0, 40.0, 50.0]
    ) + incremental_spend_sign * jnp.array([2.0, 6.0, 5.0])
    test_spend = jnp.stack(
        [
            jnp.full((3,), 500.0),
            spend_treat_test,
            spend_treat_test,
            spend_ctrl_test,
            jnp.full((3,), 1000.0),
        ],
        axis=1,
    )
    treatment_mask = jnp.array([1.0, 2.0, 2.0, 0.0, 3.0])
    total_cumul_effect = jnp.array([20.0, 48.0, 104.0])
    total_lower_cis = jnp.array([12.0, 32.0, 78.0])
    total_upper_cis = jnp.array([28.0, 64.0, 130.0])

    icpd_res = methodology._compute_icpd(
        pretest_spend=pretest_spend,
        test_spend=test_spend,
        treatment_mask=treatment_mask,
        total_cumul_effect=total_cumul_effect,
        total_lower_cis=total_lower_cis,
        total_upper_cis=total_upper_cis,
        incremental_spend_sign=incremental_spend_sign,
        treatment_cell_id=2.0,
    )
    # Verify all 5 fields of IcpdResults in PyTree round-trip.
    doubled = jax.tree_util.tree_map(lambda x: 2.0 * x, icpd_res)
    np.testing.assert_allclose(
        doubled.cumulative_icpd, 2.0 * icpd_res.cumulative_icpd
    )
    np.testing.assert_allclose(doubled.lower_bound, 2.0 * icpd_res.lower_bound)
    np.testing.assert_allclose(doubled.upper_bound, 2.0 * icpd_res.upper_bound)
    np.testing.assert_allclose(
        doubled.cumulative_incremental_spend,
        2.0 * icpd_res.cumulative_incremental_spend,
    )
    np.testing.assert_allclose(
        doubled.counterfactual_spend, 2.0 * icpd_res.counterfactual_spend
    )
    np.testing.assert_allclose(
        icpd_res.cumulative_incremental_spend,
        jnp.array([4.0, 16.0, 26.0]),
        rtol=5e-5,
        atol=5e-5,
    )
    np.testing.assert_allclose(
        icpd_res.counterfactual_spend,
        jnp.array([60.0, 80.0, 100.0]),
        rtol=5e-5,
        atol=5e-5,
    )
    np.testing.assert_allclose(
        icpd_res.cumulative_icpd,
        jnp.array([5.0, 3.0, 4.0]),
        rtol=5e-5,
        atol=5e-5,
    )
    np.testing.assert_allclose(
        icpd_res.lower_bound,
        jnp.array([3.0, 2.0, 3.0]),
        rtol=5e-5,
        atol=5e-5,
    )
    np.testing.assert_allclose(
        icpd_res.upper_bound,
        jnp.array([7.0, 4.0, 5.0]),
        rtol=5e-5,
        atol=5e-5,
    )

  @parameterized.named_parameters(
      dict(
          testcase_name='holdback_two_sided',
          experiment_type=api.ExperimentType.HOLDBACK,
          test_type=api.TestType.TWO_SIDED,
          sign=1.0,
      ),
      dict(
          testcase_name='holdback_one_sided',
          experiment_type=api.ExperimentType.HOLDBACK,
          test_type=api.TestType.ONE_SIDED,
          sign=1.0,
      ),
      dict(
          testcase_name='go_dark_two_sided',
          experiment_type=api.ExperimentType.GO_DARK,
          test_type=api.TestType.TWO_SIDED,
          sign=-1.0,
      ),
      dict(
          testcase_name='go_dark_one_sided',
          experiment_type=api.ExperimentType.GO_DARK,
          test_type=api.TestType.ONE_SIDED,
          sign=-1.0,
      ),
      dict(
          testcase_name='heavy_up_two_sided',
          experiment_type=api.ExperimentType.HEAVY_UP,
          test_type=api.TestType.TWO_SIDED,
          sign=1.0,
      ),
      dict(
          testcase_name='heavy_up_one_sided',
          experiment_type=api.ExperimentType.HEAVY_UP,
          test_type=api.TestType.ONE_SIDED,
          sign=1.0,
      ),
  )
  def test_analyze_via_base_template(self, experiment_type, test_type, sign):
    methodology = _MeanDiffMethodology()
    # 9 geos with target treatment_cell_id = 2.0:
    # - Geo 0: cell 1.0 (untreated cell < 2.0, value 1000.0)
    # - Geos 1, 2: cell 2.0 (treated group, n_treatment_geos = 2.0)
    # - Geos 3, 4, 5: controls (0.0) that serve as placebo-treated in
    #   placebo_masks[0..2]
    # - Geo 6: clean control (0.0 in treatment_mask and all placebo_masks),
    #   equal to base = [50, 60] in train, [60, 80] in val, [90, 100, 110] in
    #   test (3 test periods so index [1] != index [-1])
    # - Geo 7: balancing control (0.0 in treatment_mask, 1.0 in all
    #   placebo_masks) whose offset over base is -(offset_3 + offset_4 +
    #   offset_5), ensuring the 5-geo real control mean is identically base
    #   while each placebo's valid control (Geo 6) is also identically base.
    # - Geo 8: cell 3.0 (untreated cell > 2.0, value 2000.0)
    pretest_train = jnp.array([
        [1000.0, 60.0, 60.0, 50.0, 50.0, 50.0, 50.0, 50.0, 2000.0],
        [1000.0, 70.0, 70.0, 60.0, 60.0, 60.0, 60.0, 60.0, 2000.0],
    ])
    # In pretest_val (base = [60, 80]):
    # - Treated Geos 1, 2 have [0, 100] (train pred = [70, 90], val residuals =
    #   [-70, +10] -> rmse = sqrt((4900 + 100) / 2) = 50.0, scale_val = 50.0
    #   actively clips py_val[0] = 0.0 to 0.05 in compute_regularized_log_ratio,
    #   and combined 4-row pretest offset mean = (10 + 10 - 60 + 20) / 4 = -5.0
    #   != train intercept +10.0).
    # - Placebo Geos 3, 4, 5 have offsets [+/-2], [+/-3], [+/-5] -> p_rmses =
    #   [2.0, 3.0, 5.0]; Geo 7 has offset [-10, +10] so control sum = 0.
    pretest_val = jnp.array([
        [1000.0, 0.0, 0.0, 62.0, 63.0, 65.0, 60.0, 50.0, 2000.0],
        [1000.0, 100.0, 100.0, 78.0, 77.0, 75.0, 80.0, 90.0, 2000.0],
    ])
    # In test (3 periods, base = [90, 100, 110], treated pred = [85, 95, 105]):
    # - Treated Geos 1, 2 have pred + sign * [75, 125, 100] ->
    #   signed_geo_avg_pointwise_effect = [75, 125, 100],
    #   signed_total_pointwise_effect = [150, 250, 200],
    #   signed_geo_avg_cumul_effect = [75, 200, 300],
    #   signed_total_cumul_effect = [150, 400, 600],
    #   final t_obs = 300.0 / 50.0 = 6.0.
    # - Placebo 0 (Geo 3, p_rmse=2): offset = sign * [-4, -3, -5] ->
    #   signed_t_placebo_pw[0] = [-2.0, -1.5, -2.5],
    #   signed_t_placebo[0] = [-2.0, -3.5, -6.0].
    # - Placebo 1 (Geo 4, p_rmse=3): offset = sign * [1.5, 4.5, 6.0] ->
    #   signed_t_placebo_pw[1] = [0.5, 1.5, 2.0],
    #   signed_t_placebo[1] = [0.5, 2.0, 4.0].
    # - Placebo 2 (Geo 5, p_rmse=5): offset = sign * [10.0, 15.0, 25.0] ->
    #   signed_t_placebo_pw[2] = [2.0, 3.0, 5.0],
    #   signed_t_placebo[2] = [2.0, 5.0, 10.0].
    # - Geo 7 offset = -sign * [7.5, 16.5, 26.0] so real control mean is base.
    base_test = jnp.array([90.0, 100.0, 110.0])
    treat_test = jnp.array([85.0, 95.0, 105.0]) + sign * jnp.array(
        [75.0, 125.0, 100.0]
    )
    p0_test = base_test + sign * jnp.array([-4.0, -3.0, -5.0])
    p1_test = base_test + sign * jnp.array([1.5, 4.5, 6.0])
    p2_test = base_test + sign * jnp.array([10.0, 15.0, 25.0])
    bal_test = base_test - sign * jnp.array([7.5, 16.5, 26.0])
    c1_col = jnp.full((3,), 1000.0)
    c3_col = jnp.full((3,), 2000.0)
    test_conversions = jnp.stack(
        [
            c1_col,
            treat_test,
            treat_test,
            p0_test,
            p1_test,
            p2_test,
            base_test,
            bal_test,
            c3_col,
        ],
        axis=1,
    )
    treatment_mask = jnp.array([1.0, 2.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 3.0])
    placebo_masks = jnp.array([
        [1.0, 0.0, 0.0, 2.0, 1.0, 3.0, 0.0, 1.0, 3.0],
        [1.0, 0.0, 0.0, 1.0, 2.0, 3.0, 0.0, 1.0, 3.0],
        [1.0, 0.0, 0.0, 1.0, 3.0, 2.0, 0.0, 1.0, 3.0],
    ])

    # Spend series (4 pretest rows, 3 test rows):
    # Control spend = [10, 20, 30, 40] in pretest, [50, 60, 70] in test.
    # Treated spend = control + 5 in pretest ([15, 25, 35, 45]), and
    # control + 5 + sign * [2, 6, 5] in test -> y_spend_pred = [55, 65, 75],
    # counterfactual_spend = 2 * [55, 65, 75] = [110, 130, 150],
    # incremental_spend = 2 * [2, 6, 5] = [4, 12, 10],
    # cumulative_incremental_spend = [4, 16, 26],
    # cumulative_icpd = [150/4, 400/16, 600/26] = [37.5, 25.0, 600.0/26.0].
    pretest_spend = jnp.array([
        [500.0, 15.0, 15.0, 10.0, 10.0, 10.0, 10.0, 10.0, 1000.0],
        [500.0, 25.0, 25.0, 20.0, 20.0, 20.0, 20.0, 20.0, 1000.0],
        [500.0, 35.0, 35.0, 30.0, 30.0, 30.0, 30.0, 30.0, 1000.0],
        [500.0, 45.0, 45.0, 40.0, 40.0, 40.0, 40.0, 40.0, 1000.0],
    ])
    spend_treat_test = jnp.array([55.0, 65.0, 75.0]) + sign * jnp.array(
        [2.0, 6.0, 5.0]
    )
    spend_ctrl_test = jnp.array([50.0, 60.0, 70.0])
    test_spend = jnp.stack(
        [
            jnp.full((3,), 500.0),
            spend_treat_test,
            spend_treat_test,
            spend_ctrl_test,
            spend_ctrl_test,
            spend_ctrl_test,
            spend_ctrl_test,
            spend_ctrl_test,
            jnp.full((3,), 1000.0),
        ],
        axis=1,
    )

    # Verify both half-None spend combinations leave optional spend fields None.
    res_no_test_spend = methodology.analyze(
        pretest_train_conversions=pretest_train,
        pretest_val_conversions=pretest_val,
        test_conversions=test_conversions,
        treatment_mask=treatment_mask,
        placebo_masks=placebo_masks,
        alpha=0.1,
        experiment_type=experiment_type,
        test_type=test_type,
        treatment_cell_id=2.0,
        pretest_spend=pretest_spend,
        test_spend=None,
    )
    self.assertIsNone(res_no_test_spend.icpd)
    self.assertIsNone(res_no_test_spend.cumulative_icpd_with_cis)
    self.assertIsNone(res_no_test_spend.counterfactual_spend)

    res_no_pretest_spend = methodology.analyze(
        pretest_train_conversions=pretest_train,
        pretest_val_conversions=pretest_val,
        test_conversions=test_conversions,
        treatment_mask=treatment_mask,
        placebo_masks=placebo_masks,
        alpha=0.1,
        experiment_type=experiment_type,
        test_type=test_type,
        treatment_cell_id=2.0,
        pretest_spend=None,
        test_spend=test_spend,
    )
    self.assertIsNone(res_no_pretest_spend.icpd)
    self.assertIsNone(res_no_pretest_spend.cumulative_icpd_with_cis)
    self.assertIsNone(res_no_pretest_spend.counterfactual_spend)

    result = methodology.analyze(
        pretest_train_conversions=pretest_train,
        pretest_val_conversions=pretest_val,
        test_conversions=test_conversions,
        treatment_mask=treatment_mask,
        placebo_masks=placebo_masks,
        alpha=0.1,
        experiment_type=experiment_type,
        test_type=test_type,
        treatment_cell_id=2.0,
        pretest_spend=pretest_spend,
        test_spend=test_spend,
    )

    # Expected studentized statistics from the known fixture:
    # rmse = 50.0, n_treatment_geos = 2.0.
    # signed_geo_avg_cumul_effect = [75.0, 200.0, 300.0].
    # signed_t_placebo = [[-2.0, -3.5, -6.0], [0.5, 2.0, 4.0], [2.0, 5.0, 10.0]]
    # At final step (-1), t_obs = 300.0 / 50.0 = 6.0 and signed_t_placebo[:,-1]
    # = [-6.0, 4.0, 10.0], so ONE_SIDED p_value = (1 + 1) / (1 + 3) = 0.5 and
    # TWO_SIDED p_value = (1 + 2) / (1 + 3) = 0.75.
    expected_p_value = 0.5 if test_type == api.TestType.ONE_SIDED else 0.75
    expected_signed_t_placebo = jnp.array([
        [-2.0, -3.5, -6.0],
        [0.5, 2.0, 4.0],
        [2.0, 5.0, 10.0],
    ])
    expected_cumul_lower_geo, expected_cumul_upper_geo = jax.vmap(
        methodology_util.compute_cis, in_axes=(0, None, 1, None, None)
    )(
        jnp.array([75.0, 200.0, 300.0]),
        50.0,
        expected_signed_t_placebo,
        0.1,
        test_type,
    )
    expected_total_lower_cis = 2.0 * np.array(expected_cumul_lower_geo)
    expected_total_upper_cis = 2.0 * np.array(expected_cumul_upper_geo)
    expected_lift_se = 2.0 * float(
        methodology_util.compute_se(50.0, expected_signed_t_placebo[:, -1])
    )

    self.assertAlmostEqual(result.lift.point_estimate, 600.0, places=4)
    self.assertAlmostEqual(
        result.lift.lower_bound, float(expected_total_lower_cis[-1]), places=4
    )
    if test_type == api.TestType.ONE_SIDED:
      self.assertTrue(np.isinf(result.lift.upper_bound))
    else:
      self.assertAlmostEqual(
          result.lift.upper_bound,
          float(expected_total_upper_cis[-1]),
          places=4,
      )
    self.assertAlmostEqual(
        result.lift.standard_deviation, expected_lift_se, places=4
    )
    self.assertAlmostEqual(result.lift.p_value, expected_p_value, places=5)

    self.assertEqual(result.cumulative_lift_with_cis.shape, (3, 3))
    np.testing.assert_allclose(
        result.cumulative_lift_with_cis[:, 0],
        np.array([150.0, 400.0, 600.0]),
        rtol=5e-5,
        atol=5e-5,
    )
    np.testing.assert_allclose(
        result.cumulative_lift_with_cis[:, 1],
        expected_total_lower_cis,
        rtol=5e-5,
        atol=5e-5,
    )
    np.testing.assert_allclose(
        result.cumulative_lift_with_cis[:, 2],
        expected_total_upper_cis,
        rtol=5e-5,
        atol=5e-5,
    )

    # Verify percent_lift against direct methodology_util.get_percent_lift call
    # on the cumulative totals at the final step (-1).
    expected_log_errors = methodology_util.compute_regularized_log_ratio(
        jnp.array([0.0, 100.0]), jnp.array([70.0, 90.0]), scale=50.0
    )
    expected_log_rmse = jnp.sqrt(jnp.mean(jnp.square(expected_log_errors)))
    p0_log_err = methodology_util.compute_regularized_log_ratio(
        jnp.array([62.0, 78.0]), jnp.array([60.0, 80.0]), scale=70.0
    )
    p1_log_err = methodology_util.compute_regularized_log_ratio(
        jnp.array([63.0, 77.0]), jnp.array([60.0, 80.0]), scale=70.0
    )
    p2_log_err = methodology_util.compute_regularized_log_ratio(
        jnp.array([65.0, 75.0]), jnp.array([60.0, 80.0]), scale=70.0
    )
    expected_placebo_log_rmses = jnp.array([
        jnp.sqrt(jnp.mean(jnp.square(p0_log_err))),
        jnp.sqrt(jnp.mean(jnp.square(p1_log_err))),
        jnp.sqrt(jnp.mean(jnp.square(p2_log_err))),
    ])
    expected_pct_lift = methodology_util.get_percent_lift(
        2.0 * jnp.sum(treat_test),
        jnp.array(570.0),
        expected_log_rmse,
        jnp.array([
            jnp.sum(p0_test),
            jnp.sum(p1_test),
            jnp.sum(p2_test),
        ]),
        jnp.array([300.0, 300.0, 300.0]),
        expected_placebo_log_rmses,
        0.1,
        experiment_type,
        test_type,
    )
    self.assertAlmostEqual(
        result.percent_lift.point_estimate,
        expected_pct_lift.point_estimate,
        places=5,
    )
    self.assertAlmostEqual(
        result.percent_lift.lower_bound,
        expected_pct_lift.lower_bound,
        places=5,
    )
    if test_type == api.TestType.ONE_SIDED:
      self.assertTrue(np.isinf(result.percent_lift.upper_bound))
    else:
      self.assertAlmostEqual(
          result.percent_lift.upper_bound,
          expected_pct_lift.upper_bound,
          places=5,
      )
    self.assertAlmostEqual(
        result.percent_lift.standard_deviation,
        expected_pct_lift.standard_deviation,
        places=5,
    )
    self.assertAlmostEqual(
        result.percent_lift.p_value, expected_pct_lift.p_value, places=5
    )

    # Pointwise difference has 4 pretest rows + 3 test rows = 7 rows.
    # Full-pretest intercept is -5.0, so y_pred_pre = [45, 55, 55, 75] and
    # total_y_pred_pre = [90, 110, 110, 150], while total_y_pretest =
    # [120, 140, 0, 200] -> pretest diff = sign * [30, 30, -110, 50].
    self.assertEqual(result.pointwise_difference_with_cis.shape, (7, 3))
    np.testing.assert_allclose(
        result.pointwise_difference_with_cis[:4, 0],
        sign * np.array([30.0, 30.0, -110.0, 50.0]),
        rtol=5e-5,
        atol=5e-5,
    )
    self.assertTrue(
        np.all(np.isnan(result.pointwise_difference_with_cis[:4, 1:]))
    )
    expected_signed_t_placebo_pw = jnp.array([
        [-2.0, -1.5, -2.5],
        [0.5, 1.5, 2.0],
        [2.0, 3.0, 5.0],
    ])
    expected_pw_lower_geo, expected_pw_upper_geo = jax.vmap(
        methodology_util.compute_cis, in_axes=(0, None, 1, None, None)
    )(
        jnp.array([75.0, 125.0, 100.0]),
        50.0,
        expected_signed_t_placebo_pw,
        0.1,
        test_type,
    )
    expected_pw_lower = 2.0 * np.array(expected_pw_lower_geo)
    expected_pw_upper = 2.0 * np.array(expected_pw_upper_geo)
    np.testing.assert_allclose(
        result.pointwise_difference_with_cis[4:, 0],
        np.array([150.0, 250.0, 200.0]),
        rtol=5e-5,
        atol=5e-5,
    )
    np.testing.assert_allclose(
        result.pointwise_difference_with_cis[4:, 1],
        expected_pw_lower,
        rtol=5e-5,
        atol=5e-5,
    )
    np.testing.assert_allclose(
        result.pointwise_difference_with_cis[4:, 2],
        expected_pw_upper,
        rtol=5e-5,
        atol=5e-5,
    )

    # Counterfactual conversions has 4 pretest rows + 3 test rows = 7 rows.
    self.assertEqual(result.counterfactual_conversions_with_cis.shape, (7, 4))
    np.testing.assert_allclose(
        result.counterfactual_conversions_with_cis[:4, 0],
        np.array([120.0, 140.0, 0.0, 200.0]),
        rtol=5e-5,
        atol=5e-5,
    )
    np.testing.assert_allclose(
        result.counterfactual_conversions_with_cis[:4, 1],
        np.array([90.0, 110.0, 110.0, 150.0]),
        rtol=5e-5,
        atol=5e-5,
    )
    self.assertTrue(
        np.all(np.isnan(result.counterfactual_conversions_with_cis[:4, 2:]))
    )
    total_y_test = 2.0 * np.array(treat_test)
    if experiment_type != api.ExperimentType.GO_DARK:
      expected_cf_lower = total_y_test - expected_pw_upper
      expected_cf_upper = total_y_test - expected_pw_lower
    else:
      expected_cf_lower = total_y_test + expected_pw_lower
      expected_cf_upper = total_y_test + expected_pw_upper
    np.testing.assert_allclose(
        result.counterfactual_conversions_with_cis[4:, 0],
        total_y_test,
        rtol=5e-5,
        atol=5e-5,
    )
    np.testing.assert_allclose(
        result.counterfactual_conversions_with_cis[4:, 1],
        np.array([170.0, 190.0, 210.0]),
        rtol=5e-5,
        atol=5e-5,
    )
    np.testing.assert_allclose(
        result.counterfactual_conversions_with_cis[4:, 2],
        expected_cf_lower,
        rtol=5e-5,
        atol=5e-5,
    )
    np.testing.assert_allclose(
        result.counterfactual_conversions_with_cis[4:, 3],
        expected_cf_upper,
        rtol=5e-5,
        atol=5e-5,
    )

    # Check iCPD results: cumulative_incremental_spend = [4, 16, 26],
    # signed_total_cumul_effect = [150, 400, 600] ->
    # cumulative_icpd = [37.5, 25.0, 600.0 / 26.0].
    expected_cumul_inc_spend = np.array([4.0, 16.0, 26.0])
    expected_icpd_lower = expected_total_lower_cis / expected_cumul_inc_spend
    expected_icpd_upper = expected_total_upper_cis / expected_cumul_inc_spend
    self.assertIsNotNone(result.icpd)
    self.assertAlmostEqual(result.icpd.point_estimate, 600.0 / 26.0, places=4)
    self.assertAlmostEqual(
        result.icpd.lower_bound, float(expected_icpd_lower[-1]), places=4
    )
    if test_type == api.TestType.ONE_SIDED:
      self.assertTrue(np.isinf(result.icpd.upper_bound))
    else:
      self.assertAlmostEqual(
          result.icpd.upper_bound, float(expected_icpd_upper[-1]), places=4
      )
    self.assertAlmostEqual(
        result.icpd.standard_deviation,
        expected_lift_se / 26.0,
        places=5,
    )
    self.assertAlmostEqual(result.icpd.p_value, expected_p_value, places=5)
    self.assertIsNotNone(result.cumulative_icpd_with_cis)
    self.assertEqual(result.cumulative_icpd_with_cis.shape, (3, 3))
    np.testing.assert_allclose(
        result.cumulative_icpd_with_cis[:, 0],
        np.array([37.5, 25.0, 600.0 / 26.0]),
        rtol=5e-5,
        atol=5e-5,
    )
    np.testing.assert_allclose(
        result.cumulative_icpd_with_cis[:, 1],
        expected_icpd_lower,
        rtol=5e-5,
        atol=5e-5,
    )
    np.testing.assert_allclose(
        result.cumulative_icpd_with_cis[:, 2],
        expected_icpd_upper,
        rtol=5e-5,
        atol=5e-5,
    )
    self.assertIsNotNone(result.counterfactual_spend)
    np.testing.assert_allclose(
        result.counterfactual_spend,
        np.array([110.0, 130.0, 150.0]),
        rtol=5e-5,
        atol=5e-5,
    )


if __name__ == '__main__':
  absltest.main()
