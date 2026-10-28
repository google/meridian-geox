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

  def _assert_allclose(self, actual, expected):
    actual_arr = np.asarray(actual)
    expected_arr = np.asarray(expected)
    self.assertEqual(actual_arr.shape, expected_arr.shape)
    np.testing.assert_allclose(
        actual_arr, expected_arr, rtol=5e-5, atol=5e-5
    )

  def _assert_estimate(self, actual: api.Estimate | None, expected):
    self.assertIsNotNone(actual)
    assert actual is not None
    self._assert_allclose(
        [
            actual.point_estimate,
            actual.lower_bound,
            actual.upper_bound,
            actual.standard_deviation,
            actual.p_value,
        ],
        expected,
    )

  def test_compute_placebo_effect_from_mask_excludes_original_treated_geos(
      self,
  ):
    methodology = _MeanDiffMethodology()
    # Geo 0: originally treated (mask=1, +1000 offset catches control leak).
    # Geo 1: placebo cell 2 (p_mask=2), train intercept +5, val [0, 164] (clips
    #   0 -> 0.082 at scale_val=82; combined pretest intercept = 6 != 5).
    # Geos 2, 3: valid controls (mask=0, p_mask=0).
    # Geo 4: untreated cell 3 (mask=3, p_mask=3, +2000 catches >= cell_id bug).
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
    # Val pred [75, 85] vs [0, 164] -> p_rmse = sqrt(5933),
    # p_log_rmse = sqrt(0.5 * (log(0.082/75)^2 + log(164/85)^2)) = 4.84376936.
    self._assert_allclose(py_test, [106.0, 126.0])
    self._assert_allclose(py_pred, [96.0, 106.0])
    self._assert_allclose(p_rmse, np.sqrt(5933.0))
    self._assert_allclose(p_log_rmse, 4.84376936)

  @parameterized.named_parameters(
      dict(testcase_name='positive_spend', incremental_spend_sign=1.0),
      dict(testcase_name='go_dark_spend', incremental_spend_sign=-1.0),
  )
  def test_compute_icpd_and_pytree_registration(self, incremental_spend_sign):
    methodology = _MeanDiffMethodology()
    # Cell 2 treats Geos 1, 2 (n=2) against control Geo 3 (mask=0), ignoring
    # cells 1 (Geo 0) and 3 (Geo 4). Pretest spend intercept = +10 -> pred spend
    # [30, 40, 50], counterfactual [60, 80, 100], signed cumul [4, 16, 26].
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

    icpd_res = methodology._compute_icpd(
        pretest_spend=pretest_spend,
        test_spend=test_spend,
        treatment_mask=jnp.array([1.0, 2.0, 2.0, 0.0, 3.0]),
        total_cumul_effect=jnp.array([20.0, 48.0, 104.0]),
        total_lower_cis=jnp.array([12.0, 32.0, 78.0]),
        total_upper_cis=jnp.array([28.0, 64.0, 130.0]),
        incremental_spend_sign=incremental_spend_sign,
        treatment_cell_id=2.0,
    )
    self._assert_allclose(
        icpd_res.cumulative_incremental_spend, [4.0, 16.0, 26.0]
    )
    self._assert_allclose(icpd_res.counterfactual_spend, [60.0, 80.0, 100.0])
    self._assert_allclose(icpd_res.cumulative_icpd, [5.0, 3.0, 4.0])
    self._assert_allclose(icpd_res.lower_bound, [3.0, 2.0, 3.0])
    self._assert_allclose(icpd_res.upper_bound, [7.0, 4.0, 5.0])

    # Verify all 5 fields round-trip in exact order through JAX PyTree map.
    doubled = jax.tree_util.tree_map(lambda x: 2.0 * x, icpd_res)
    self._assert_allclose(doubled.cumulative_icpd, [10.0, 6.0, 8.0])
    self._assert_allclose(doubled.lower_bound, [6.0, 4.0, 6.0])
    self._assert_allclose(doubled.upper_bound, [14.0, 8.0, 10.0])
    self._assert_allclose(
        doubled.cumulative_incremental_spend, [8.0, 32.0, 52.0]
    )
    self._assert_allclose(doubled.counterfactual_spend, [120.0, 160.0, 200.0])

  @parameterized.named_parameters(
      dict(testcase_name='both_none', has_pre=False, has_test=False),
      dict(testcase_name='only_pretest', has_pre=True, has_test=False),
      dict(testcase_name='only_test', has_pre=False, has_test=True),
  )
  def test_analyze_missing_spend_leaves_icpd_fields_none(
      self, has_pre, has_test
  ):
    methodology = _MeanDiffMethodology()
    result = methodology.analyze(
        pretest_train_conversions=jnp.array(
            [[20.0, 10.0, 10.0], [30.0, 20.0, 20.0]]
        ),
        pretest_val_conversions=jnp.array(
            [[42.0, 30.0, 30.0], [48.0, 40.0, 40.0]]
        ),
        test_conversions=jnp.array([[65.0, 50.0, 50.0], [75.0, 60.0, 60.0]]),
        treatment_mask=jnp.array([1.0, 0.0, 0.0]),
        placebo_masks=jnp.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
        alpha=0.1,
        experiment_type=api.ExperimentType.HOLDBACK,
        pretest_spend=jnp.ones((4, 3)) if has_pre else None,
        test_spend=jnp.ones((2, 3)) if has_test else None,
    )
    self.assertIsNone(result.icpd)
    self.assertIsNone(result.cumulative_icpd_with_cis)
    self.assertIsNone(result.counterfactual_spend)

  @parameterized.named_parameters(
      dict(
          testcase_name='zero_spend_two_sided',
          test_type=api.TestType.TWO_SIDED,
          spend_deltas=[0.0, 0.0, 0.0],
          expected_p_value=1.0,
      ),
      dict(
          testcase_name='mixed_zero_pos_neg_spend_one_sided',
          test_type=api.TestType.ONE_SIDED,
          spend_deltas=[0.0, 4.0, -6.0],
          expected_p_value=2.0 / 3.0,
      ),
  )
  def test_analyze_non_positive_incremental_spend_returns_nan(
      self, test_type, spend_deltas, expected_p_value
  ):
    methodology = _MeanDiffMethodology()
    # Geo 0 treated: rmse=0.5, cumul lift=[5.0, 10.0, 0.125] -> t_obs[-1]=0.25.
    # Geos 1..3 controls have placebo t-stats [0.5, 0.0] (straddling 0.25 so
    # +/-1 or sign shifts on estimate[-1] flip p_value; lift SE = 0.125 > 0).
    spend_ctrl = jnp.array([50.0, 60.0, 70.0])
    spend_treat = jnp.array([55.0, 65.0, 75.0]) + jnp.array(spend_deltas)
    result = methodology.analyze(
        pretest_train_conversions=jnp.array(
            [[20.0, 10.0, 10.0, 10.0], [30.0, 20.0, 20.0, 20.0]]
        ),
        pretest_val_conversions=jnp.array(
            [[40.5, 32.0, 28.0, 30.0], [49.5, 38.0, 42.0, 40.0]]
        ),
        test_conversions=jnp.array([
            [65.0, 51.0, 50.0, 49.0],
            [75.0, 60.0, 60.0, 60.0],
            [70.125, 70.0, 70.0, 70.0],
        ]),
        treatment_mask=jnp.array([1.0, 0.0, 0.0, 0.0]),
        placebo_masks=jnp.array(
            [[0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]]
        ),
        alpha=0.1,
        experiment_type=api.ExperimentType.HOLDBACK,
        test_type=test_type,
        pretest_spend=jnp.array([
            [15.0, 10.0, 10.0, 10.0],
            [25.0, 20.0, 20.0, 20.0],
            [35.0, 30.0, 30.0, 30.0],
            [45.0, 40.0, 40.0, 40.0],
        ]),
        test_spend=jnp.stack(
            [spend_treat, spend_ctrl, spend_ctrl, spend_ctrl], axis=1
        ),
    )

    # Final cumul spend is 0.0 or -2.0 -> all final iCPD stats must be NaN.
    self._assert_allclose(result.lift.standard_deviation, 0.125)
    self._assert_allclose(result.lift.p_value, expected_p_value)
    self._assert_estimate(
        result.icpd, [np.nan, np.nan, np.nan, np.nan, expected_p_value]
    )
    expected_cumul_icpd = np.full((3, 3), np.nan)
    if spend_deltas[1] > 0:
      # Step 1 has +4.0 cumul spend, lift 10.0, lower CI 9.775.
      expected_cumul_icpd[1] = [2.5, 2.44375, np.inf]
    self._assert_allclose(result.cumulative_icpd_with_cis, expected_cumul_icpd)
    self._assert_allclose(result.counterfactual_spend, [55.0, 65.0, 75.0])

  @parameterized.named_parameters(
      dict(
          testcase_name='holdback_two_sided',
          experiment_type=api.ExperimentType.HOLDBACK,
          test_type=api.TestType.TWO_SIDED,
          sign=1.0,
          expected_pct_lift=(
              1.05263158, -0.99992626, 800.09563424, 15.31307667, 0.75
          ),
      ),
      dict(
          testcase_name='holdback_one_sided',
          experiment_type=api.ExperimentType.HOLDBACK,
          test_type=api.TestType.ONE_SIDED,
          sign=1.0,
          expected_pct_lift=(
              1.05263158, -0.99986187, np.inf, 15.31307667, 0.75
          ),
      ),
      dict(
          testcase_name='go_dark_two_sided',
          experiment_type=api.ExperimentType.GO_DARK,
          test_type=api.TestType.TWO_SIDED,
          sign=-1.0,
          expected_pct_lift=(
              0.999, -168.28605508, 0.99999664, 0.00808851, 0.75
          ),
      ),
      dict(
          testcase_name='go_dark_one_sided',
          experiment_type=api.ExperimentType.GO_DARK,
          test_type=api.TestType.ONE_SIDED,
          sign=-1.0,
          expected_pct_lift=(0.999, -74.49718252, np.inf, 0.00808851, 0.5),
      ),
      dict(
          testcase_name='heavy_up_two_sided',
          experiment_type=api.ExperimentType.HEAVY_UP,
          test_type=api.TestType.TWO_SIDED,
          sign=1.0,
          expected_pct_lift=(
              1.05263158, -0.99992626, 800.09563424, 15.31307667, 0.75
          ),
      ),
      dict(
          testcase_name='heavy_up_one_sided',
          experiment_type=api.ExperimentType.HEAVY_UP,
          test_type=api.TestType.ONE_SIDED,
          sign=1.0,
          expected_pct_lift=(
              1.05263158, -0.99986187, np.inf, 15.31307667, 0.75
          ),
      ),
  )
  def test_analyze_via_base_template(
      self, experiment_type, test_type, sign, expected_pct_lift
  ):
    methodology = _MeanDiffMethodology()
    # 9 geos targeting cell 2 (Geos 1, 2; n=2), ignoring cells 1 (Geo 0) and 3
    # (Geo 8). Geos 3, 4, 5 are placebo-treated (p_rmses=[2, 3, 5]); Geo 6 is a
    # clean control ([50, 60] train, [60, 80] val, [90, 100, 110] test); Geo 7
    # balances placebo offsets so both real and placebo control means == Geo 6.
    # Treated Geos 1, 2 have train intercept +10, val [0, 100] (rmse=50, clips
    # 0 -> 0.05; full pretest intercept = -5 -> test pred [85, 95, 105]).
    pretest_train = jnp.array([
        [1000.0, 60.0, 60.0, 50.0, 50.0, 50.0, 50.0, 50.0, 2000.0],
        [1000.0, 70.0, 70.0, 60.0, 60.0, 60.0, 60.0, 60.0, 2000.0],
    ])
    pretest_val = jnp.array([
        [1000.0, 0.0, 0.0, 62.0, 63.0, 65.0, 60.0, 50.0, 2000.0],
        [1000.0, 100.0, 100.0, 78.0, 77.0, 75.0, 80.0, 90.0, 2000.0],
    ])
    base_test = jnp.array([90.0, 100.0, 110.0])
    treat_test = jnp.array([85.0, 95.0, 105.0]) + sign * jnp.array(
        [75.0, 125.0, 100.0]
    )
    test_conversions = jnp.stack(
        [
            jnp.full((3,), 1000.0),
            treat_test,
            treat_test,
            base_test + sign * jnp.array([-4.0, -3.0, -5.0]),
            base_test + sign * jnp.array([1.5, 4.5, 6.0]),
            base_test + sign * jnp.array([10.0, 15.0, 25.0]),
            base_test,
            base_test - sign * jnp.array([7.5, 16.5, 26.0]),
            jnp.full((3,), 2000.0),
        ],
        axis=1,
    )
    pretest_spend = jnp.array([
        [500.0, 15.0, 15.0, 10.0, 10.0, 10.0, 10.0, 10.0, 1000.0],
        [500.0, 25.0, 25.0, 20.0, 20.0, 20.0, 20.0, 20.0, 1000.0],
        [500.0, 35.0, 35.0, 30.0, 30.0, 30.0, 30.0, 30.0, 1000.0],
        [500.0, 45.0, 45.0, 40.0, 40.0, 40.0, 40.0, 40.0, 1000.0],
    ])
    spend_treat = jnp.array([55.0, 65.0, 75.0]) + sign * jnp.array(
        [2.0, 6.0, 5.0]
    )
    spend_ctrl = jnp.array([50.0, 60.0, 70.0])
    test_spend = jnp.stack(
        [
            jnp.full((3,), 500.0),
            spend_treat,
            spend_treat,
            spend_ctrl,
            spend_ctrl,
            spend_ctrl,
            spend_ctrl,
            spend_ctrl,
            jnp.full((3,), 1000.0),
        ],
        axis=1,
    )

    result = methodology.analyze(
        pretest_train_conversions=pretest_train,
        pretest_val_conversions=pretest_val,
        test_conversions=test_conversions,
        treatment_mask=jnp.array([1.0, 2.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 3.0]),
        placebo_masks=jnp.array([
            [1.0, 0.0, 0.0, 2.0, 1.0, 3.0, 0.0, 1.0, 3.0],
            [1.0, 0.0, 0.0, 1.0, 2.0, 3.0, 0.0, 1.0, 3.0],
            [1.0, 0.0, 0.0, 1.0, 3.0, 2.0, 0.0, 1.0, 3.0],
        ]),
        alpha=0.1,
        experiment_type=experiment_type,
        test_type=test_type,
        treatment_cell_id=2.0,
        pretest_spend=pretest_spend,
        test_spend=test_spend,
    )

    # Signed placebo t-stats: cumul [[-2, -3.5, -6], [0.5, 2, 4], [2, 5, 10]],
    # pointwise [[-2, -1.5, -2.5], [0.5, 1.5, 2], [2, 3, 5]]; t_obs[-1] = 6.0.
    is_one_sided = test_type == api.TestType.ONE_SIDED
    p_val = 0.5 if is_one_sided else 0.75
    lift_se = 100.0 * np.sqrt(392.0 / 9.0)
    cumul_lo = np.array(
        [-20.0, -40.0, -280.0] if is_one_sided else [-35.0, -70.0, -340.0]
    )
    cumul_hi = np.array(
        [np.inf, np.inf, np.inf] if is_one_sided else [325.0, 695.0, 1100.0]
    )
    pw_lo = np.array(
        [-20.0, -20.0, -240.0] if is_one_sided else [-35.0, -35.0, -270.0]
    )
    pw_hi = np.array(
        [np.inf, np.inf, np.inf] if is_one_sided else [325.0, 370.0, 405.0]
    )

    self._assert_estimate(
        result.lift, [600.0, cumul_lo[-1], cumul_hi[-1], lift_se, p_val]
    )
    self._assert_allclose(
        result.cumulative_lift_with_cis,
        np.column_stack(([150.0, 400.0, 600.0], cumul_lo, cumul_hi)),
    )
    self._assert_estimate(result.percent_lift, expected_pct_lift)

    nan4 = np.full(4, np.nan)
    expected_pw = np.vstack((
        np.column_stack(
            (sign * np.array([30.0, 30.0, -110.0, 50.0]), nan4, nan4)
        ),
        np.column_stack(([150.0, 250.0, 200.0], pw_lo, pw_hi)),
    ))
    self._assert_allclose(result.pointwise_difference_with_cis, expected_pw)

    total_y_test = 2.0 * np.array(treat_test)
    cf_lo, cf_hi = (
        (total_y_test + pw_lo, total_y_test + pw_hi)
        if sign < 0
        else (total_y_test - pw_hi, total_y_test - pw_lo)
    )
    expected_cf = np.vstack((
        np.column_stack((
            [120.0, 140.0, 0.0, 200.0],
            [90.0, 110.0, 110.0, 150.0],
            nan4,
            nan4,
        )),
        np.column_stack((total_y_test, [170.0, 190.0, 210.0], cf_lo, cf_hi)),
    ))
    self._assert_allclose(
        result.counterfactual_conversions_with_cis, expected_cf
    )

    cumul_inc_spend = np.array([4.0, 16.0, 26.0])
    icpd_lo = cumul_lo / cumul_inc_spend
    icpd_hi = cumul_hi / cumul_inc_spend
    self._assert_estimate(
        result.icpd,
        [600.0 / 26.0, icpd_lo[-1], icpd_hi[-1], lift_se / 26.0, p_val],
    )
    self._assert_allclose(
        result.cumulative_icpd_with_cis,
        np.column_stack(([37.5, 25.0, 600.0 / 26.0], icpd_lo, icpd_hi)),
    )
    self._assert_allclose(result.counterfactual_spend, [110.0, 130.0, 150.0])


if __name__ == '__main__':
  absltest.main()
