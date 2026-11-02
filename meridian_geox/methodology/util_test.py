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

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
from meridian_geox import api
from meridian_geox.methodology import util
import numpy as np

# Placebo t-statistics whose absolute values are 1, ..., 9.
_PLACEBO_T = (1.0, -2.0, 3.0, -4.0, 5.0, -6.0, 7.0, -8.0, 9.0)


class UtilTest(parameterized.TestCase):

  def test_compute_regularized_log_ratio_standard(self):
    y_num = jnp.array([2.0])
    y_den = jnp.array([1.0])
    expected = jnp.log(2.0 / 1.0)
    result = util.compute_regularized_log_ratio(y_num, y_den)
    np.testing.assert_allclose(result, expected, atol=1e-6)

  def test_compute_regularized_log_ratio_zero(self):
    # Both exactly zero.
    y_num = jnp.array([0.0])
    y_den = jnp.array([0.0])
    # Both are thresholded to 1e-9, so ratio is 1.0, log is 0.0
    expected = jnp.array([0.0])
    result = util.compute_regularized_log_ratio(y_num, y_den)
    np.testing.assert_allclose(result, expected, atol=1e-6)

  def test_compute_regularized_log_ratio_asymmetric_extreme(self):
    # num is 100, den is 0.0
    y_num = jnp.array([100.0])
    y_den = jnp.array([0.0])
    # scale = 100.0, threshold = 1e-3 * 100.0 = 0.1
    # num is max(100.0, 0.1) = 100.0
    # den is max(0.0, 0.1) = 0.1
    expected = jnp.log(100.0 / 0.1)
    result = util.compute_regularized_log_ratio(y_num, y_den)
    np.testing.assert_allclose(result, expected, atol=1e-6)

  def test_compute_regularized_log_ratio_negative(self):
    # Negative values are clipped to a positive threshold before log.
    y_num = jnp.array([-10.0])
    y_den = jnp.array([-5.0])
    # scale = 10.0, threshold = 1e-3 * 10.0 = 0.01
    # num -> max(-10.0, 0.01) = 0.01
    # den -> max(-5.0, 0.01) = 0.01
    # Both are clipped to threshold, log ratio is log(1) = 0.0
    expected = jnp.array([0.0])
    result = util.compute_regularized_log_ratio(y_num, y_den)
    np.testing.assert_allclose(result, expected, atol=1e-6)

  def test_compute_regularized_log_ratio_custom_scale(self):
    y_num = jnp.array([0.0])
    y_den = jnp.array([0.0])
    # Provide a global scale
    scale = jnp.array(100.0)
    # threshold = 1e-3 * 100 = 0.1
    # Both are clipped to 0.1, log ratio is log(1) = 0.0
    expected = jnp.array([0.0])
    result = util.compute_regularized_log_ratio(y_num, y_den, scale=scale)
    np.testing.assert_allclose(result, expected, atol=1e-6)

  def test_compute_studentized_p_value(self):
    p_val = util.compute_studentized_p_value(
        estimate=10.0,
        rmse=1.0,
        t_placebo=jnp.array([0.0, 5.0, 10.0, 15.0]),
        test_type=api.TestType.TWO_SIDED,
    )
    # n_extreme_left = sum([0, 5, 10, 15] <= 10.0) = 3
    # n_extreme_right = sum([0, 5, 10, 15] >= 10.0) = 2
    # n_extreme = 2 * min(3, 2) = 4
    # p_val = (1 + 4) / (1 + 4) = 1.0
    self.assertAlmostEqual(float(p_val), 1.0)

  def test_compute_studentized_p_value_absolute(self):
    p_val = util.compute_studentized_p_value(
        estimate=10.0,
        rmse=1.0,
        t_placebo=jnp.array([0.0, 5.0, 10.0, 15.0]),
        test_type=api.TestType.TWO_SIDED,
        use_absolute_placebo_statistics=True,
    )
    # n_extreme = sum(|[0, 5, 10, 15]| >= |10.0|) = 2
    # p_val = (1 + 2) / (1 + 4) = 0.6
    self.assertAlmostEqual(float(p_val), 0.6)

  @parameterized.parameters(-5.0, 5.0)
  def test_compute_studentized_p_value_two_sided_symmetric(self, estimate):
    p_val = util.compute_studentized_p_value(
        estimate=estimate,
        rmse=1.0,
        t_placebo=jnp.array(_PLACEBO_T),
        test_type=api.TestType.TWO_SIDED,
        use_absolute_placebo_statistics=True,
    )
    # n_extreme = sum(|t| >= 5) = 5, so p_val = (1 + 5) / (1 + 9) = 0.6.
    self.assertAlmostEqual(float(p_val), 0.6)

  @parameterized.parameters((5.0, 0.3), (0.0, 0.55), (-5.0, 0.8))
  def test_compute_studentized_p_value_one_sided(self, estimate, expected):
    p_val = util.compute_studentized_p_value(
        estimate=estimate,
        rmse=1.0,
        t_placebo=jnp.array(_PLACEBO_T),
        test_type=api.TestType.ONE_SIDED,
        use_absolute_placebo_statistics=True,
    )
    # p = #{v >= t_obs} / 20 over the values t_obs, -t_obs and +-t_j, which is
    # 6 / 20 for t_obs = 5, 11 / 20 for t_obs = 0 and 16 / 20 for t_obs = -5.
    self.assertAlmostEqual(float(p_val), expected)

  @parameterized.parameters(
      (api.TestType.TWO_SIDED, 0.05), (api.TestType.ONE_SIDED, 0.025)
  )
  def test_compute_studentized_p_value_absolute_is_exact(
      self, test_type, expected
  ):
    # |t_obs| = 20 exceeds all 19 |t_j|, so the two-sided p-value is 1 / 20.
    p_val = util.compute_studentized_p_value(
        estimate=20.0,
        rmse=1.0,
        t_placebo=jnp.arange(1.0, 20.0),
        test_type=test_type,
        use_absolute_placebo_statistics=True,
    )
    self.assertEqual(p_val, expected)

  def test_compute_se(self):
    rmse = 2.0
    t_placebo = jnp.array([-1.0, 0.0, 1.0])
    se = util.compute_se(rmse, t_placebo)
    # std([-1, 0, 1]) = sqrt(2/3)
    expected_se = jnp.std(t_placebo) * rmse
    self.assertAlmostEqual(se, expected_se)

  def test_compute_se_absolute(self):
    se = util.compute_se(
        2.0, jnp.array([1.0, 2.0, 3.0]), use_absolute_placebo_statistics=True
    )
    # sqrt(mean([1, 4, 9])) * 2 = sqrt(14 / 3) * 2
    self.assertAlmostEqual(float(se), np.sqrt(14.0 / 3.0) * 2.0, places=5)

  def test_compute_cis_two_sided(self):
    lower, upper = util.compute_cis(
        100.0, 10.0, jnp.array([-1.0, 0.0, 1.0]), 0.1, api.TestType.TWO_SIDED
    )
    # The 0.95 and 0.05 quantiles of [-1, 0, 1] are 0.9 and -0.9.
    self.assertAlmostEqual(float(lower), 91.0, places=4)
    self.assertAlmostEqual(float(upper), 109.0, places=4)

  def test_compute_cis_one_sided(self):
    lower, upper = util.compute_cis(
        100.0, 10.0, jnp.array([-1.0, 0.0, 1.0]), 0.1, api.TestType.ONE_SIDED
    )
    # The 0.9 quantile of [-1, 0, 1] is 0.8.
    self.assertAlmostEqual(float(lower), 92.0, places=4)
    self.assertEqual(upper, jnp.inf)

  @parameterized.named_parameters(
      dict(testcase_name='mixed_signs', t_placebo=_PLACEBO_T),
      dict(
          testcase_name='all_positive',
          t_placebo=tuple(abs(t) for t in _PLACEBO_T),
      ),
  )
  def test_compute_cis_two_sided_absolute(self, t_placebo):
    lower, upper = util.compute_cis(
        100.0,
        10.0,
        jnp.array(t_placebo),
        0.2,
        api.TestType.TWO_SIDED,
        use_absolute_placebo_statistics=True,
    )
    # k = ceil(0.8 * (9 + 1)) = 8 and the 8th smallest |t| is 8, so the CI is
    # 100 -/+ 8 * 10 regardless of the signs of t.
    self.assertAlmostEqual(float(lower), 20.0, places=4)
    self.assertAlmostEqual(float(upper), 180.0, places=4)

  def test_compute_cis_two_sided_integer_rank(self):
    lower, upper = util.compute_cis(
        100.0,
        10.0,
        jnp.array(_PLACEBO_T),
        0.1,
        api.TestType.TWO_SIDED,
        use_absolute_placebo_statistics=True,
    )
    # k = 0.9 * (9 + 1) = 9 exactly, so the 9th smallest |t| (9) is used.
    self.assertAlmostEqual(float(lower), 10.0, places=4)
    self.assertAlmostEqual(float(upper), 190.0, places=4)

  @parameterized.named_parameters(
      dict(testcase_name='mixed_signs', t_placebo=_PLACEBO_T),
      dict(
          testcase_name='all_negative',
          t_placebo=tuple(-abs(t) for t in _PLACEBO_T),
      ),
  )
  def test_compute_cis_one_sided_absolute(self, t_placebo):
    lower, upper = util.compute_cis(
        100.0,
        10.0,
        jnp.array(t_placebo),
        0.2,
        api.TestType.ONE_SIDED,
        use_absolute_placebo_statistics=True,
    )
    # k = ceil(0.6 * (9 + 1)) = 6 and the 6th smallest |t| is 6, so the lower
    # bound is 100 - 6 * 10 regardless of the signs of t.
    self.assertAlmostEqual(float(lower), 40.0, places=4)
    self.assertEqual(upper, jnp.inf)

  @parameterized.parameters(
      (api.TestType.TWO_SIDED, 0.05, 19),
      (api.TestType.TWO_SIDED, 0.1, 9),
      (api.TestType.TWO_SIDED, 0.2, 4),
      (api.TestType.TWO_SIDED, 1 / 3, 2),
      (api.TestType.TWO_SIDED, 0.5, 1),
      (api.TestType.ONE_SIDED, 0.05, 9),
      (api.TestType.ONE_SIDED, 0.1, 4),
      (api.TestType.ONE_SIDED, 0.2, 2),
      (api.TestType.ONE_SIDED, 0.25, 1),
  )
  def test_compute_cis_absolute_finite_iff_enough_placebos(
      self, test_type, alpha, min_count
  ):
    t_placebo = jnp.arange(1.0, min_count + 1.0)
    lower, upper = util.compute_cis(
        0.0,
        1.0,
        t_placebo,
        alpha,
        test_type,
        use_absolute_placebo_statistics=True,
    )
    self.assertTrue(np.isfinite(lower))
    if test_type == api.TestType.TWO_SIDED:
      self.assertTrue(np.isfinite(upper))
    lower, upper = util.compute_cis(
        0.0,
        1.0,
        t_placebo[:-1],
        alpha,
        test_type,
        use_absolute_placebo_statistics=True,
    )
    self.assertEqual(lower, -jnp.inf)
    self.assertEqual(upper, jnp.inf)

  @parameterized.product(
      test_type=(api.TestType.TWO_SIDED, api.TestType.ONE_SIDED),
      alpha=(0.05, 0.1),
  )
  def test_compute_cis_excludes_zero_iff_p_value_at_most_alpha(
      self, test_type, alpha
  ):
    # With 19 placebos, alpha * (19 + 1) is an integer, so some p-values are
    # exactly alpha.
    t_placebo = jax.random.normal(jax.random.key(0), (19,)) + 0.7
    estimates = jnp.concatenate(
        [t_placebo, -t_placebo, jnp.linspace(-4.0, 4.0, 41)]
    )
    for estimate in estimates:
      lower, upper = util.compute_cis(
          estimate,
          1.0,
          t_placebo,
          alpha,
          test_type,
          use_absolute_placebo_statistics=True,
      )
      p_val = util.compute_studentized_p_value(
          estimate,
          1.0,
          t_placebo,
          test_type,
          use_absolute_placebo_statistics=True,
      )
      self.assertEqual(
          bool(lower > 0.0 or upper < 0.0),
          float(p_val) <= alpha,
          msg=f'estimate={estimate}',
      )

  def test_compute_cis_absolute_with_zero_rmse_matches_p_value(self):
    # t_obs = 1e-10 / 1e-9 = 0.1 is below all |t|, so p = 1 and the CI must
    # contain zero.
    t_placebo = jnp.array(_PLACEBO_T)
    lower, upper = util.compute_cis(
        1e-10, 0.0, t_placebo, 0.1, use_absolute_placebo_statistics=True
    )
    p_val = util.compute_studentized_p_value(
        1e-10, 0.0, t_placebo, use_absolute_placebo_statistics=True
    )
    self.assertLess(lower, 0.0)
    self.assertGreater(upper, 0.0)
    self.assertEqual(p_val, 1.0)

  def test_compute_cis_one_sided_contains_estimate(self):
    # Placebos centered well below zero, against the tested direction.
    t_placebo = jax.random.normal(jax.random.key(0), (19,)) - 3.0
    for estimate in jnp.linspace(-4.0, 4.0, 41):
      lower, _ = util.compute_cis(
          estimate,
          1.0,
          t_placebo,
          0.1,
          api.TestType.ONE_SIDED,
          use_absolute_placebo_statistics=True,
      )
      self.assertLessEqual(lower, estimate, msg=f'estimate={estimate}')

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

    estimate = util.get_percent_lift(
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

  @parameterized.named_parameters(
      dict(
          testcase_name='signed_holdback_two_sided',
          use_absolute_placebo_statistics=False,
          experiment_type=api.ExperimentType.HOLDBACK,
          test_type=api.TestType.TWO_SIDED,
          log_estimate=0.125,
          expected_estimate=np.exp(0.125) - 1.0,
          expected_lower=np.exp(-0.245) - 1.0,
          expected_upper=np.exp(0.445) - 1.0,
          expected_p_value=0.9,
      ),
      dict(
          testcase_name='signed_go_dark_two_sided',
          use_absolute_placebo_statistics=False,
          experiment_type=api.ExperimentType.GO_DARK,
          test_type=api.TestType.TWO_SIDED,
          log_estimate=-0.125,
          expected_estimate=1.0 - np.exp(-0.125),
          expected_lower=1.0 - np.exp(0.195),
          expected_upper=1.0 - np.exp(-0.495),
          expected_p_value=0.7,
      ),
      dict(
          testcase_name='signed_holdback_one_sided',
          use_absolute_placebo_statistics=False,
          experiment_type=api.ExperimentType.HOLDBACK,
          test_type=api.TestType.ONE_SIDED,
          log_estimate=0.125,
          expected_estimate=np.exp(0.125) - 1.0,
          expected_lower=np.exp(-0.165) - 1.0,
          expected_upper=np.inf,
          expected_p_value=0.5,
      ),
      dict(
          testcase_name='signed_go_dark_one_sided',
          use_absolute_placebo_statistics=False,
          experiment_type=api.ExperimentType.GO_DARK,
          test_type=api.TestType.ONE_SIDED,
          log_estimate=-0.125,
          expected_estimate=1.0 - np.exp(-0.125),
          expected_lower=1.0 - np.exp(0.115),
          expected_upper=np.inf,
          expected_p_value=0.4,
      ),
      dict(
          testcase_name='absolute_holdback_two_sided',
          use_absolute_placebo_statistics=True,
          experiment_type=api.ExperimentType.HOLDBACK,
          test_type=api.TestType.TWO_SIDED,
          log_estimate=0.125,
          expected_estimate=np.exp(0.125) - 1.0,
          expected_lower=np.exp(-0.275) - 1.0,
          expected_upper=np.exp(0.525) - 1.0,
          expected_p_value=0.8,
      ),
      dict(
          testcase_name='absolute_go_dark_two_sided',
          use_absolute_placebo_statistics=True,
          experiment_type=api.ExperimentType.GO_DARK,
          test_type=api.TestType.TWO_SIDED,
          log_estimate=-0.125,
          expected_estimate=1.0 - np.exp(-0.125),
          expected_lower=1.0 - np.exp(0.275),
          expected_upper=1.0 - np.exp(-0.525),
          expected_p_value=0.8,
      ),
      dict(
          testcase_name='absolute_holdback_one_sided',
          use_absolute_placebo_statistics=True,
          experiment_type=api.ExperimentType.HOLDBACK,
          test_type=api.TestType.ONE_SIDED,
          log_estimate=0.125,
          expected_estimate=np.exp(0.125) - 1.0,
          expected_lower=np.exp(-0.175) - 1.0,
          expected_upper=np.inf,
          expected_p_value=0.4,
      ),
      dict(
          testcase_name='absolute_go_dark_one_sided',
          use_absolute_placebo_statistics=True,
          experiment_type=api.ExperimentType.GO_DARK,
          test_type=api.TestType.ONE_SIDED,
          log_estimate=-0.125,
          expected_estimate=1.0 - np.exp(-0.125),
          expected_lower=1.0 - np.exp(0.175),
          expected_upper=np.inf,
          expected_p_value=0.4,
      ),
      dict(
          testcase_name='absolute_go_dark_one_sided_increase',
          use_absolute_placebo_statistics=True,
          experiment_type=api.ExperimentType.GO_DARK,
          test_type=api.TestType.ONE_SIDED,
          log_estimate=0.125,
          expected_estimate=1.0 - np.exp(0.125),
          expected_lower=1.0 - np.exp(0.425),
          expected_upper=np.inf,
          expected_p_value=0.65,
      ),
  )
  def test_get_percent_lift_bounds_and_p_value(
      self,
      use_absolute_placebo_statistics,
      experiment_type,
      test_type,
      log_estimate,
      expected_estimate,
      expected_lower,
      expected_upper,
      expected_p_value,
  ):
    # Placebo log effects are t * log_rmse, so the placebo t-statistics are t,
    # and t_obs = log_estimate / log_rmse = +/-2.5. With alpha = 0.2:
    # Signed: the log bounds use the interpolated quantiles of t at levels 0.1,
    # 0.2, 0.8 and 0.9: -6.4, -4.8, 5.8 and 7.4.
    # Absolute: the two-sided p-value is (1 + 7) / (1 + 9) = 0.8. Two-sided
    # bounds use k = 8 and |t|_(8) = 8, so the log CI is log_estimate -/+
    # 8 * 0.05. One-sided bounds use k = 6 and |t|_(6) = 6 on the side opposite
    # to the tested direction. The one-sided p-value is 0.8 / 2 if the estimate
    # is in the tested direction and 1 - 7 / 20 otherwise, since 7 of the |t_j|
    # exceed 2.5.
    log_rmse = 0.05
    t = jnp.array(_PLACEBO_T)
    estimate = util.get_percent_lift(
        y_test=jnp.exp(jnp.array(log_estimate)),
        y_pred=jnp.array(1.0),
        log_rmse=jnp.array(log_rmse),
        y_test_placebos=jnp.exp(t * log_rmse),
        y_pred_placebos=jnp.ones_like(t),
        placebo_log_rmses=jnp.full_like(t, log_rmse),
        alpha=0.2,
        experiment_type=experiment_type,
        test_type=test_type,
        use_absolute_placebo_statistics=use_absolute_placebo_statistics,
    )

    self.assertAlmostEqual(estimate.point_estimate, expected_estimate, places=5)
    self.assertAlmostEqual(estimate.lower_bound, expected_lower, places=5)
    self.assertAlmostEqual(estimate.upper_bound, expected_upper, places=5)
    self.assertAlmostEqual(estimate.p_value, expected_p_value, places=5)


if __name__ == '__main__':
  absltest.main()
