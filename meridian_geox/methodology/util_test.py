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

import logging
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import jax.numpy as jnp
from meridian_geox import api
from meridian_geox.methodology import util
import numpy as np


class UtilTest(parameterized.TestCase):

  def test_compute_studentized_p_value_two_sided(self):
    t_placebo = jnp.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    estimate = 1.5
    rmse = 1.0
    # t_obs = 1.5. Extreme left <= 1.5: 4 (-2, -1, 0, 1).
    # Extreme right >= 1.5: 1 (2).
    # n_extreme = 2 * min(4, 1) = 2.
    # p_value = (1 + 2) / (1 + 5) = 3 / 6 = 0.5.
    expected = jnp.array(0.5)
    result = util.compute_studentized_p_value(
        estimate, rmse, t_placebo, test_type=api.TestType.TWO_SIDED
    )
    np.testing.assert_allclose(result, expected, atol=1e-6)

  def test_compute_studentized_p_value_one_sided(self):
    t_placebo = jnp.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    estimate = 1.5
    rmse = 1.0
    # t_obs = 1.5. Extreme right >= 1.5: 1 (2).
    # p_value = (1 + 1) / (1 + 5) = 2 / 6 = 1/3.
    expected = jnp.array(1 / 3)
    result = util.compute_studentized_p_value(
        estimate, rmse, t_placebo, test_type=api.TestType.ONE_SIDED
    )
    np.testing.assert_allclose(result, expected, atol=1e-6)

  def test_compute_cis_two_sided(self):
    t_placebo = jnp.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    estimate = 0.0
    rmse = 1.0
    alpha = 0.4  # quantile 0.2 and 0.8
    # quantile(0.8) = 1.2, quantile(0.2) = -1.2
    # lower = 0 - 1.2 = -1.2, upper = 0 - (-1.2) = 1.2
    lower, upper = util.compute_cis(
        estimate, rmse, t_placebo, alpha, test_type=api.TestType.TWO_SIDED
    )
    np.testing.assert_allclose(lower, -1.2, atol=1e-5)
    np.testing.assert_allclose(upper, 1.2, atol=1e-5)

  def test_compute_regularized_log_ratio_unscaled(self):
    y_num = jnp.array([10.0, 0.0])
    y_den = jnp.array([20.0, 5.0])
    # For element 1: num=10, den=20. scale=20. threshold=0.02.
    # ratio=10/20=0.5. log(0.5) ~ -0.693147
    # For element 2: num=0, den=5. scale=5. threshold=0.005.
    # num clipped to 0.005. ratio=0.005/5=0.001. log(0.001) ~ -6.907755
    expected = jnp.array([np.log(0.5), np.log(0.001)])
    result = util.compute_regularized_log_ratio(y_num, y_den)
    np.testing.assert_allclose(result, expected, atol=1e-6)

  def test_compute_regularized_log_ratio_scaled(self):
    y_num = jnp.array([0.0])
    y_den = jnp.array([0.0])
    # Provide a global scale
    scale = jnp.array(100.0)
    # threshold = 1e-3 * 100 = 0.1
    # Both are clipped to 0.1, log ratio is log(1) = 0.0
    expected = jnp.array([0.0])
    result = util.compute_regularized_log_ratio(y_num, y_den, scale=scale)
    np.testing.assert_allclose(result, expected, atol=1e-6)

  @mock.patch.object(logging, 'warning')
  def test_check_and_warn_placebo_bias_triggered(self, mock_warning):
    # Mean = 3.0, std = ~0.0. Bias is very high.
    placebo_effects = jnp.array([[3.0, 3.0, 3.0]])
    res = util.check_and_warn_placebo_bias(placebo_effects, metric_name='lift')
    mock_warning.assert_called_once()
    self.assertEqual(res['metric'], 'Placebo distribution bias (lift)')
    self.assertEqual(res['threshold'], '[-0.0, 0.0]')
    self.assertIn('deviates significantly', res['message'])

  @mock.patch.object(logging, 'warning')
  def test_check_and_warn_placebo_bias_not_triggered(self, mock_warning):
    # Mean = 0.0, std = > 0.
    placebo_effects = jnp.array([[-1.0, 0.0, 1.0]])
    res = util.check_and_warn_placebo_bias(placebo_effects, metric_name='lift')
    mock_warning.assert_not_called()
    self.assertEqual(res['message'], '')

  @mock.patch.object(logging, 'warning')
  def test_check_and_warn_ci_triggered(self, mock_warning):
    lower_ci = jnp.array([1.0])
    estimate = jnp.array([0.0])
    upper_ci = jnp.array([2.0])
    res = util.check_and_warn_ci(
        lower_ci, estimate, upper_ci, metric_name='lift'
    )
    mock_warning.assert_called_once()
    self.assertEqual(res['threshold'], '[1.0, 2.0]')
    self.assertIn('does not contain the point estimate', res['message'])

  @mock.patch.object(logging, 'warning')
  def test_check_and_warn_ci_not_triggered(self, mock_warning):
    lower_ci = jnp.array([0.0])
    estimate = jnp.array([1.0])
    upper_ci = jnp.array([2.0])
    res = util.check_and_warn_ci(
        lower_ci, estimate, upper_ci, metric_name='lift'
    )
    mock_warning.assert_not_called()
    self.assertEqual(res['message'], '')


if __name__ == '__main__':
  absltest.main()
