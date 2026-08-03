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
import jax.numpy as jnp
from meridian_geox.methodology import util
import numpy as np


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


if __name__ == '__main__':
  absltest.main()
