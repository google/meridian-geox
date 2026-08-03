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

"""Tests for GeoX utility functions."""

from absl.testing import absltest
from absl.testing import parameterized
import jax.numpy as jnp
from meridian_geox import util


class UtilTest(parameterized.TestCase):

  def test_cell_id_from_cell_name_success(self):
    self.assertEqual(util.cell_id_from_cell_name('cell_3'), 3)
    self.assertEqual(util.cell_id_from_cell_name('cell_123'), 123)

  @parameterized.named_parameters(
      ('zero', 'cell_0'),
      ('negative', 'cell_-5'),
      ('non_integer', 'cell_abc'),
      ('wrong_prefix', 'cell1'),
      ('other_string', 'group_1'),
  )
  def test_cell_id_from_cell_name_value_error(self, cell_name):
    with self.assertRaises(ValueError):
      util.cell_id_from_cell_name(cell_name)

  def test_filter_by_r2_1d_pass(self):
    r2_scores = jnp.array([0.9, 0.8, 0.7])
    mask = util.filter_by_r2(
        r2_scores=r2_scores,
        min_r2=0.8,
        min_count_error=1,
        min_count_warning=2,
        error_message='error',
        warning_message='warning',
    )
    self.assertEqual(mask.tolist(), [True, True, False])

  def test_filter_by_r2_2d_pass(self):
    # 2 candidates, 2 cells
    r2_scores = jnp.array([[0.9, 0.8], [0.7, 0.9]])
    # Candidate 1 passes (both >= 0.8)
    # Candidate 2 fails (0.7 < 0.8)
    mask = util.filter_by_r2(
        r2_scores=r2_scores,
        min_r2=0.8,
        min_count_error=1,
        min_count_warning=1,
        error_message='error',
        warning_message='warning',
        cell_names=['cell_1', 'cell_2'],
    )
    self.assertEqual(mask.tolist(), [True, False])

  def test_filter_by_r2_too_many_dimensions_raises_value_error(self):
    r2_scores = jnp.zeros((1, 2, 3))
    with self.assertRaisesRegex(ValueError, 'r2_scores must be 1D or 2D'):
      util.filter_by_r2(
          r2_scores=r2_scores,
          min_r2=0.8,
          min_count_error=1,
          min_count_warning=1,
          error_message='error',
          warning_message='warning',
      )


if __name__ == '__main__':
  absltest.main()
