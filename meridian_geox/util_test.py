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


if __name__ == '__main__':
  absltest.main()
