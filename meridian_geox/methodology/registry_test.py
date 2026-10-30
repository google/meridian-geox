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
import jax.numpy as jnp
from meridian_geox import api
from meridian_geox.methodology import base
from meridian_geox.methodology import registry


class _DummyMethodology(base.CounterfactualMethodology):
  """Minimal concrete methodology for testing registry dispatch."""

  def _fit_and_predict_single(
      self,
      data_train: jnp.ndarray,
      data_eval: jnp.ndarray,
      treatment_group_mask: jnp.ndarray,
      control_group_mask: jnp.ndarray,
  ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    raise NotImplementedError()


class RegistryTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    saved_registry = dict(registry._METHODOLOGY_REGISTRY)
    registry._METHODOLOGY_REGISTRY.clear()
    self.addCleanup(registry._METHODOLOGY_REGISTRY.update, saved_registry)
    self.addCleanup(registry._METHODOLOGY_REGISTRY.clear)

  def test_register_and_get_methodology_returns_singleton(self):
    registry.register_methodology(api.Methodology.TBR)(_DummyMethodology)

    self.assertTrue(registry.is_methodology_registered(api.Methodology.TBR))
    instance = registry.get_methodology(api.Methodology.TBR)
    self.assertIsInstance(instance, _DummyMethodology)
    self.assertIs(instance, registry.get_methodology(api.Methodology.TBR))

  def test_get_unregistered_methodology_raises_value_error(self):
    registry.register_methodology(api.Methodology.TBR)(_DummyMethodology)

    self.assertFalse(registry.is_methodology_registered(api.Methodology.SDID))
    with self.assertRaisesRegex(
        ValueError,
        r'Unsupported methodology: Methodology\.SDID\. Supported'
        r' methodologies: \[TBR\]\.',
    ):
      registry.get_methodology(api.Methodology.SDID)

  def test_register_duplicate_methodology_raises_value_error(self):
    registry.register_methodology(api.Methodology.TBR)(_DummyMethodology)

    with self.assertRaisesRegex(ValueError, 'already registered'):
      registry.register_methodology(api.Methodology.TBR)(_DummyMethodology)


if __name__ == '__main__':
  absltest.main()
