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

"""Registry and dispatch for GeoX counterfactual methodologies."""

from collections.abc import Callable
from typing import TypeVar
from meridian_geox import api
from meridian_geox.methodology import base

MethodologyClsT = TypeVar(
    'MethodologyClsT', bound=type[base.CounterfactualMethodology]
)

_METHODOLOGY_REGISTRY: dict[api.Methodology, base.CounterfactualMethodology] = (
    {}
)


def register_methodology(
    methodology_enum: api.Methodology,
) -> Callable[[MethodologyClsT], MethodologyClsT]:
  """Decorator to register a singleton CounterfactualMethodology instance."""

  def _decorator(cls: MethodologyClsT) -> MethodologyClsT:
    if methodology_enum in _METHODOLOGY_REGISTRY:
      raise ValueError(f'Methodology {methodology_enum} is already registered.')
    _METHODOLOGY_REGISTRY[methodology_enum] = cls()
    return cls

  return _decorator


def is_methodology_registered(methodology_enum: api.Methodology) -> bool:
  """Returns True if `methodology_enum` has a registered implementation."""
  return methodology_enum in _METHODOLOGY_REGISTRY


def get_methodology(
    methodology_enum: api.Methodology,
) -> base.CounterfactualMethodology:
  """Returns the registered singleton CounterfactualMethodology instance."""
  if methodology_enum not in _METHODOLOGY_REGISTRY:
    supported = ', '.join(m.name for m in _METHODOLOGY_REGISTRY)
    raise ValueError(
        f'Unsupported methodology: {methodology_enum}. Supported methodologies:'
        f' [{supported}].'
    )
  return _METHODOLOGY_REGISTRY[methodology_enum]
