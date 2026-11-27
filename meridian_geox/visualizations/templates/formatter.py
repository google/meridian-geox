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

"""Specs for the GeoX design and analysis HTML reports.

Each card is described by a `CardSpec` and a list of block specs. Block specs
can be limited to one report view (`report_view`).
"""

from collections.abc import Mapping, Sequence
import dataclasses
from typing import Any, Literal

# Executive view hides blocks shown only in the Full report
# (report_view='full_only').
ReportView = Literal['all_views', 'full_only']
Tone = Literal['neutral', 'success', 'warning', 'error']


@dataclasses.dataclass(frozen=True, kw_only=True)
class _Scoped:
  """Report view scoping shared by the GeoX specs.

  Attributes:
    report_view: Report view that shows the spec: 'all_views', or
      'full_only' (Full report only).
  """

  report_view: ReportView = 'all_views'


@dataclasses.dataclass(frozen=True)
class CardSpec:
  """One card of the report.

  Attributes:
    id: Element ID of the card.
    title: Card title.
    subtitle: Optional line under the title.
  """

  id: str
  title: str
  subtitle: str | None = None


@dataclasses.dataclass(frozen=True)
class StatsSpec:
  """One tile of a stats row.

  Attributes:
    title: Tile title.
    stat: Headline value.
    note: Small muted line under the stat (above the detail), for example
      a formula, legend or confidence interval.
    detail: Detail line under the stat.
    tone: Colour tone of the tile.
  """

  title: str
  stat: str
  note: str | None = None
  detail: str | None = None
  tone: Tone = 'neutral'


@dataclasses.dataclass(frozen=True)
class StatsSectionSpec(_Scoped):
  """A row of stats inside a card.

  Attributes:
    stats: Tiles of the row.
  """

  stats: Sequence[StatsSpec]


@dataclasses.dataclass(frozen=True)
class CustomBlockSpec(_Scoped):
  """A one-off block rendered from a template next to this module with params.

  For example the option banner, and in later cards the launch steps,
  how-it-works and snippet blocks.

  Attributes:
    template: File name of the template.
    params: Template variables.
  """

  template: str
  params: Mapping[str, Any] = dataclasses.field(default_factory=dict)

  def __post_init__(self):
    # render() passes `report_view` as a keyword argument next to the params,
    # so a param with that name would collide.
    reserved = sorted({'report_view'} & set(self.params))
    if reserved:
      raise ValueError(f'CustomBlockSpec params use reserved keys: {reserved}.')


@dataclasses.dataclass(frozen=True)
class ChipSpec:
  """A label/value pair shown in the report header.

  Attributes:
    label: Chip label.
    value: Chip value.
  """

  label: str
  value: str
