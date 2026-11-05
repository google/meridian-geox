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

"""Functions for formatting the GeoX design and analysis reports.

Branched from meridian/templates/formatter.py. The Meridian spec pattern is
kept: reports describe each card with a `CardSpec` plus chart, table and stats
specs. Rendering the specs to HTML is added in a follow-up change.

GeoX additions:

* Every spec carries an `audience`: `'all'` (both views), `'executive'`
  (Executive summary only) or `'technical'` (Full report only).
* Specs can be scoped to one design `candidate`; the page shows only the specs
  of the selected candidate.
* A few more specs (stats sections, callouts, text, buttons, rows and
  template-rendered blocks) and output helpers (CSV export, script JSON).
"""

import base64
from collections.abc import Mapping, Sequence
import csv
import dataclasses
import io
import json
import math
import re
from typing import Any, Literal, Union

Audience = Literal['all', 'executive', 'technical']
Tone = Literal['neutral', 'info', 'success', 'warning', 'error']


@dataclasses.dataclass(frozen=True, kw_only=True)
class _Scoped:
  """View and candidate scoping shared by the GeoX specs."""

  audience: Audience = 'all'
  # Design ID this spec belongs to, or None if it is shared by all candidates.
  candidate: str | None = None


@dataclasses.dataclass(frozen=True)
class CardSpec:
  id: str
  title: str
  subtitle: str | None = None
  audience: Audience = 'all'
  # Short interaction cue shown under the title.
  cue: str | None = None


@dataclasses.dataclass(frozen=True)
class ChartSpec(_Scoped):
  # Element ID; leave empty to get a page-unique one.
  id: str
  chart_json: str
  description: str | None = None
  errors: Sequence[str] | None = None
  warnings: Sequence[str] | None = None
  infos: Sequence[str] | None = None
  # "How to read" text: a popover in the Full report, inline otherwise.
  reading_guide: str | None = None
  css_class: str = ''


@dataclasses.dataclass(frozen=True)
class TableSpec(_Scoped):
  id: str
  title: str
  column_headers: Sequence[str]
  row_values: Sequence[Sequence[str]]
  description: str | None = None
  errors: Sequence[str] | None = None
  warnings: Sequence[str] | None = None
  infos: Sequence[str] | None = None
  # Design ID per row; with `selectable`, clicking a row selects it.
  row_ids: Sequence[str] | None = None
  selectable: bool = False
  # Optional per-row tone used to tint the status (last) cell.
  row_tones: Sequence[Tone] | None = None
  # If set, the rows are collapsed behind a toggle with this label.
  collapsed_summary: str | None = None


@dataclasses.dataclass(frozen=True)
class StatsSpec:
  title: str
  stat: str
  delta: str | None = None
  # Muted confidence-interval line under the stat.
  ci: str | None = None
  detail: str | None = None
  tone: Tone = 'neutral'


@dataclasses.dataclass(frozen=True)
class StatsSectionSpec(_Scoped):
  """A row of stats inside a card."""

  stats: Sequence[StatsSpec]


@dataclasses.dataclass(frozen=True)
class CalloutSpec(_Scoped):
  """A highlighted message box."""

  text: str
  title: str | None = None
  tone: Tone = 'info'
  # (label, url) pairs shown after the text, for example documentation.
  links: Sequence[tuple[str, str]] = ()

  def __post_init__(self):
    for _, url in self.links:
      if not url.startswith(('https://', 'http://')):
        raise ValueError(f'Callout links must be http(s) URLs, got {url!r}.')


@dataclasses.dataclass(frozen=True)
class TextSpec(_Scoped):
  """A paragraph of plain text."""

  text: str


@dataclasses.dataclass(frozen=True)
class Button:
  """A button that downloads `payload` as `filename`.

  Attributes:
    label: Button text.
    payload: Text to download, for example CSV.
    filename: Download file name.
    mime: MIME type of the download.
  """

  label: str
  payload: str
  filename: str = 'download.csv'
  mime: str = 'text/csv'

  @property
  def href(self) -> str:
    encoded = base64.b64encode(self.payload.encode('utf-8')).decode('ascii')
    return f'data:{self.mime};charset=utf-8;base64,{encoded}'


@dataclasses.dataclass(frozen=True)
class ButtonsSpec(_Scoped):
  """A row of buttons."""

  buttons: Sequence[Button]


@dataclasses.dataclass(frozen=True)
class RowSpec(_Scoped):
  """Lays out child specs side by side."""

  specs: Sequence['Spec']


@dataclasses.dataclass(frozen=True)
class TemplateSpec(_Scoped):
  """A block rendered by one of the templates next to this module."""

  template: str
  params: Mapping[str, Any] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass(frozen=True)
class ChipSpec:
  """A label/value pair shown in the report header."""

  label: str
  value: str


Spec = Union[
    ChartSpec,
    TableSpec,
    StatsSectionSpec,
    CalloutSpec,
    TextSpec,
    ButtonsSpec,
    RowSpec,
    TemplateSpec,
]


_NUMBER = re.compile(r'^-?(\d+(\.\d*)?|\.\d+)([eE][-+]?\d+)?$')
# Characters that make spreadsheet apps evaluate a cell as a formula.
_FORMULA_PREFIXES = ('=', '+', '-', '@', '\t', '\r', '\n')


def without_formulas(value: Any) -> str:
  """Neutralises spreadsheet formulas in a CSV cell value.

  Numbers (including negative ones) are kept as they are.

  Args:
    value: The cell value.

  Returns:
    The value as text, prefixed with `'` if a spreadsheet would evaluate it.
  """
  text = '' if value is None else str(value)
  if _NUMBER.match(text):
    return text
  if text.startswith(_FORMULA_PREFIXES):
    return "'" + text
  return text


def to_csv(rows: Sequence[Mapping[str, Any]], columns: Sequence[str]) -> str:
  """Serialises rows to CSV text with formula neutralisation."""
  buffer = io.StringIO()
  writer = csv.writer(buffer, lineterminator='\n')
  writer.writerow([without_formulas(c) for c in columns])
  for row in rows:
    writer.writerow([without_formulas(row.get(c)) for c in columns])
  return buffer.getvalue()


def json_for_script(value: Any) -> str:
  """Serialises a value as JSON that is safe inside a `<script>` element."""
  text = json.dumps(_finite(value), separators=(',', ':'))
  return (
      text.replace('<', '\\u003c')
      .replace('>', '\\u003e')
      .replace('&', '\\u0026')
  )


def _finite(value: Any) -> Any:
  """Replaces NaN and infinities (invalid JSON) with None, recursively."""
  if isinstance(value, float):
    return value if math.isfinite(value) else None
  if isinstance(value, dict):
    return {k: _finite(v) for k, v in value.items()}
  if isinstance(value, (list, tuple)):
    return [_finite(v) for v in value]
  if hasattr(value, 'item'):  # NumPy scalars.
    return _finite(value.item())
  return value


