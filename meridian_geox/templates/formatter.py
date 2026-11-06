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

"""Typed report specs and HTML rendering for the GeoX reports.

Reports are assembled from `Card`s that hold typed blocks (text, tiles, tables,
charts, buttons). `render_report` turns them into one self-contained HTML page
through the Jinja templates next to this module.

Every card and block carries an `audience`:

* `'all'`: shown in both the Executive summary and the Full report.
* `'executive'`: shown only in the Executive summary (for example inline
  reading guides).
* `'technical'`: shown only in the Full report.

Blocks can also be scoped to one design `candidate`; the page script shows only
the blocks of the currently selected candidate.
"""

import base64
import csv
import dataclasses
import hashlib
import io
import json
import math
import os
import re
from typing import Any, Literal, Optional, Sequence, Union

import altair as alt
import jinja2
import markupsafe
from meridian_geox import charts

Audience = Literal['all', 'executive', 'technical']
Tone = Literal['neutral', 'info', 'success', 'warning', 'error']

# Base64 data URIs at least this long are hoisted out of the chart specs and
# embedded once per page.
HOIST_MIN_CHARS = 1024
RESOURCE_PREFIX = 'geox-resource:'

_TEMPLATE_DIR = os.path.dirname(__file__)

# Characters that make spreadsheet apps evaluate a cell as a formula.
_FORMULA_PREFIXES = ('=', '+', '-', '@', '\t', '\r', '\n')


def escape(text: Any) -> str:
  """HTML-escapes a value for interpolation into a trusted HTML fragment."""
  return str(markupsafe.escape('' if text is None else str(text)))


@dataclasses.dataclass
class Block:
  """Base class for card blocks."""

  audience: Audience = dataclasses.field(default='all', kw_only=True)
  # Design ID this block belongs to, or None if it is shared by all candidates.
  candidate: Optional[str] = dataclasses.field(default=None, kw_only=True)
  kind: str = dataclasses.field(default='block', init=False)


@dataclasses.dataclass
class Text(Block):
  """A paragraph of plain text."""

  text: str
  emphasis: bool = False

  def __post_init__(self):
    self.kind = 'text'


@dataclasses.dataclass
class Callout(Block):
  """A highlighted message box."""

  text: str
  title: Optional[str] = None
  tone: Tone = 'info'
  # (label, url) pairs shown after the text, for example documentation.
  links: Sequence[tuple[str, str]] = ()

  def __post_init__(self):
    self.kind = 'callout'
    for _, url in self.links:
      if not url.startswith(('https://', 'http://')):
        raise ValueError(f'Callout links must be http(s) URLs, got {url!r}.')


@dataclasses.dataclass
class Tile:
  """A single headline metric."""

  label: str
  value: str
  detail: Optional[str] = None
  # Muted confidence-interval line under the value.
  ci: Optional[str] = None
  tone: Tone = 'neutral'


@dataclasses.dataclass
class Tiles(Block):
  """A row of headline metric tiles."""

  tiles: Sequence[Tile]

  def __post_init__(self):
    self.kind = 'tiles'


@dataclasses.dataclass
class Table(Block):
  """A data table.

  If `row_ids` is set, each row carries a design ID; with `selectable=True`,
  clicking a row selects that candidate on the page.
  """

  columns: Sequence[str]
  rows: Sequence[Sequence[str]]
  row_ids: Optional[Sequence[str]] = None
  selectable: bool = False
  caption: Optional[str] = None
  # Optional per-row tone used to tint status cells.
  row_tones: Optional[Sequence[Tone]] = None

  def __post_init__(self):
    self.kind = 'table'


@dataclasses.dataclass
class Chart(Block):
  """An Altair chart, rendered with vega-embed."""

  chart: Union[alt.TopLevelMixin, dict[str, Any]]
  framing: Optional[charts.Framing] = None
  css_class: str = ''

  def __post_init__(self):
    self.kind = 'chart'


@dataclasses.dataclass
class Button:
  """A download or copy button.

  Attributes:
    label: Button text.
    action: `'download'` downloads `payload` as `filename`; `'copy'` copies
      `payload` to the clipboard.
    payload: CSV text to download, or text to copy.
    filename: Download file name.
    mime: MIME type of the download.
  """

  label: str
  action: Literal['download', 'copy']
  payload: str
  filename: str = 'download.csv'
  mime: str = 'text/csv'

  @property
  def href(self) -> str:
    encoded = base64.b64encode(self.payload.encode('utf-8')).decode('ascii')
    return f'data:{self.mime};charset=utf-8;base64,{encoded}'


@dataclasses.dataclass
class Buttons(Block):
  """A row of buttons."""

  buttons: Sequence[Button]

  def __post_init__(self):
    self.kind = 'buttons'


@dataclasses.dataclass
class Html(Block):
  """A trusted HTML fragment.

  Every user-supplied string inside `html` must already be escaped with
  `escape`.
  """

  html: str

  def __post_init__(self):
    self.kind = 'html'


@dataclasses.dataclass
class Row(Block):
  """Lays out child blocks side by side."""

  blocks: Sequence[Block]

  def __post_init__(self):
    self.kind = 'row'


@dataclasses.dataclass
class Card:
  """A report card: a titled section made of blocks."""

  card_id: str
  title: str
  blocks: Sequence[Block]
  subtitle: Optional[str] = None
  audience: Audience = 'all'
  # Short interaction cue shown under the title.
  cue: Optional[str] = None


@dataclasses.dataclass
class Chip:
  """A label/value pair shown in the report header."""

  label: str
  value: str


@dataclasses.dataclass
class Report:
  """A full HTML report."""

  title: str
  kind: Literal['design', 'analysis']
  cards: Sequence[Card]
  subtitle: Optional[str] = None
  chips: Sequence[Chip] = ()
  # Design ID selected when the page opens, if the report has candidates.
  default_candidate: Optional[str] = None


_NUMBER = re.compile(r'^-?(\d+(\.\d*)?|\.\d+)([eE][-+]?\d+)?$')


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


def to_csv(rows: Sequence[dict[str, Any]], columns: Sequence[str]) -> str:
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
  if hasattr(value, 'item') and not isinstance(value, (str, bytes)):
    try:
      return _finite(value.item())
    except (TypeError, ValueError):
      return value
  return value


def strip_bom(text: str) -> str:
  """Removes a leading UTF-8 byte order mark."""
  return text[1:] if text.startswith('\ufeff') else text


class ResourceStore:
  """Collects large data URIs so each is embedded once per page."""

  def __init__(self):
    self.resources: dict[str, str] = {}

  def hoist(self, value: Any, key: Optional[str] = None) -> Any:
    """Returns `value` with large data URIs replaced by resource pointers.

    Only `url` properties are hoisted, since the page resolves pointers there.

    Args:
      value: A chart spec (or part of one).
      key: The property name that holds `value`.

    Returns:
      The spec with large `url` data URIs replaced.
    """
    if isinstance(value, dict):
      return {k: self.hoist(v, k) for k, v in value.items()}
    if isinstance(value, list):
      return [self.hoist(v) for v in value]
    if (
        key == 'url'
        and isinstance(value, str)
        and value.startswith('data:')
        and len(value) >= HOIST_MIN_CHARS
    ):
      key = hashlib.sha256(value.encode('utf-8')).hexdigest()[:16]
      self.resources[key] = value
      return RESOURCE_PREFIX + key
    return value


def chart_to_dict(chart: Union[alt.TopLevelMixin, dict[str, Any]]) -> dict[
    str, Any
]:
  """Converts an Altair chart to a validated Vega-Lite dict."""
  if isinstance(chart, dict):
    return chart
  return charts.configure(chart).to_dict()


def _non_empty(blocks: Sequence[Optional[Block]]) -> list[Block]:
  return [b for b in blocks if b is not None]


def create_template_env() -> jinja2.Environment:
  """Creates the Jinja environment with autoescaping enabled everywhere."""
  env = jinja2.Environment(
      loader=jinja2.FileSystemLoader(_TEMPLATE_DIR),
      autoescape=jinja2.select_autoescape(
          default=True, default_for_string=True
      ),
      trim_blocks=True,
      lstrip_blocks=True,
  )
  return env


class _ChartRegistry:
  """Assigns element IDs to charts and collects their specs."""

  def __init__(self, store: ResourceStore):
    self._store = store
    self.specs: dict[str, dict[str, Any]] = {}

  def register(self, block: Chart) -> str:
    chart_id = f'geox-chart-{len(self.specs) + 1}'
    self.specs[chart_id] = self._store.hoist(chart_to_dict(block.chart))
    return chart_id


def render_report(report: Report) -> str:
  """Renders a report to a self-contained HTML string."""
  env = create_template_env()
  store = ResourceStore()
  registry = _ChartRegistry(store)
  card_template = env.get_template('card.html.jinja')
  rendered_cards = [
      markupsafe.Markup(
          card_template.render(
              card=card,
              blocks=_non_empty(card.blocks),
              register_chart=registry.register,
          )
      )
      for card in report.cards
  ]
  summary = env.get_template('summary.html.jinja')
  return summary.render(
      report=report,
      cards=rendered_cards,
      chart_specs=markupsafe.Markup(json_for_script(registry.specs)),
      resources={
          key: markupsafe.Markup(json_for_script(strip_bom(value)))
          for key, value in store.resources.items()
      },
      default_candidate=report.default_candidate or '',
      style=markupsafe.Markup(_read_template_file('style.css')),
  )


def _read_template_file(name: str) -> str:
  with open(os.path.join(_TEMPLATE_DIR, name), encoding='utf-8') as f:
    return f.read()


# Sandbox for reports embedded in notebook outputs: scripts and downloads run
# inside the frame, which cannot reach the notebook page.
_IFRAME_SANDBOX = (
    'allow-scripts allow-downloads allow-popups allow-modals '
    'allow-popups-to-escape-sandbox'
)


def iframe(html: str, height: int = 1100) -> str:
  """Wraps a full report page in an `<iframe srcdoc>` for notebook display.

  Notebook frontends (Colab, Cider, Jupyter) drop the `<html>`/`<body>` tags of
  displayed HTML and may not run its scripts, which breaks charts, candidate
  selection and the Executive/Full toggle. Inside an iframe the page keeps its
  own document, exactly as when the saved file is opened in a browser.

  Args:
    html: The full report HTML.
    height: Frame height in pixels; the report scrolls inside the frame.

  Returns:
    An HTML fragment holding the iframe.
  """
  if height <= 0:
    raise ValueError(f'height must be positive, got {height}.')
  return (
      f'<iframe srcdoc="{escape(html)}" sandbox="{_IFRAME_SANDBOX}" '
      f'style="width:100%;height:{int(height)}px;border:0" '
      'title="Meridian GeoX report"></iframe>'
  )


def write_html(html: str, filename: str, filepath: str) -> str:
  """Writes `html` to `<filepath>/<filename>` and returns the local path."""
  if not filename:
    raise ValueError('filename must be non-empty.')
  os.makedirs(filepath or '.', exist_ok=True)
  path = os.path.join(filepath or '.', filename)
  with open(path, 'w', encoding='utf-8') as f:
    f.write(html)
  return path
