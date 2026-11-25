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

"""Specs and HTML rendering for the GeoX design and analysis reports.

Each card is described by a `CardSpec` and a list of block specs. Block specs
can be limited to one report view (`audience`) and to one design
(`candidate`). `create_card_html` renders a card through the Jinja templates
next to this module and `create_summary_html` wraps the cards in the page.
"""

import base64
from collections.abc import Mapping, Sequence
import csv
import dataclasses
import hashlib
import io
import json
import math
import os
import re
from typing import Any, Literal, Union
import weakref

import jinja2
import markupsafe

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
  """One card of the report."""

  id: str
  title: str
  subtitle: str | None = None
  audience: Audience = 'all'
  # Short interaction cue shown under the title.
  cue: str | None = None


@dataclasses.dataclass(frozen=True)
class ChartSpec(_Scoped):
  """An embedded Vega-Lite chart."""

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
  """A table, optionally with selectable rows."""

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

  def __post_init__(self):
    for name in ('row_ids', 'row_tones'):
      values = getattr(self, name)
      if values is not None and len(values) != len(self.row_values):
        raise ValueError(f'{name} must have one entry per row.')
    if self.selectable and self.row_ids is None:
      raise ValueError('Selectable tables need row_ids.')


@dataclasses.dataclass(frozen=True)
class StatsSpec:
  """One tile of a stats row."""

  title: str
  stat: str
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
    object.__setattr__(self, 'links', tuple(self.links))
    for _, url in self.links:
      if not url.startswith(('https://', 'http://')):
        raise ValueError(f'Callout links must be http(s) URLs, got {url!r}.')


@dataclasses.dataclass(frozen=True)
class TextSpec(_Scoped):
  """A paragraph of plain text."""

  text: str
  # Renders as a smaller, muted note instead of body text.
  detail: bool = False
  # (label, url) documentation links shown after the text.
  links: tuple[tuple[str, str], ...] = ()


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
    """Data URI that downloads the payload."""
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

  def __post_init__(self):
    reserved = sorted({'audience', 'candidate'} & set(self.params))
    if reserved:
      raise ValueError(f'TemplateSpec params use reserved keys: {reserved}.')


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

_TEMPLATES = {
    ChartSpec: 'chart.html.jinja',
    TableSpec: 'table.html.jinja',
    StatsSectionSpec: 'stats_section.html.jinja',
    CalloutSpec: 'callout.html.jinja',
    TextSpec: 'text.html.jinja',
    ButtonsSpec: 'buttons.html.jinja',
    RowSpec: 'row.html.jinja',
}

_template_loader = jinja2.FileSystemLoader(
    os.path.abspath(os.path.dirname(__file__))
)


def create_template_env() -> jinja2.Environment:
  """Creates a Jinja2 template environment for one report page."""
  env = jinja2.Environment(
      loader=_template_loader,
      autoescape=jinja2.select_autoescape(
          enabled_extensions=('html', 'htm', 'xml', 'jinja')
      ),
      trim_blocks=True,
      lstrip_blocks=True,
  )
  _PAGES[env] = _Page()
  return env


def create_summary_html(
    template_env: jinja2.Environment,
    title: str,
    cards: Sequence[str],
    *,
    kind: Literal['design', 'analysis'] = 'design',
    subtitle: str | None = None,
    chips: Sequence[ChipSpec] = (),
    default_candidate: str | None = None,
) -> str:
  """Creates the HTML for the summary page.

  Args:
    template_env: The environment the cards were rendered with.
    title: Page title.
    cards: Card HTML snippets from `create_card_html`.
    kind: Report kind, shown above the title.
    subtitle: Optional line under the title.
    chips: Header chips.
    default_candidate: Design ID selected when the page opens, if any.

  Returns:
    A self-contained HTML page.
  """
  page = _page(template_env)
  return template_env.get_template('summary.html.jinja').render(
      title=title,
      cards=[markupsafe.Markup(card) for card in cards],
      kind=kind,
      subtitle=subtitle,
      chips=chips,
      chart_specs=markupsafe.Markup(json_for_script(page.charts)),
      resources={
          key: markupsafe.Markup(json_for_script(value.removeprefix('\ufeff')))
          for key, value in page.resources.items()
      },
      default_candidate=default_candidate or '',
  )


def create_card_html(
    template_env: jinja2.Environment,
    card_spec: CardSpec,
    specs: Sequence[Spec | None] | None = None,
) -> str:
  """Creates a card's HTML from its card spec and the blocks inside it.

  Args:
    template_env: The page's template environment.
    card_spec: The card.
    specs: Blocks shown in order, any `Spec` type; None entries are
      skipped.

  Returns:
    The card HTML.
  """
  _page(template_env)  # Fails early for environments of other pages.
  card_params: dict[str, Any] = dataclasses.asdict(card_spec)
  blocks = [spec for spec in specs or () if spec is not None]
  card_params['blocks'] = (
      [_create_spec_html(template_env, b) for b in blocks] if blocks else None
  )
  return template_env.get_template('card.html.jinja').render(card_params)


def _create_stats_htmls(
    template_env: jinja2.Environment, specs: Sequence[StatsSpec]
) -> Sequence[str]:
  """Creates a list of stats HTML snippets given a list of stats specs."""
  return [
      markupsafe.Markup(
          template_env.get_template('stats.html.jinja').render(
              dataclasses.asdict(spec)
          )
      )
      for spec in specs
  ]


def _create_spec_html(template_env: jinja2.Environment, spec: Spec) -> str:
  """Renders one spec through its template."""
  if isinstance(spec, TemplateSpec):
    return markupsafe.Markup(
        template_env.get_template(spec.template).render(
            **spec.params, audience=spec.audience, candidate=spec.candidate
        )
    )
  template = template_env.get_template(_TEMPLATES[type(spec)])
  params: dict[str, Any] = {
      field.name: getattr(spec, field.name)
      for field in dataclasses.fields(spec)
  }
  if isinstance(spec, ChartSpec):
    params['id'] = _page(template_env).register_chart(spec)
  elif isinstance(spec, StatsSectionSpec):
    params['stats'] = _create_stats_htmls(template_env, spec.stats)
  elif isinstance(spec, RowSpec):
    params['specs'] = [
        _create_spec_html(template_env, child) for child in spec.specs
    ]
  return markupsafe.Markup(template.render(params))


# ----------------------------------------------------------------- GeoX page

# Base64 data URIs at least this long are hoisted out of the chart specs and
# embedded once per page.
HOIST_MIN_CHARS = 1024
RESOURCE_PREFIX = 'geox-resource:'


class _Page:
  """Charts and shared resources of one report page."""

  def __init__(self):
    self.charts: dict[str, Any] = {}
    self.resources: dict[str, str] = {}

  def register_chart(self, spec: ChartSpec) -> str:
    """Stores a chart and returns its page-unique element ID."""
    chart_id = spec.id
    if chart_id in self.charts:
      raise ValueError(f'Duplicate chart id {chart_id!r}.')
    if not chart_id:
      n = len(self.charts) + 1
      while f'geox-chart-{n}' in self.charts:
        n += 1
      chart_id = f'geox-chart-{n}'
    self.charts[chart_id] = self.hoist(json.loads(spec.chart_json))
    return chart_id

  def hoist(self, value: Any, key: str | None = None) -> Any:
    """Returns `value` with large `url` data URIs replaced by pointers.

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
      digest = hashlib.sha256(value.encode('utf-8')).hexdigest()[:16]
      self.resources[digest] = value
      return RESOURCE_PREFIX + digest
    return value


_PAGES: 'weakref.WeakKeyDictionary[jinja2.Environment, _Page]' = (
    weakref.WeakKeyDictionary()
)


def _page(template_env: jinja2.Environment) -> _Page:
  if template_env not in _PAGES:
    raise ValueError('Use create_template_env() to create the environment.')
  return _PAGES[template_env]


_NUMBER = re.compile(r'-?(\d+(\.\d*)?|\.\d+)([eE][-+]?\d+)?')
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
  if _NUMBER.fullmatch(text):
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
      f'<iframe srcdoc="{markupsafe.escape(html)}" sandbox="{_IFRAME_SANDBOX}" '
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
