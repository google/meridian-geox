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
specs, `create_card_html` renders a card through the Jinja templates next to
this module and `create_summary_html` wraps the cards in the page.

GeoX additions:

* Every spec carries an `audience`: `'all'` (both views), `'executive'`
  (Executive summary only) or `'technical'` (Full report only).
* Specs can be scoped to one design `candidate`; the page shows only the specs
  of the selected candidate.
* A few more specs (stats sections, callouts, text, buttons, rows and
  template-rendered blocks) and page-level helpers (charts and large boundary
  files embedded once, CSV export, notebook iframe).
"""

from collections.abc import Mapping, Sequence
import dataclasses
import json
import math
import os
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
class StatsSpec:
  """One tile of a stats row."""

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
    StatsSectionSpec,
    CalloutSpec,
    TemplateSpec,
]

_TEMPLATES = {
    StatsSectionSpec: 'stats_section.html.jinja',
    CalloutSpec: 'callout.html.jinja',
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
  if isinstance(spec, StatsSectionSpec):
    params['stats'] = _create_stats_htmls(template_env, spec.stats)
  return markupsafe.Markup(template.render(params))


class _Page:
  """Charts and shared resources of one report page."""

  def __init__(self):
    self.charts: dict[str, Any] = {}
    self.resources: dict[str, str] = {}


_PAGES: 'weakref.WeakKeyDictionary[jinja2.Environment, _Page]' = (
    weakref.WeakKeyDictionary()
)


def _page(template_env: jinja2.Environment) -> _Page:
  if template_env not in _PAGES:
    raise ValueError('Use create_template_env() to create the environment.')
  return _PAGES[template_env]


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
