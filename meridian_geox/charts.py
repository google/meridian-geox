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

"""Shared chart styling for the GeoX HTML reports.

This module holds the palette, fonts and small helpers that every report chart
uses, so that the design and analysis reports look consistent.
"""

from collections.abc import Sequence
import dataclasses

import altair as alt

# Fonts.
FONT = 'Google Sans, Roboto, Arial, sans-serif'

# Neutral palette.
GREY_900 = '#202124'
GREY_700 = '#5F6368'
GREY_500 = '#9AA0A6'
GREY_300 = '#DADCE0'
GREY_100 = '#F1F3F4'

# Semantic palette.
BLUE = '#1A73E8'
GREEN = '#188038'
RED = '#D93025'
ORANGE = '#E8710A'
PURPLE = '#A142F4'
TEAL = '#12B5CB'
PINK = '#E52592'
BROWN = '#8D6E63'

CONTROL_COLOR = '#34A853'
EXCLUDED_COLOR = RED

# Treatment cell colours, in cell order.
# Green and red are reserved for control and excluded geos.
TREATMENT_COLORS = (BLUE, ORANGE, PURPLE, TEAL, PINK, BROWN)

CONTROL_LABEL = 'Control'
EXCLUDED_LABEL = 'Excluded'
NOT_IN_TEST_LABEL = 'Not in test'

DEFAULT_WIDTH = 520
DEFAULT_HEIGHT = 260


def treatment_color(index: int) -> str:
  """Returns the colour for the treatment cell at the given 0-based index."""
  if index < 0:
    raise ValueError(f'index must be non-negative, got {index}.')
  return TREATMENT_COLORS[index % len(TREATMENT_COLORS)]


def group_scale(treatment_labels: Sequence[str]) -> alt.Scale:
  """Returns a colour scale for treatment, control and excluded groups."""
  domain = list(treatment_labels) + [
      CONTROL_LABEL,
      EXCLUDED_LABEL,
      NOT_IN_TEST_LABEL,
  ]
  colors = [treatment_color(i) for i in range(len(treatment_labels))] + [
      CONTROL_COLOR,
      EXCLUDED_COLOR,
      GREY_100,
  ]
  return alt.Scale(domain=domain, range=colors)


@dataclasses.dataclass(frozen=True)
class Framing:
  """Title and reading guide for a chart.

  Attributes:
    title: Short chart title, phrased as the question it answers.
    subtitle: Optional one-line subtitle under the title.
    reading_guide: Plain-language "how to read this chart" text, shown inline
      in the Executive view and behind an "ⓘ How to read" button in the Full
      view.
    takeaway: One-sentence conclusion shown under the chart ("→ ...").

  `framed` uses only `title` and `subtitle`; `reading_guide` and `takeaway` are
  card metadata rendered by the HTML chart template.
  """

  title: str
  subtitle: str | None = None
  reading_guide: str | None = None
  takeaway: str | None = None


def framed(chart: alt.TopLevelMixin, framing: Framing) -> alt.TopLevelMixin:
  """Applies the title from a `Framing` to a chart."""
  title = alt.TitleParams(
      text=framing.title,
      subtitle=framing.subtitle or alt.Undefined,
      anchor='start',
      font=FONT,
      fontSize=14,
      fontWeight=600,
      color=GREY_900,
      subtitleColor=GREY_700,
      subtitleFont=FONT,
      offset=12,
  )
  return chart.properties(title=title)


def configure(chart: alt.TopLevelMixin) -> alt.TopLevelMixin:
  """Applies the shared GeoX look to a top-level chart."""
  return (
      chart.configure(font=FONT, background='transparent')
      .configure_view(stroke=None)
      .configure_axis(
          labelColor=GREY_700,
          titleColor=GREY_700,
          gridColor=GREY_100,
          domainColor=GREY_300,
          tickColor=GREY_300,
          labelFontSize=11,
          titleFontSize=12,
          titleFontWeight=500,
      )
      .configure_legend(
          labelColor=GREY_700,
          titleColor=GREY_700,
          orient='bottom',
          labelFontSize=11,
          titleFontSize=12,
          symbolType='circle',
      )
  )


def to_json(chart: alt.TopLevelMixin) -> str:
  """Returns the Vega-Lite JSON of a chart with the shared GeoX look."""
  # Large charts (for example daily heatmaps of many geos) exceed Altair's
  # default 5,000-row limit; the data is embedded in the page either way.
  # pyrefly: ignore[bad-context-manager]
  with alt.data_transformers.disable_max_rows():
    return configure(chart).to_json()
