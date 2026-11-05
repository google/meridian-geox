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

import dataclasses
from typing import Optional, Sequence

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
YELLOW = '#F9AB00'
ORANGE = '#E8710A'
PURPLE = '#A142F4'
TEAL = '#12B5CB'
PINK = '#E52592'
BROWN = '#8D6E63'

CONTROL_COLOR = '#34A853'
EXCLUDED_COLOR = RED
OBSERVED_COLOR = BLUE
COUNTERFACTUAL_COLOR = GREY_700
CI_COLOR = '#AECBFA'
COOLDOWN_COLOR = '#FEF7E0'
# Ramp-up days at the start of a test (lighter than the cooldown yellow).
RAMP_UP_COLOR = '#FDD663'
TEST_COLOR = '#E8F0FE'
FEASIBLE_COLOR = GREEN
INFEASIBLE_COLOR = GREY_500

# Treatment cell colours, in cell order.
# Green and red are reserved for control and excluded geos.
TREATMENT_COLORS = (BLUE, ORANGE, PURPLE, TEAL, PINK, BROWN)

CONTROL_LABEL = 'Control'
EXCLUDED_LABEL = 'Excluded'
NOT_IN_TEST_LABEL = 'Not in test'

# Name of the Vega signal that holds the currently selected design ID. Charts
# that depend on the selected candidate declare a variable parameter with this
# name, and the page script updates it when the user picks a candidate.
DESIGN_SELECTION = 'geox_selected_design'

DEFAULT_WIDTH = 520
DEFAULT_HEIGHT = 260


def treatment_color(index: int) -> str:
  """Returns the colour for the treatment cell at the given 0-based index."""
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


def selection_param(default_design_id: str) -> alt.Parameter:
  """Returns the variable parameter that tracks the selected design."""
  return alt.param(name=DESIGN_SELECTION, value=default_design_id)


@dataclasses.dataclass(frozen=True)
class Framing:
  """Title and reading guide for a chart.

  Attributes:
    title: Short chart title, phrased as the question it answers.
    subtitle: Optional one-line subtitle under the title.
    reading_guide: Plain-language "how to read this chart" text. It is shown
      as a hover badge on screen and inline in the printed Executive PDF.
    takeaway: One-sentence conclusion shown under the chart ("→ ...").
  """

  title: str
  subtitle: Optional[str] = None
  reading_guide: Optional[str] = None
  takeaway: Optional[str] = None


def framed(chart: alt.TopLevelMixin, framing: Framing) -> alt.TopLevelMixin:
  """Applies the title from a `Framing` to a chart."""
  title = alt.TitleParams(
      text=framing.title,
      subtitle=framing.subtitle or '',
      anchor='start',
      font=FONT,
      fontSize=15,
      fontWeight=500,
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
