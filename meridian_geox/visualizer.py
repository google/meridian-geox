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

"""Lightweight public facade for the GeoX reporting charts and data helpers.

The implementation lives in focused modules: `maps` (boundary maps),
`report_data` (data derived from design/analysis objects), `design_charts`,
`analysis_charts` and `charts` (shared styling).
"""

# pylint: disable=unused-import,g-importing-member
from meridian_geox.analysis_charts import anomaly_heatmap
from meridian_geox.analysis_charts import confidence_curve
from meridian_geox.analysis_charts import cumulative_chart
from meridian_geox.analysis_charts import cumulative_icpd_chart
from meridian_geox.analysis_charts import effect_totals_chart
from meridian_geox.analysis_charts import observed_counterfactual_chart
from meridian_geox.analysis_charts import pointwise_chart
from meridian_geox.analysis_charts import spend_pacing_chart as plot_spend_pacing
from meridian_geox.charts import DESIGN_SELECTION
from meridian_geox.charts import framed
from meridian_geox.charts import Framing
from meridian_geox.charts import group_scale
from meridian_geox.charts import TREATMENT_COLORS
from meridian_geox.design_charts import aa_check_chart
from meridian_geox.design_charts import calendar_chart
from meridian_geox.design_charts import data_quality_chart
from meridian_geox.design_charts import largest_geos_chart
from meridian_geox.design_charts import power_simulator
from meridian_geox.design_charts import pretest_fit_chart
from meridian_geox.design_charts import sensitivity_curve
from meridian_geox.design_charts import tradeoff_chart
from meridian_geox.maps import boundary_coverage
from meridian_geox.maps import BoundaryCoverage
from meridian_geox.maps import byo_city_map_config
from meridian_geox.maps import byo_dma_map_config
from meridian_geox.maps import MapConfig
from meridian_geox.maps import normalize_geo_id
from meridian_geox.maps import plot_geo_assignment
from meridian_geox.maps import US_STATES_TOPOJSON_URL
from meridian_geox.maps import WORLD_COUNTRIES_50M_TOPOJSON_URL
from meridian_geox.report_data import design_summaries
from meridian_geox.report_data import DesignGoals
from meridian_geox.report_data import effect_split
from meridian_geox.report_data import execution_checks
from meridian_geox.report_data import experiment_schedule
from meridian_geox.report_data import geo_table
from meridian_geox.report_data import resolve_geo_metadata
from meridian_geox.report_data import spend_changes
# pylint: enable=unused-import,g-importing-member
