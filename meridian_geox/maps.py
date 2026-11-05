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

"""Boundary-polygon maps for the GeoX reports.

Maps fill user-provided boundary polygons (TopoJSON or GeoJSON) over a context
basemap. There is no centroid or tile fallback: without `polygon_url`, map
panels show a short setup note instead.
"""

import base64
import dataclasses
import json
import os
import re
from typing import Any, Iterable, Optional
from urllib import parse

import altair as alt
from meridian_geox import charts
import pandas as pd

US_STATES_TOPOJSON_URL = (
    'https://cdn.jsdelivr.net/npm/us-atlas@3.0.1/states-10m.json'
)
WORLD_COUNTRIES_50M_TOPOJSON_URL = (
    'https://cdn.jsdelivr.net/npm/world-atlas@2.0.2/countries-50m.json'
)

# Google Ads DMA criteria IDs are 200000 + the Nielsen DMA code.
_DMA_CRITERIA_OFFSET = 200000
_GEO_TARGET_PREFIX = 'geoTargetConstants/'

DMA_CODE = 'dma_code'
MAP_WIDTH = 520
MAP_HEIGHT = 320


@dataclasses.dataclass(frozen=True)
class MapConfig:
  """Configuration of report map panels.

  Attributes:
    projection: Vega projection name ('albersUsa', 'mercator', 'equalEarth').
    basemap_url: Context TopoJSON URL, or None for no basemap.
    basemap_feature: TopoJSON object drawn from `basemap_url`.
    highlight_id: Basemap feature ID shaded darker beneath the geo polygons.
    fit_to_geos: Zooms the projection to the geo boundaries.
    region_name: Region label shown in the report header chip.
    polygon_url: Local file path, http(s) URL, or data: URI for boundary
      polygons (TopoJSON if `polygon_feature` is set, else a GeoJSON
      FeatureCollection).
    polygon_feature: TopoJSON object name for boundaries, or None for GeoJSON.
    polygon_lookup_key: Boundary feature field matched against
      `polygon_join_key` (for example 'id' or 'properties.name').
    polygon_join_key: Geo table column joined to `polygon_lookup_key`
      ('location', or 'dma_code' derived via `normalize_geo_id`).
  """

  projection: str = 'albersUsa'
  basemap_url: Optional[str] = US_STATES_TOPOJSON_URL
  basemap_feature: str = 'states'
  highlight_id: Optional[str] = None
  fit_to_geos: bool = False
  region_name: Optional[str] = None
  polygon_url: Optional[str] = None
  polygon_feature: Optional[str] = None
  polygon_lookup_key: str = 'id'
  polygon_join_key: str = 'location'


def byo_dma_map_config(
    polygon_url: str,
    region_name: str = 'United States (DMA boundaries)',
) -> MapConfig:
  """Returns a map config for US DMA TopoJSON boundaries.

  The TopoJSON must hold a `nielsen_dma` object whose geometries carry the
  numeric DMA code in `id`. No boundaries ship with the library: bring a file
  you are licensed to use.

  Args:
    polygon_url: Path, URL, or data: URI of the DMA TopoJSON file.
    region_name: Region label shown in the report header.
  """
  return MapConfig(
      projection='albersUsa',
      basemap_url=US_STATES_TOPOJSON_URL,
      basemap_feature='states',
      region_name=region_name,
      polygon_url=polygon_url,
      polygon_feature='nielsen_dma',
      polygon_lookup_key='id',
      polygon_join_key=DMA_CODE,
  )


def byo_city_map_config(
    polygon_url: str,
    iso_numeric: str,
    region_name: str,
    feature_key: str = 'properties.name',
) -> MapConfig:
  """Returns a map config for GeoJSON boundaries in one country.

  Args:
    polygon_url: Path, URL, or data: URI of a GeoJSON FeatureCollection.
    iso_numeric: ISO 3166-1 numeric country code (for example '276').
    region_name: Region label shown in the report header.
    feature_key: Feature property path matched against `location`.
  """
  return MapConfig(
      projection='mercator',
      basemap_url=WORLD_COUNTRIES_50M_TOPOJSON_URL,
      basemap_feature='countries',
      highlight_id=str(iso_numeric),
      fit_to_geos=True,
      region_name=region_name,
      polygon_url=polygon_url,
      polygon_feature=None,
      polygon_lookup_key=feature_key,
      polygon_join_key='location',
  )


def normalize_geo_id(value: Any) -> str:
  """Normalises a geo ID so that equivalent forms join to one key.

  `'geoTargetConstants/200501'`, `'200501'`, `200501`, `'501'` and `'0501'`
  all normalise to `'501'`. Non-numeric IDs are returned stripped.

  Args:
    value: The geo ID.

  Returns:
    The normalised geo ID string.
  """
  text = str(value).strip()
  if text.startswith(_GEO_TARGET_PREFIX):
    text = text[len(_GEO_TARGET_PREFIX):]
  if re.fullmatch(r'\d+(\.0+)?', text):
    number = int(float(text))
    if _DMA_CRITERIA_OFFSET < number < _DMA_CRITERIA_OFFSET + 1000:
      number -= _DMA_CRITERIA_OFFSET
    return str(number)
  return text


def _is_remote(url: str) -> bool:
  return url.startswith(('http://', 'https://'))


def _decode_data_uri(uri: str) -> str:
  header, _, payload = uri.partition(',')
  if ';base64' in header:
    return base64.b64decode(payload).decode('utf-8-sig')
  return parse.unquote(payload)


def load_boundaries(map_config: MapConfig) -> Optional[dict[str, Any]]:
  """Loads and parses local or data: URI boundaries.

  Args:
    map_config: The map config.

  Returns:
    The parsed TopoJSON/GeoJSON object, or None when `polygon_url` is unset or
    remote (remote files are fetched by the viewer's browser).
  """
  url = map_config.polygon_url
  if not url or _is_remote(url):
    return None
  if url.startswith('data:'):
    return json.loads(_decode_data_uri(url))
  with open(os.path.expanduser(url), encoding='utf-8-sig') as f:
    return json.load(f)


def polygon_data_url(map_config: MapConfig) -> Optional[str]:
  """Returns the URL the browser loads polygons from.

  Local files are inlined as base64 data URIs so saved reports work offline.

  Args:
    map_config: The map config.
  """
  url = map_config.polygon_url
  if not url:
    return None
  if _is_remote(url) or url.startswith('data:'):
    return url
  with open(os.path.expanduser(url), 'rb') as f:
    content = f.read()
  if content.startswith(b'\xef\xbb\xbf'):
    content = content[3:]
  encoded = base64.b64encode(content).decode('ascii')
  return f'data:application/json;base64,{encoded}'


def _get_path(obj: dict[str, Any], path: str) -> Any:
  value: Any = obj
  for part in path.split('.'):
    if not isinstance(value, dict):
      return None
    value = value.get(part)
  return value


def boundary_keys(
    boundaries: dict[str, Any], map_config: MapConfig
) -> set[str]:
  """Returns the lookup keys of all features in a boundary file."""
  if map_config.polygon_feature:
    obj = boundaries.get('objects', {}).get(map_config.polygon_feature, {})
    features = obj.get('geometries', [])
  else:
    features = boundaries.get('features', [])
  keys = set()
  for feature in features:
    value = _get_path(feature, map_config.polygon_lookup_key)
    if value is not None:
      keys.add(_join_value(value, map_config))
  return keys


def _join_value(value: Any, map_config: MapConfig) -> str:
  if map_config.polygon_join_key == DMA_CODE:
    return normalize_geo_id(value)
  return str(value).strip()


@dataclasses.dataclass(frozen=True)
class BoundaryCoverage:
  """Which geos have a matching boundary polygon.

  Attributes:
    configured: Whether a boundary file was configured.
    checked: Whether the file could be checked locally (remote files can only
      be checked by the viewer's browser).
    matched: Geos with a matching polygon.
    unmatched: Geos without a matching polygon.
  """

  configured: bool
  checked: bool
  matched: tuple[str, ...] = ()
  unmatched: tuple[str, ...] = ()

  @property
  def share_matched(self) -> float:
    total = len(self.matched) + len(self.unmatched)
    return len(self.matched) / total if total else 0.0


def join_keys(locations: Iterable[str], map_config: MapConfig) -> list[str]:
  """Returns the polygon join key of each location."""
  return [_join_value(loc, map_config) for loc in locations]


def boundary_coverage(
    locations: Iterable[str], map_config: MapConfig
) -> BoundaryCoverage:
  """Checks which geos have a matching boundary polygon.

  Args:
    locations: Geo locations as they appear in the data.
    map_config: The map config.

  Returns:
    The boundary coverage.
  """
  locations = sorted({str(loc) for loc in locations})
  if not map_config.polygon_url:
    return BoundaryCoverage(configured=False, checked=False)
  boundaries = load_boundaries(map_config)
  if boundaries is None:
    return BoundaryCoverage(configured=True, checked=False, matched=())
  keys = boundary_keys(boundaries, map_config)
  matched, unmatched = [], []
  for location, key in zip(locations, join_keys(locations, map_config)):
    (matched if key in keys else unmatched).append(location)
  return BoundaryCoverage(
      configured=True,
      checked=True,
      matched=tuple(matched),
      unmatched=tuple(unmatched),
  )


SETUP_NOTE = (
    'Map omitted: no boundary file was passed. Pass map_config='
    'geox.byo_dma_map_config(polygon_url=...) for US DMAs, or '
    'geox.byo_city_map_config(...) for GeoJSON regions, to fill each geo on a '
    'map.'
)


def plot_geo_assignment(
    geo_table: pd.DataFrame,
    map_config: MapConfig,
    color_scale: alt.Scale,
    title: str = 'Where the test runs',
    width: int = MAP_WIDTH,
    height: int = MAP_HEIGHT,
    show_legend: bool = True,
) -> Optional[alt.LayerChart]:
  """Plots the geo groups on boundary polygons.

  Args:
    geo_table: One row per geo with `location`, `geo_name`, `group` and
      `share` (pretest conversion share) columns.
    map_config: The map config.
    color_scale: Colour scale over group labels.
    title: Chart title.
    width: Chart width in pixels.
    height: Chart height in pixels.
    show_legend: Whether to draw the group colour legend. Pass False when the
      page lists the groups next to the map.

  Returns:
    The map chart, or None when no boundary file is configured.
  """
  data_url = polygon_data_url(map_config)
  if data_url is None:
    return None
  table = geo_table.copy()
  table['join_key'] = join_keys(table['location'], map_config)
  table = table.drop_duplicates('join_key')
  lookup_fields = ['location', 'geo_name', 'group', 'share']
  for field in lookup_fields:
    if field not in table.columns:
      table[field] = None

  if map_config.polygon_feature:
    polygon_format = alt.DataFormat(
        type='topojson', feature=map_config.polygon_feature
    )
  else:
    polygon_format = alt.DataFormat(type='json', property='features')

  key_expr = f'datum.{map_config.polygon_lookup_key}'
  if map_config.polygon_join_key == DMA_CODE:
    # DMA IDs may be numbers or zero-padded strings in the TopoJSON.
    key_expr = f'toString(toNumber({key_expr}))'
  else:
    key_expr = f'trim(toString({key_expr}))'

  polygons = (
      alt.Chart(alt.Data(url=data_url, format=polygon_format))
      .mark_geoshape(stroke='white', strokeWidth=0.4)
      .transform_calculate(geox_key=key_expr)
      .transform_lookup(
          lookup='geox_key',
          from_=alt.LookupData(
              data=table[['join_key'] + lookup_fields], key='join_key',
              fields=lookup_fields,
          ),
      )
      .transform_calculate(
          group=f'datum.group ? datum.group : "{charts.NOT_IN_TEST_LABEL}"',
          geo_label=(
              'datum.geo_name ? datum.geo_name : (datum.location ? '
              'datum.location : datum.geox_key)'
          ),
      )
      .encode(
          color=alt.Color(
              'group:N',
              scale=color_scale,
              title='Group',
              legend=alt.Legend() if show_legend else None,
          ),
          tooltip=[
              alt.Tooltip('geo_label:N', title='Geo'),
              alt.Tooltip('group:N', title='Group'),
              alt.Tooltip('share:Q', title='Pretest share', format='.2%'),
          ],
      )
  )
  if map_config.fit_to_geos:
    polygons = polygons.transform_filter(
        f'datum.group != "{charts.NOT_IN_TEST_LABEL}"'
    )

  layers = []
  if map_config.basemap_url:
    basemap = alt.Chart(
        alt.Data(
            url=map_config.basemap_url,
            format=alt.DataFormat(
                type='topojson', feature=map_config.basemap_feature
            ),
        )
    )
    if map_config.fit_to_geos and map_config.highlight_id:
      basemap = basemap.transform_filter(
          f'toString(datum.id) == "{map_config.highlight_id}"'
      )
    elif map_config.fit_to_geos:
      basemap = None
    if basemap is not None:
      highlight = map_config.highlight_id or ''
      layers.append(
          basemap.mark_geoshape(stroke='white', strokeWidth=0.6).encode(
              color=alt.condition(
                  f'toString(datum.id) == "{highlight}"',
                  alt.value(charts.GREY_300),
                  alt.value(charts.GREY_100),
              )
          )
      )
  layers.append(polygons)
  return (
      alt.layer(*layers)
      .project(type=map_config.projection)
      .properties(width=width, height=height, title=title)
      .resolve_scale(color='independent')
  )
