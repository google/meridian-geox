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

Maps fill user-provided boundary polygons (TopoJSON or GeoJSON); no boundaries
ship with the library, so bring a file you are licensed to use. Without
`polygon_url`, map panels show a short setup note instead.
"""

import base64
import dataclasses
import json
import os
import re
from typing import Any, Iterable, Optional
from urllib import parse

import altair as alt
from meridian_geox.visualizations import charts
import pandas as pd

# Google Ads DMA criteria IDs are 200000 + the Nielsen DMA code.
_DMA_CRITERIA_OFFSET = 200000
_GEO_TARGET_PREFIX = 'geoTargetConstants/'
_REMOTE = ('http://', 'https://')

DMA_CODE = 'dma_code'


@dataclasses.dataclass(frozen=True)
class MapConfig:
  """Report map settings; the defaults read US DMA TopoJSON boundaries."""

  projection: str = 'albersUsa'
  fit_to_geos: bool = False
  region_name: Optional[str] = None
  # Local path, http(s) URL or data: URI of the boundaries.
  polygon_url: Optional[str] = None
  # TopoJSON object name, or None for GeoJSON features.
  polygon_feature: Optional[str] = 'nielsen_dma'
  # Feature field matched against the geos.
  polygon_lookup_key: str = 'id'
  # 'dma_code' to match normalized DMA codes, else 'location'.
  polygon_join_key: str = DMA_CODE


def byo_dma_map_config(
    polygon_url: str, region_name: str = 'United States (DMA boundaries)'
) -> MapConfig:
  """Returns a config for a TopoJSON `nielsen_dma` object keyed by DMA code."""
  return MapConfig(region_name=region_name, polygon_url=polygon_url)


def byo_city_map_config(
    polygon_url: str,
    region_name: str = 'Custom regions',
    feature_key: str = 'properties.name',
) -> MapConfig:
  """Returns a GeoJSON config matching `feature_key` against `location`."""
  return MapConfig(
      projection='mercator',
      fit_to_geos=True,
      region_name=region_name,
      polygon_url=polygon_url,
      polygon_feature=None,
      polygon_lookup_key=feature_key,
      polygon_join_key='location',
  )


def normalize_geo_id(value: Any) -> str:
  """Maps 'geoTargetConstants/200501', '200501', '0501' and 501 to '501'."""
  text = str(value).strip().removeprefix(_GEO_TARGET_PREFIX)
  if re.fullmatch(r'\d+(\.0+)?', text):
    number = int(float(text))
    if _DMA_CRITERIA_OFFSET < number < _DMA_CRITERIA_OFFSET + 1000:
      number -= _DMA_CRITERIA_OFFSET
    return str(number)
  return text


def _join_keys(values: Iterable[Any], map_config: MapConfig) -> list[str]:
  if map_config.polygon_join_key == DMA_CODE:
    return [normalize_geo_id(v) for v in values]
  return [str(v).strip() for v in values]


@dataclasses.dataclass(frozen=True)
class BoundaryCoverage:
  """Which geos have a matching boundary polygon."""

  configured: bool
  # False for remote files, which only the viewer's browser reads.
  checked: bool
  matched: tuple[str, ...] = ()
  unmatched: tuple[str, ...] = ()


def boundary_coverage(
    locations: Iterable[str], map_config: MapConfig
) -> BoundaryCoverage:
  """Checks which geos have a matching boundary polygon."""
  locations = sorted({str(loc) for loc in locations})
  url = map_config.polygon_url
  if not url:
    return BoundaryCoverage(configured=False, checked=False)
  if url.startswith(_REMOTE):
    return BoundaryCoverage(configured=True, checked=False)
  if url.startswith('data:'):
    header, _, payload = url.partition(',')
    if ';base64' in header:
      boundaries = json.loads(base64.b64decode(payload).decode('utf-8-sig'))
    else:
      boundaries = json.loads(parse.unquote(payload))
  else:
    with open(os.path.expanduser(url), encoding='utf-8-sig') as f:
      boundaries = json.load(f)
  if map_config.polygon_feature:
    objects = boundaries.get('objects', {})
    features = objects.get(map_config.polygon_feature, {}).get('geometries', [])
  else:
    features = boundaries.get('features', [])
  values = []
  for feature in features:
    value = feature
    for part in map_config.polygon_lookup_key.split('.'):
      value = value.get(part) if isinstance(value, dict) else None
    if value is not None:
      values.append(value)
  keys = set(_join_keys(values, map_config))
  matched, unmatched = [], []
  for location, key in zip(locations, _join_keys(locations, map_config)):
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
    title: str = '',
    show_legend: bool = True,
) -> Optional[alt.Chart]:
  """Maps a `location`/`group` (+ `geo_name`, `share`) table, or None."""
  url = map_config.polygon_url
  if not url:
    return None
  if not url.startswith((*_REMOTE, 'data:')):
    # Local files are inlined so that saved reports work offline.
    with open(os.path.expanduser(url), 'rb') as f:
      content = f.read().removeprefix(b'\xef\xbb\xbf')
    url = 'data:application/json;base64,' + base64.b64encode(content).decode()
  fields = ['location', 'geo_name', 'group', 'share']
  table = geo_table.assign(
      **{f: None for f in fields if f not in geo_table.columns},
      join_key=_join_keys(geo_table['location'], map_config),
  ).drop_duplicates('join_key')
  polygon_format = (
      alt.DataFormat(type='topojson', feature=map_config.polygon_feature)
      if map_config.polygon_feature
      else alt.DataFormat(type='json', property='features')
  )
  key = f'datum.{map_config.polygon_lookup_key}'
  # DMA IDs may be numbers or zero-padded strings in the TopoJSON.
  key = (
      f'toString(toNumber({key}))' if map_config.polygon_join_key == DMA_CODE
      else f'trim(toString({key}))'
  )
  polygons = (
      alt.Chart(alt.Data(url=url, format=polygon_format))
      .mark_geoshape(stroke='white', strokeWidth=0.4)
      .transform_calculate(geox_key=key)
      .transform_lookup(
          lookup='geox_key',
          from_=alt.LookupData(
              data=table[['join_key'] + fields], key='join_key', fields=fields
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
  return polygons.project(type=map_config.projection).properties(
      width=520, height=320, title=title
  )
