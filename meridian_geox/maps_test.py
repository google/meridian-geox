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

"""Tests for GeoX boundary maps."""

import base64
import json
import os
import tempfile

from absl.testing import absltest
from absl.testing import parameterized
from meridian_geox import charts
from meridian_geox import maps
import pandas as pd

_TOPOJSON = {
    'type': 'Topology',
    'arcs': [[[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1]]],
    'objects': {
        'nielsen_dma': {
            'type': 'GeometryCollection',
            'geometries': [
                {'type': 'Polygon', 'id': 501, 'arcs': [[0]]},
                {'type': 'Polygon', 'id': '803', 'arcs': [[0]]},
                {'type': 'Polygon', 'id': 602, 'arcs': [[0]]},
            ],
        }
    },
}

_GEOJSON = {
    'type': 'FeatureCollection',
    'features': [
        {
            'type': 'Feature',
            'properties': {'name': 'Berlin'},
            'geometry': {
                'type': 'Polygon',
                'coordinates': [[[13, 52], [14, 52], [14, 53], [13, 52]]],
            },
        },
        {
            'type': 'Feature',
            'properties': {'name': 'Hamburg'},
            'geometry': {
                'type': 'Polygon',
                'coordinates': [[[9, 53], [10, 53], [10, 54], [9, 53]]],
            },
        },
    ],
}


class NormalizeGeoIdTest(parameterized.TestCase):

  @parameterized.parameters(
      ('geoTargetConstants/200501', '501'),
      ('200501', '501'),
      (200501, '501'),
      ('501', '501'),
      ('0501', '501'),
      (501.0, '501'),
      (' Berlin ', 'Berlin'),
      ('1023191', '1023191'),
  )
  def test_normalize(self, value, expected):
    self.assertEqual(maps.normalize_geo_id(value), expected)


class MapConfigTest(absltest.TestCase):

  def test_default_config(self):
    config = maps.MapConfig()
    self.assertEqual(config.projection, 'albersUsa')
    self.assertEqual(config.basemap_url, maps.US_STATES_TOPOJSON_URL)
    self.assertIsNone(config.polygon_url)

  def test_byo_dma(self):
    config = maps.byo_dma_map_config('dma_boundaries.json')
    self.assertEqual(config.polygon_url, 'dma_boundaries.json')
    self.assertEqual(config.polygon_feature, 'nielsen_dma')
    self.assertEqual(config.polygon_join_key, maps.DMA_CODE)
    self.assertEqual(config.region_name, 'United States (DMA boundaries)')

  def test_byo_city(self):
    config = maps.byo_city_map_config('de.geojson', '276', 'Germany')
    self.assertEqual(config.projection, 'mercator')
    self.assertEqual(config.basemap_feature, 'countries')
    self.assertEqual(config.highlight_id, '276')
    self.assertTrue(config.fit_to_geos)
    self.assertIsNone(config.polygon_feature)
    self.assertEqual(config.polygon_lookup_key, 'properties.name')


class BoundaryTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    tmp = self.enter_context(tempfile.TemporaryDirectory())
    self.topo_path = os.path.join(tmp, 'dma.json')
    with open(self.topo_path, 'w', encoding='utf-8') as f:
      f.write('\ufeff' + json.dumps(_TOPOJSON))
    self.geo_path = os.path.join(tmp, 'de.geojson')
    with open(self.geo_path, 'w') as f:
      json.dump(_GEOJSON, f)

  def test_dma_coverage(self):
    config = maps.byo_dma_map_config(polygon_url=self.topo_path)
    coverage = maps.boundary_coverage(
        ['200501', 'geoTargetConstants/200803', '999'], config
    )
    self.assertTrue(coverage.configured)
    self.assertTrue(coverage.checked)
    self.assertEqual(
        coverage.matched, ('200501', 'geoTargetConstants/200803')
    )
    self.assertEqual(coverage.unmatched, ('999',))
    self.assertAlmostEqual(coverage.share_matched, 2 / 3)

  def test_geojson_coverage(self):
    config = maps.byo_city_map_config(self.geo_path, '276', 'Germany')
    coverage = maps.boundary_coverage(['Berlin', 'Munich'], config)
    self.assertEqual(coverage.matched, ('Berlin',))
    self.assertEqual(coverage.unmatched, ('Munich',))

  def test_data_uri_coverage(self):
    uri = 'data:application/json;base64,' + base64.b64encode(
        json.dumps(_GEOJSON).encode()
    ).decode()
    config = maps.byo_city_map_config(uri, '276', 'Germany')
    self.assertEqual(
        maps.boundary_coverage(['Hamburg'], config).matched, ('Hamburg',)
    )

  def test_unconfigured_and_remote(self):
    self.assertFalse(
        maps.boundary_coverage(['a'], maps.MapConfig()).configured
    )
    remote = maps.byo_dma_map_config('https://example.com/dma.json')
    coverage = maps.boundary_coverage(['501'], remote)
    self.assertTrue(coverage.configured)
    self.assertFalse(coverage.checked)

  def test_polygon_data_url_inlines_local_files(self):
    config = maps.byo_dma_map_config(polygon_url=self.topo_path)
    url = maps.polygon_data_url(config)
    assert url is not None
    self.assertTrue(url.startswith('data:application/json;base64,'))
    decoded = base64.b64decode(url.split(',', 1)[1]).decode('utf-8')
    self.assertEqual(json.loads(decoded), _TOPOJSON)
    remote = maps.byo_dma_map_config('https://example.com/dma.json')
    self.assertEqual(
        maps.polygon_data_url(remote), 'https://example.com/dma.json'
    )
    self.assertIsNone(maps.polygon_data_url(maps.MapConfig()))

  def _geo_table(self, locations):
    return pd.DataFrame({
        'location': locations,
        'geo_name': [f'Geo {loc}' for loc in locations],
        'group': ['Treatment', 'Control'][: len(locations)],
        'share': [0.6, 0.4][: len(locations)],
    })

  def test_plot_geo_assignment_dma(self):
    config = maps.byo_dma_map_config(polygon_url=self.topo_path)
    chart = maps.plot_geo_assignment(
        self._geo_table(['200501', '803']),
        config,
        charts.group_scale(['Treatment']),
    )
    assert chart is not None
    spec = chart.to_dict()
    self.assertLen(spec['layer'], 2)
    self.assertEqual(spec['projection']['type'], 'albersUsa')
    polygon_layer = spec['layer'][1]
    self.assertEqual(polygon_layer['data']['format']['feature'], 'nielsen_dma')
    lookup = [t for t in polygon_layer['transform'] if 'lookup' in t][0]
    data = lookup['from']['data']
    values = data.get('values') or spec['datasets'][data['name']]
    keys = [row['join_key'] for row in values]
    self.assertEqual(keys, ['501', '803'])

  def test_plot_geo_assignment_city_fits_to_country(self):
    config = maps.byo_city_map_config(self.geo_path, '276', 'Germany')
    chart = maps.plot_geo_assignment(
        self._geo_table(['Berlin']), config, charts.group_scale(['Treatment'])
    )
    assert chart is not None
    spec = chart.to_dict()
    self.assertEqual(spec['projection']['type'], 'mercator')
    basemap_filters = [
        t['filter'] for t in spec['layer'][0]['transform'] if 'filter' in t
    ]
    self.assertEqual(basemap_filters, ['toString(datum.id) == "276"'])
    self.assertEqual(spec['layer'][1]['data']['format']['property'], 'features')

  def test_plot_geo_assignment_without_boundaries(self):
    self.assertIsNone(
        maps.plot_geo_assignment(
            self._geo_table(['a']), maps.MapConfig(), charts.group_scale([])
        )
    )


if __name__ == '__main__':
  absltest.main()
