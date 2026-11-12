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
from urllib import parse

from absl.testing import absltest
from absl.testing import parameterized
from meridian_geox.visualizations import charts
from meridian_geox.visualizations import maps
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


def _feature(properties):
  return {
      'type': 'Feature',
      'properties': properties,
      'geometry': {
          'type': 'Polygon',
          'coordinates': [[[13, 52], [14, 52], [14, 53], [13, 52]]],
      },
  }


# The last two features have no `name` and are skipped.
_GEOJSON = {
    'type': 'FeatureCollection',
    'features': [
        _feature({'name': 'Berlin'}),
        _feature({'name': 'Hamburg'}),
        _feature(None),
        _feature({'other': 'x'}),
    ],
}


def _data_uri(value, encode):
  text = json.dumps(value)
  if encode:
    return 'data:application/json;base64,' + base64.b64encode(
        text.encode()
    ).decode()
  return 'data:application/json,' + parse.quote(text)


class MapConfigTest(parameterized.TestCase):

  @parameterized.parameters(
      ('geoTargetConstants/200501', '501'),
      (200501, '501'),
      ('0501', '501'),
      (501.0, '501'),
      (' Berlin ', 'Berlin'),
      ('1023191', '1023191'),
  )
  def test_normalize_geo_id(self, value, expected):
    self.assertEqual(maps.normalize_geo_id(value), expected)

  def test_byo_dma(self):
    config = maps.byo_dma_map_config('dma.json')
    self.assertEqual(
        config,
        maps.MapConfig(
            region_name='United States (DMA boundaries)',
            polygon_url='dma.json',
        ),
    )
    self.assertEqual(config.polygon_feature, 'nielsen_dma')
    self.assertEqual(config.polygon_join_key, maps.DMA_CODE)

  def test_byo_city(self):
    config = maps.byo_city_map_config('de.geojson')
    self.assertEqual(config.region_name, 'Custom regions')
    self.assertEqual(config.polygon_lookup_key, 'properties.name')
    config = maps.byo_city_map_config('de.geojson', 'Germany', 'id')
    self.assertEqual(
        (config.projection, config.fit_to_geos, config.region_name),
        ('mercator', True, 'Germany'),
    )
    self.assertIsNone(config.polygon_feature)
    self.assertEqual(config.polygon_lookup_key, 'id')
    self.assertEqual(config.polygon_join_key, 'location')


class BoundaryTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    tmp = self.enter_context(tempfile.TemporaryDirectory())
    self.topo_path = os.path.join(tmp, 'dma.json')
    with open(self.topo_path, 'w', encoding='utf-8') as f:
      f.write('\ufeff' + json.dumps(_TOPOJSON))

  def test_dma_coverage(self):
    coverage = maps.boundary_coverage(
        ['200501', 'geoTargetConstants/200803', '999'],
        maps.byo_dma_map_config(self.topo_path),
    )
    self.assertEqual(
        coverage,
        maps.BoundaryCoverage(
            configured=True,
            checked=True,
            matched=('200501', 'geoTargetConstants/200803'),
            unmatched=('999',),
        ),
    )

  @parameterized.parameters(True, False)
  def test_geojson_data_uri_coverage(self, encode):
    config = maps.byo_city_map_config(_data_uri(_GEOJSON, encode))
    coverage = maps.boundary_coverage(['Munich', 'Berlin', 'x'], config)
    self.assertEqual(coverage.matched, ('Berlin',))
    self.assertEqual(coverage.unmatched, ('Munich', 'x'))

  def test_unconfigured_and_remote(self):
    self.assertEqual(
        maps.boundary_coverage(['a'], maps.MapConfig()),
        maps.BoundaryCoverage(configured=False, checked=False),
    )
    remote = maps.byo_dma_map_config('https://example.com/dma.json')
    self.assertEqual(
        maps.boundary_coverage(['501'], remote),
        maps.BoundaryCoverage(configured=True, checked=False),
    )

  def _spec(self, config, table=None, **kwargs):
    if table is None:
      table = pd.DataFrame({
          'location': ['200501', '803'],
          'geo_name': ['Geo A', 'Geo B'],
          'group': ['Treatment', 'Control'],
          'share': [0.6, 0.4],
      })
    chart = maps.plot_geo_assignment(
        table, config, charts.group_scale(['Treatment']), **kwargs
    )
    assert chart is not None
    return chart.to_dict()

  def _lookup_values(self, spec):
    data = [t for t in spec['transform'] if 'lookup' in t][0]['from']['data']
    return data.get('values') or spec['datasets'][data['name']]

  def test_plot_data_url(self):
    inlined = self._spec(maps.byo_dma_map_config(self.topo_path))
    url = inlined['data']['url']
    self.assertTrue(url.startswith('data:application/json;base64,'))
    decoded = base64.b64decode(url.split(',', 1)[1]).decode('utf-8')
    self.assertEqual(json.loads(decoded), _TOPOJSON)
    for kept in ('https://example.com/dma.json', 'data:application/json,{}'):
      spec = self._spec(maps.byo_dma_map_config(kept))
      self.assertEqual(spec['data']['url'], kept)

  def test_plot_dma(self):
    spec = self._spec(maps.byo_dma_map_config(self.topo_path), title='T')
    self.assertEqual(spec['projection']['type'], 'albersUsa')
    self.assertEqual(spec['title'], 'T')
    self.assertEqual(spec['data']['format']['feature'], 'nielsen_dma')
    self.assertEqual(
        [row['join_key'] for row in self._lookup_values(spec)], ['501', '803']
    )
    self.assertIn(
        {'calculate': 'toString(toNumber(datum.id))', 'as': 'geox_key'},
        spec['transform'],
    )
    self.assertNotIn('filter', json.dumps(spec['transform']))

  def test_plot_geojson_fits_to_test_geos(self):
    table = pd.DataFrame({'location': [' Berlin'], 'group': ['Treatment']})
    config = maps.byo_city_map_config(_data_uri(_GEOJSON, encode=True))
    spec = self._spec(config, table)
    self.assertEqual(spec['projection']['type'], 'mercator')
    self.assertEqual(spec['data']['format']['property'], 'features')
    key = 'trim(toString(datum.properties.name))'
    self.assertIn({'calculate': key, 'as': 'geox_key'}, spec['transform'])
    filters = [t['filter'] for t in spec['transform'] if 'filter' in t]
    self.assertEqual(filters, [f'datum.group != "{charts.NOT_IN_TEST_LABEL}"'])
    # Missing lookup columns are filled with None.
    self.assertEqual(
        self._lookup_values(spec),
        [{'join_key': 'Berlin', 'location': ' Berlin', 'geo_name': None,
          'group': 'Treatment', 'share': None}],
    )

  @parameterized.parameters(True, False)
  def test_plot_legend(self, show_legend):
    spec = self._spec(
        maps.byo_dma_map_config(self.topo_path), show_legend=show_legend
    )
    legend = spec['encoding']['color'].get('legend', 'missing')
    self.assertEqual(legend is None, not show_legend)

  def test_plot_without_boundaries(self):
    table = pd.DataFrame({'location': ['a'], 'group': ['Treatment']})
    scale = charts.group_scale([])
    self.assertIsNone(maps.plot_geo_assignment(table, maps.MapConfig(), scale))


if __name__ == '__main__':
  absltest.main()
