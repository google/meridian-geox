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

"""End-to-end tests for GeoX design and analysis."""

import datetime
import os
from absl.testing import absltest
from meridian_geox import analysis
from meridian_geox import api
from meridian_geox import design
import pandas as pd


class SingleCellE2ETest(absltest.TestCase):

  def _get_data_path(self, filename: str) -> str:
    return os.path.join(
        os.path.dirname(__file__),
        'data',
        filename,
    )

  def test_single_cell_stratified_sampling_holdback_design(self):
    # 1. Load design data.
    design_data = pd.read_csv(
        self._get_data_path('example_design_data_single_cell.csv'),
        sep=None,
        engine='python',
    )
    design_data['date'] = pd.to_datetime(design_data['date'])
    design_data['location'] = design_data['location'].astype(str)

    # 2. Run single cell experiment design.
    design_config = api.DesignConfig(
        experiment_duration=datetime.timedelta(days=30),
        experiment_types=api.ExperimentType.HOLDBACK,
        methodology=api.Methodology.TBR,
        geo_assignment_rule=api.GeoAssignmentRule.STRATIFIED_SAMPLING,
        cost_per_incremental_conversion=1,
        cell_count=1,
        design_output_count=5,
        n_candidates=1000,
    )

    constraints = api.Constraints(
        excluded_geos={'105'},
        budget_constraint=api.Budget(budget=500000),
        max_conversions_percent=0.3,
    )
    design_set = design.run_design(design_data, design_config, constraints)

    self.assertNotEmpty(design_set.designs)
    design_id = design_set.design_metrics.design_id.iloc[0]
    selected_design = design_set.designs[design_id]

    self.assertNotEmpty(selected_design.control_geos)
    self.assertNotEmpty(selected_design.designs['cell_1'].treatment_geos)
    # Check union of treatment, control, and excluded equals input geos.
    input_geos = set(design_data['location'].unique())
    design_geos = set(selected_design.control_geos)
    design_geos.update(selected_design.designs['cell_1'].treatment_geos)
    if selected_design.excluded_geos:
      design_geos.update(selected_design.excluded_geos)
    self.assertEqual(input_geos, design_geos)

    cell_design = selected_design.designs['cell_1']
    self.assertGreater(cell_design.minimum_detectable_effect, 0.0)
    self.assertLess(cell_design.minimum_detectable_effect, 1.0)
    self.assertGreaterEqual(cell_design.p_value, 0.0)
    self.assertLessEqual(cell_design.p_value, 1.0)

    treatment_conversion_pct = design_set.design_metrics.iloc[0][
        'treatment_conversions_pct'
    ]
    self.assertGreater(treatment_conversion_pct, 0.0)
    self.assertLess(treatment_conversion_pct, 100.0)

  def test_single_cell_stratified_sampling_holdback_design_with_outliers(self):
    # 1. Load design data.
    design_data = pd.read_csv(
        self._get_data_path('example_design_data_single_cell.csv'),
        sep=None,
        engine='python',
    )
    design_data['date'] = pd.to_datetime(design_data['date'])
    design_data['location'] = design_data['location'].astype(str)

    # Inject data quality issues to simulate outliers/invalid geos.
    outlier_geo = '10'
    outlier_date = pd.to_datetime('2020-01-15')

    design_data.loc[design_data['location'] == outlier_geo, 'conversions'] = 0.0
    if 'spend' in design_data.columns:
      design_data.loc[design_data['location'] == outlier_geo, 'spend'] = 100.0

    design_data.loc[
        design_data['date'] == outlier_date, 'conversions'
    ] *= 99999.0

    # 2. Run single cell experiment design.
    design_config = api.DesignConfig(
        experiment_duration=datetime.timedelta(days=30),
        experiment_types=api.ExperimentType.HOLDBACK,
        methodology=api.Methodology.TBR,
        geo_assignment_rule=api.GeoAssignmentRule.STRATIFIED_SAMPLING,
        cost_per_incremental_conversion=1,
        cell_count=1,
        design_output_count=5,
        n_candidates=1000,
    )

    constraints = api.Constraints(
        excluded_geos={'105'},
        budget_constraint=api.Budget(budget=500000),
        max_conversions_percent=0.3,
    )
    design_set = design.run_design(design_data, design_config, constraints)

    self.assertNotEmpty(design_set.designs)
    design_id = design_set.design_metrics.design_id.iloc[0]
    selected_design = design_set.designs[design_id]

    self.assertNotEmpty(selected_design.control_geos)
    self.assertNotEmpty(selected_design.designs['cell_1'].treatment_geos)

    # Check union of treatment, control, and excluded equals input geos.
    input_geos = set(design_data['location'].unique())
    design_geos = set(selected_design.control_geos)
    design_geos.update(selected_design.designs['cell_1'].treatment_geos)
    if selected_design.excluded_geos:
      design_geos.update(selected_design.excluded_geos)
    self.assertEqual(input_geos, design_geos)

    # Validate that quality checks automatically excluded the bad data.
    self.assertIsNotNone(selected_design.quality_check_result)
    self.assertIn(
        outlier_geo, selected_design.quality_check_result.outlier_geos
    )
    self.assertIn(
        outlier_date, selected_design.quality_check_result.outlier_dates
    )

    self.assertIn(outlier_geo, selected_design.excluded_geos)
    self.assertIn(outlier_date, selected_design.excluded_dates)

    cell_design = selected_design.designs['cell_1']
    self.assertGreater(cell_design.minimum_detectable_effect, 0.0)
    self.assertLess(cell_design.minimum_detectable_effect, 1.0)
    self.assertGreaterEqual(cell_design.p_value, 0.0)
    self.assertLessEqual(cell_design.p_value, 1.0)

    treatment_conversion_pct = design_set.design_metrics.iloc[0][
        'treatment_conversions_pct'
    ]
    self.assertGreater(treatment_conversion_pct, 0.0)
    self.assertLess(treatment_conversion_pct, 100.0)

  def test_single_cell_random_holdback_design(self):
    # 1. Load design data.
    design_data = pd.read_csv(
        self._get_data_path('example_design_data_single_cell.csv'),
        sep=None,
        engine='python',
    )
    design_data['date'] = pd.to_datetime(design_data['date'])
    design_data['location'] = design_data['location'].astype(str)

    # 2. Run single cell experiment design.
    design_config = api.DesignConfig(
        experiment_duration=datetime.timedelta(days=30),
        experiment_types=api.ExperimentType.HOLDBACK,
        methodology=api.Methodology.TBR,
        geo_assignment_rule=api.GeoAssignmentRule.RANDOM,
        cost_per_incremental_conversion=1,
        cell_count=1,
        design_output_count=5,
        n_candidates=1000,
    )

    constraints = api.Constraints(
        excluded_geos={'105'},
        budget_constraint=api.Budget(budget=500000),
        max_conversions_percent=0.3,
    )
    design_set = design.run_design(design_data, design_config, constraints)

    self.assertNotEmpty(design_set.designs)
    design_id = design_set.design_metrics.design_id.iloc[0]
    selected_design = design_set.designs[design_id]

    self.assertNotEmpty(selected_design.control_geos)
    self.assertNotEmpty(selected_design.designs['cell_1'].treatment_geos)

    input_geos = set(design_data['location'].unique())
    design_geos = set(selected_design.control_geos)
    design_geos.update(selected_design.designs['cell_1'].treatment_geos)
    if selected_design.excluded_geos:
      design_geos.update(selected_design.excluded_geos)
    self.assertEqual(input_geos, design_geos)

    cell_design = selected_design.designs['cell_1']
    self.assertGreater(cell_design.minimum_detectable_effect, 0.0)
    self.assertLess(cell_design.minimum_detectable_effect, 1.0)
    self.assertGreaterEqual(cell_design.p_value, 0.0)
    self.assertLessEqual(cell_design.p_value, 1.0)

    treatment_conversion_pct = design_set.design_metrics.iloc[0][
        'treatment_conversions_pct'
    ]
    self.assertGreater(treatment_conversion_pct, 0.0)
    self.assertLess(treatment_conversion_pct, 100.0)

  def test_single_cell_stratified_sampling_holdback_analysis(self):
    # 1. Load analysis data.
    analysis_data = pd.read_csv(
        self._get_data_path('example_analysis_data_single_cell_holdback.csv'),
        sep=None,
        engine='python',
    )
    analysis_data['date'] = pd.to_datetime(analysis_data['date'])
    analysis_data['location'] = analysis_data['location'].astype(str)

    # 2. Load design from pre-saved JSON.
    json_path = self._get_data_path(
        'example_design_stratified_sampling_holdback.json'
    )
    with open(json_path, 'r') as f:
      selected_design = api.Design.load_from_json(f.read())

    # 3. Run single cell experiment analysis.
    analysis_config = api.AnalysisConfig(
        design=selected_design,
        analysis_start_date=pd.to_datetime('2020-04-01'),
        analysis_end_date=pd.to_datetime('2020-04-30'),
    )

    analysis_result = analysis.analyze(analysis_data, analysis_config)
    self.assertIn('cell_1', analysis_result.results)

    metrics = analysis_result.results['cell_1']
    self.assertIsNotNone(metrics.lift.point_estimate)
    self.assertGreaterEqual(metrics.lift.p_value, 0.0)
    self.assertLessEqual(metrics.lift.p_value, 1.0)
    self.assertIsNotNone(metrics.percent_lift.point_estimate)

  def test_single_cell_stratified_sampling_holdback_analysis_with_outliers(
      self,
  ):
    # 1. Load analysis data.
    analysis_data = pd.read_csv(
        self._get_data_path('example_analysis_data_single_cell_holdback.csv'),
        sep=None,
        engine='python',
    )
    analysis_data['date'] = pd.to_datetime(analysis_data['date'])
    analysis_data['location'] = analysis_data['location'].astype(str)

    # Inject data quality issues to simulate outliers.
    outlier_date = pd.to_datetime('2020-01-15')

    analysis_data.loc[
        analysis_data['date'] == outlier_date, 'conversions'
    ] *= 99999.0

    # 2. Load design from pre-saved JSON.
    json_path = self._get_data_path(
        'example_design_stratified_sampling_holdback.json'
    )
    with open(json_path, 'r') as f:
      selected_design = api.Design.load_from_json(f.read())

    # 3. Run single cell experiment analysis.
    analysis_config = api.AnalysisConfig(
        design=selected_design,
        analysis_start_date=pd.to_datetime('2020-04-01'),
        analysis_end_date=pd.to_datetime('2020-04-30'),
    )

    analysis_result = analysis.analyze(analysis_data, analysis_config)
    self.assertIn('cell_1', analysis_result.results)

    # Validate that quality checks automatically excluded the bad data.
    self.assertIsNotNone(analysis_result.quality_check_result)
    self.assertIn(
        outlier_date, analysis_result.quality_check_result.outlier_dates
    )

    self.assertIn(outlier_date, analysis_result.excluded_dates)

    metrics = analysis_result.results['cell_1']
    self.assertIsNotNone(metrics.lift.point_estimate)
    self.assertGreaterEqual(metrics.lift.p_value, 0.0)
    self.assertLessEqual(metrics.lift.p_value, 1.0)
    self.assertIsNotNone(metrics.percent_lift.point_estimate)

  def test_stratified_holdback_design_and_analysis_with_outliers(self):
    # 1. Load design data.
    design_data = pd.read_csv(
        self._get_data_path('example_design_data_single_cell.csv'),
        sep=None,
        engine='python',
    )
    design_data['date'] = pd.to_datetime(design_data['date'])
    design_data['location'] = design_data['location'].astype(str)

    # Load analysis data.
    analysis_data = pd.read_csv(
        self._get_data_path('example_analysis_data_single_cell_holdback.csv'),
        sep=None,
        engine='python',
    )
    analysis_data['date'] = pd.to_datetime(analysis_data['date'])
    analysis_data['location'] = analysis_data['location'].astype(str)

    # Inject data quality issues to simulate outliers/invalid geos for design.
    outlier_geo = '10'
    outlier_date = pd.to_datetime('2020-01-15')

    design_data.loc[design_data['location'] == outlier_geo, 'conversions'] = 0.0
    if 'spend' in design_data.columns:
      design_data.loc[design_data['location'] == outlier_geo, 'spend'] = 100.0

    design_data.loc[
        design_data['date'] == outlier_date, 'conversions'
    ] *= 99999.0

    # 2. Run single cell experiment design.
    design_config = api.DesignConfig(
        experiment_duration=datetime.timedelta(days=30),
        experiment_types=api.ExperimentType.HOLDBACK,
        methodology=api.Methodology.TBR,
        geo_assignment_rule=api.GeoAssignmentRule.STRATIFIED_SAMPLING,
        cost_per_incremental_conversion=1,
        cell_count=1,
        design_output_count=5,
        n_candidates=500,
    )

    constraints = api.Constraints(
        excluded_geos={'105'},
        budget_constraint=api.Budget(budget=500000),
        max_conversions_percent=0.3,
    )
    design_set = design.run_design(design_data, design_config, constraints)

    design_id = design_set.design_metrics.design_id.iloc[0]
    selected_design = design_set.designs[design_id]

    # 3. Serialize and deserialize
    design_json = selected_design.export_to_json()
    loaded_design = api.Design.load_from_json(design_json)

    # Validate that quality checks automatically excluded the bad data in
    # design.
    self.assertIn(outlier_geo, loaded_design.excluded_geos)
    self.assertIn(outlier_date, loaded_design.excluded_dates)

    analysis_data.loc[
        analysis_data['date'] == outlier_date, 'conversions'
    ] *= 99999.0

    # 5. Run single cell experiment analysis.
    analysis_config = api.AnalysisConfig(
        design=loaded_design,
        analysis_start_date=pd.to_datetime('2020-04-01'),
        analysis_end_date=pd.to_datetime('2020-04-30'),
    )

    analysis_result = analysis.analyze(analysis_data, analysis_config)
    self.assertIn('cell_1', analysis_result.results)

    self.assertIsNotNone(analysis_result.quality_check_result)
    self.assertIn(
        outlier_date, analysis_result.quality_check_result.outlier_dates
    )

    self.assertIn(outlier_geo, analysis_result.excluded_geos)
    self.assertIn(outlier_date, analysis_result.excluded_dates)

    metrics = analysis_result.results['cell_1']
    self.assertIsNotNone(metrics.lift.point_estimate)
    self.assertGreaterEqual(metrics.lift.p_value, 0.0)
    self.assertLessEqual(metrics.lift.p_value, 1.0)
    self.assertIsNotNone(metrics.percent_lift.point_estimate)


class MulticellE2ETest(absltest.TestCase):

  def _get_data_path(self, filename: str) -> str:
    return os.path.join(
        os.path.dirname(__file__),
        'data',
        filename,
    )

  def test_multicell_stratified_sampling_design_go_dark_heavy_up(self):
    design_data = pd.read_csv(
        self._get_data_path(
            'example_design_data_multi_cell_go_dark_heavy_up.csv'
        ),
        sep=None,
        engine='python',
    )
    design_data['date'] = pd.to_datetime(design_data['date'])
    design_data['location'] = design_data['location'].astype(str)

    design_config = api.DesignConfig(
        experiment_duration=datetime.timedelta(days=30),
        experiment_types={
            'cell_1': api.ExperimentType.GO_DARK,
            'cell_2': api.ExperimentType.HEAVY_UP,
        },
        methodology=api.Methodology.TBR,
        geo_assignment_rule=api.GeoAssignmentRule.STRATIFIED_SAMPLING,
        cell_count=2,
        design_output_count=5,
        n_candidates=1000,
    )

    constraints = api.Constraints(
        excluded_geos={'105'},
        budget_constraint={
            'cell_1': api.Budget(budget_pct=-1.0),
            'cell_2': api.Budget(budget_pct=1.0),
        },
        max_conversions_percent=0.3,
    )
    design_set = design.run_design(design_data, design_config, constraints)

    self.assertNotEmpty(design_set.designs)
    design_id = design_set.design_metrics.design_id.iloc[0]
    selected_design = design_set.designs[design_id]

    self.assertNotEmpty(selected_design.control_geos)
    input_geos = set(design_data['location'].unique())
    design_geos = set(selected_design.control_geos)
    design_geos.update(selected_design.designs['cell_1'].treatment_geos)
    design_geos.update(selected_design.designs['cell_2'].treatment_geos)
    if selected_design.excluded_geos:
      design_geos.update(selected_design.excluded_geos)
    self.assertEqual(input_geos, design_geos)

    for cell_key in ['cell_1', 'cell_2']:
      cell = selected_design.designs[cell_key]
      self.assertNotEmpty(cell.treatment_geos)
      self.assertGreater(cell.minimum_detectable_effect, 0.0)
      self.assertGreaterEqual(cell.p_value, 0.0)
      self.assertLessEqual(cell.p_value, 1.0)

      cell_metrics = design_set.design_metrics[
          (design_set.design_metrics['design_id'] == design_id)
          & (design_set.design_metrics['cell'] == cell_key)
      ]
      treatment_conversion_pct = cell_metrics['treatment_conversions_pct'].iloc[
          0
      ]
      self.assertGreater(treatment_conversion_pct, 0.0)
      self.assertLess(treatment_conversion_pct, 100.0)

  def test_multicell_stratified_sampling_analysis_go_dark_heavy_up(self):
    analysis_data = pd.read_csv(
        self._get_data_path(
            'example_analysis_data_multi_cell_go_dark_heavy_up.csv'
        ),
        sep=None,
        engine='python',
    )
    analysis_data['date'] = pd.to_datetime(analysis_data['date'])
    analysis_data['location'] = analysis_data['location'].astype(str)

    json_path = self._get_data_path(
        'example_multicell_design_stratified_sampling_go_dark_heavy_up.json'
    )
    with open(json_path, 'r') as f:
      selected_design = api.Design.load_from_json(f.read())

    analysis_config = api.AnalysisConfig(
        design=selected_design,
        analysis_start_date=pd.to_datetime('2020-04-01'),
        analysis_end_date=pd.to_datetime('2020-04-30'),
    )

    analysis_result = analysis.analyze(analysis_data, analysis_config)
    self.assertIn('cell_1', analysis_result.results)
    self.assertIn('cell_2', analysis_result.results)

    for cell_key in ['cell_1', 'cell_2']:
      metrics = analysis_result.results[cell_key]
      self.assertIsNotNone(metrics.lift.point_estimate)
      self.assertGreaterEqual(metrics.lift.p_value, 0.0)
      self.assertLessEqual(metrics.lift.p_value, 1.0)
      self.assertIsNotNone(metrics.percent_lift.point_estimate)


if __name__ == '__main__':
  absltest.main()
