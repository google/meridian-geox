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

import os
from absl.testing import absltest
from meridian_geox import analysis
from meridian_geox import api
from meridian_geox import design
import numpy as np
import pandas as pd


class SingleCellE2ETest(absltest.TestCase):

  def _get_data_path(self, filename: str) -> str:
    return os.path.join(
        os.path.dirname(__file__),
        'data',
        filename,
    )

  def _compute_treatment_conversion_pct(self, selected_design) -> float:
    treatment_geos = selected_design.designs['cell_1'].treatment_geos
    total_conversions = selected_design.data['conversions'].sum()
    treatment_conversions = selected_design.data[
        selected_design.data['location'].isin(treatment_geos)
    ]['conversions'].sum()
    return treatment_conversions / total_conversions

  def test_single_cell_stratified_sampling_holdback_design(self):
    # 1. Load design data.
    design_data = pd.read_csv(
        self._get_data_path('example_design_data_single_cell.csv')
    )
    design_data['date'] = pd.to_datetime(design_data['date'])
    design_data['location'] = design_data['location'].astype(str)

    # 2. Run single cell experiment design.
    design_config = api.DesignConfig(
        experiment_duration=30,
        experiment_types=api.ExperimentType.HOLDBACK,
        methodology=api.Methodology.TBR,
        geo_assignment_rule=api.GeoAssignmentRule.STRATIFIED_SAMPLING,
        cell_count=1,
    )

    constraints = api.Constraints(
        excluded_geos={'105'}, budget_constraint=api.Budget(budget=50000)
    )
    design_set = design.run_design(design_data, design_config, constraints)

    self.assertNotEmpty(design_set.designs)
    design_id = design_set.design_metrics.design_id.iloc[0]
    selected_design = design_set.designs[design_id]

    # Verify design results.
    self.assertLen(selected_design.control_geos, 85)
    self.assertLen(selected_design.designs['cell_1'].treatment_geos, 28)
    np.testing.assert_allclose(
        selected_design.designs['cell_1'].minimum_detectable_effect,
        0.0140,
        rtol=0.01,
    )
    np.testing.assert_allclose(
        selected_design.designs['cell_1'].p_value, 0.1915, rtol=0.01
    )
    np.testing.assert_allclose(
        selected_design.designs['cell_1'].budget, 7569.0, rtol=0.01
    )
    treatment_conversion_pct = self._compute_treatment_conversion_pct(
        selected_design
    )
    np.testing.assert_allclose(treatment_conversion_pct, 0.294, rtol=0.01)

  def test_single_cell_stratified_sampling_holdback_analysis(self):
    # 1. Load analysis data.
    analysis_data = pd.read_csv(
        self._get_data_path('example_analysis_data_single_cell_holdback.csv')
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
        analysis_start_date=pd.Timestamp('2020-04-01'),
        analysis_end_date=pd.Timestamp('2020-04-30'),
        alpha=0.1,
    )

    analysis_result = analysis.analyze(analysis_data, analysis_config)
    self.assertIn('cell_1', analysis_result.results)
    metrics = analysis_result.results['cell_1']

    # Verify analysis results.
    np.testing.assert_allclose(metrics.lift.point_estimate, 26667.5, rtol=0.01)
    np.testing.assert_allclose(metrics.lift.p_value, 0.002, rtol=0.01)
    np.testing.assert_allclose(
        metrics.percent_lift.point_estimate, 0.049, rtol=0.01
    )

  def test_single_cell_stratified_sampling_godark_design(self):
    # 1. Load design data.
    design_data = pd.read_csv(
        self._get_data_path('example_design_data_single_cell.csv')
    )
    design_data['date'] = pd.to_datetime(design_data['date'])
    design_data['location'] = design_data['location'].astype(str)

    # 2. Run single cell experiment design.
    design_config = api.DesignConfig(
        experiment_duration=30,
        experiment_types=api.ExperimentType.GO_DARK,
        methodology=api.Methodology.TBR,
        geo_assignment_rule=api.GeoAssignmentRule.STRATIFIED_SAMPLING,
        cell_count=1,
    )

    constraints = api.Constraints(
        excluded_geos={'105'}, budget_constraint=api.Budget(budget_pct=1.0)
    )
    design_set = design.run_design(design_data, design_config, constraints)

    self.assertNotEmpty(design_set.designs)
    design_id = design_set.design_metrics.design_id.iloc[0]
    selected_design = design_set.designs[design_id]

    # Verify design results.
    self.assertLen(selected_design.control_geos, 85)
    self.assertLen(selected_design.designs['cell_1'].treatment_geos, 28)
    np.testing.assert_allclose(
        selected_design.designs['cell_1'].minimum_detectable_effect,
        0.0134,
        rtol=0.01,
    )
    np.testing.assert_allclose(
        selected_design.designs['cell_1'].p_value, 0.6508, rtol=0.01
    )
    np.testing.assert_allclose(
        selected_design.designs['cell_1'].budget, 6236.45, rtol=0.01
    )
    treatment_conversion_pct = self._compute_treatment_conversion_pct(
        selected_design
    )
    np.testing.assert_allclose(treatment_conversion_pct, 0.295, rtol=0.01)

  def test_single_cell_stratified_sampling_godark_analysis(self):
    # 1. Load analysis data.
    analysis_data = pd.read_csv(
        self._get_data_path('example_analysis_data_single_cell_godark.csv')
    )
    analysis_data['date'] = pd.to_datetime(analysis_data['date'])
    analysis_data['location'] = analysis_data['location'].astype(str)

    # 2. Load design from pre-saved JSON.
    json_path = self._get_data_path(
        'example_design_stratified_sampling_godark.json'
    )
    with open(json_path, 'r') as f:
      selected_design = api.Design.load_from_json(f.read())

    # 3. Run single cell experiment analysis.
    analysis_config = api.AnalysisConfig(
        design=selected_design,
        analysis_start_date=pd.Timestamp('2020-04-01'),
        analysis_end_date=pd.Timestamp('2020-04-30'),
        alpha=0.1,
    )

    analysis_result = analysis.analyze(analysis_data, analysis_config)
    self.assertIn('cell_1', analysis_result.results)
    metrics = analysis_result.results['cell_1']

    # Verify analysis results.
    np.testing.assert_allclose(metrics.lift.point_estimate, 27272.4, rtol=0.01)
    np.testing.assert_allclose(metrics.lift.p_value, 0.002, rtol=0.01)
    np.testing.assert_allclose(
        metrics.percent_lift.point_estimate, 0.05, rtol=0.01
    )

  def test_single_cell_random_holdback_design(self):
    # 1. Load design data.
    design_data = pd.read_csv(
        self._get_data_path('example_design_data_single_cell.csv')
    )
    design_data['date'] = pd.to_datetime(design_data['date'])
    design_data['location'] = design_data['location'].astype(str)

    # 2. Run single cell experiment design.
    design_config = api.DesignConfig(
        experiment_duration=30,
        experiment_types=api.ExperimentType.HOLDBACK,
        methodology=api.Methodology.TBR,
        geo_assignment_rule=api.GeoAssignmentRule.RANDOM,
        cell_count=1,
    )

    constraints = api.Constraints(budget_constraint=api.Budget(budget=50000))
    design_set = design.run_design(design_data, design_config, constraints)

    self.assertNotEmpty(design_set.designs)
    design_id = design_set.design_metrics.design_id.iloc[0]
    selected_design = design_set.designs[design_id]

    # Verify design results.
    self.assertLen(selected_design.control_geos, 80)
    self.assertLen(selected_design.designs['cell_1'].treatment_geos, 34)
    np.testing.assert_allclose(
        selected_design.designs['cell_1'].minimum_detectable_effect,
        0.0173,
        rtol=0.01,
    )
    np.testing.assert_allclose(
        selected_design.designs['cell_1'].p_value, 0.1252, rtol=0.01
    )
    np.testing.assert_allclose(
        selected_design.designs['cell_1'].budget, 9471.79, rtol=0.01
    )
    treatment_conversion_pct = self._compute_treatment_conversion_pct(
        selected_design
    )
    np.testing.assert_allclose(treatment_conversion_pct, 0.299, rtol=0.01)


class MulticellE2ETest(absltest.TestCase):

  def _get_data_path(self, filename: str) -> str:
    return os.path.join(
        os.path.dirname(__file__),
        'data',
        filename,
    )

  def _compute_treatment_conversion_pct(
      self, selected_design: api.Design, cell: str
  ) -> float:
    treatment_geos = selected_design.designs[cell].treatment_geos
    total_conversions = selected_design.data['conversions'].sum()  # pyrefly: ignore[unsupported-operation]
    treatment_conversions = selected_design.data[  # pyrefly: ignore[unsupported-operation]
        selected_design.data['location'].isin(treatment_geos)  # pyrefly: ignore[unsupported-operation]
    ]['conversions'].sum()
    return treatment_conversions / total_conversions

  def test_multicell_stratified_sampling_design_go_dark_heavy_up(self):
    # 1. Load design data.
    design_data = pd.read_csv(
        self._get_data_path(
            'example_design_data_multi_cell_go_dark_heavy_up.csv'
        )
    )
    design_data['date'] = pd.to_datetime(design_data['date'])
    design_data['location'] = design_data['location'].astype(str)

    # 2. Run multicell experiment design.
    design_config = api.DesignConfig(
        experiment_duration=30,
        experiment_types={
            'cell_1': api.ExperimentType.GO_DARK,
            'cell_2': api.ExperimentType.HEAVY_UP,
        },
        methodology=api.Methodology.TBR,
        geo_assignment_rule=api.GeoAssignmentRule.STRATIFIED_SAMPLING,
        cell_count=2,
    )

    constraints = api.Constraints(
        excluded_geos={'105'},
        budget_constraint={
            'cell_1': api.Budget(budget_pct=1.0),
            'cell_2': api.Budget(budget_pct=1.0),
        },
    )
    design_set = design.run_design(design_data, design_config, constraints)

    self.assertNotEmpty(design_set.designs)
    design_id = design_set.design_metrics.design_id.iloc[0]
    selected_design = design_set.designs[design_id]

    # Verify design results.
    self.assertLen(selected_design.control_geos, 81)

    # Cell 1 (GO_DARK) assertions.
    self.assertLen(selected_design.designs['cell_1'].treatment_geos, 16)
    np.testing.assert_allclose(
        selected_design.designs['cell_1'].minimum_detectable_effect,
        0.0357,
        rtol=0.01,
    )
    np.testing.assert_allclose(
        selected_design.designs['cell_1'].p_value, 0.1909, rtol=0.01
    )
    np.testing.assert_allclose(
        selected_design.designs['cell_1'].budget, 78400.2, rtol=0.01
    )
    treatment_conversion_pct_1 = self._compute_treatment_conversion_pct(
        selected_design, 'cell_1'
    )
    np.testing.assert_allclose(treatment_conversion_pct_1, 0.144, rtol=0.01)

    # Cell 2 (HEAVY_UP) assertions.
    self.assertLen(selected_design.designs['cell_2'].treatment_geos, 16)
    np.testing.assert_allclose(
        selected_design.designs['cell_2'].minimum_detectable_effect,
        0.0342,
        rtol=0.01,
    )
    np.testing.assert_allclose(
        selected_design.designs['cell_2'].p_value, 0.9764, rtol=0.01
    )
    np.testing.assert_allclose(
        selected_design.designs['cell_2'].budget, 87354.3, rtol=0.01
    )
    treatment_conversion_pct_2 = self._compute_treatment_conversion_pct(
        selected_design, 'cell_2'
    )
    np.testing.assert_allclose(treatment_conversion_pct_2, 0.148, rtol=0.01)

  def test_multicell_stratified_sampling_analysis_go_dark_heavy_up(self):
    # 1. Load analysis data.
    analysis_data = pd.read_csv(
        self._get_data_path(
            'example_analysis_data_multi_cell_go_dark_heavy_up.csv'
        )
    )
    analysis_data['date'] = pd.to_datetime(analysis_data['date'])
    analysis_data['location'] = analysis_data['location'].astype(str)

    # 2. Load design from pre-saved JSON.
    json_path = self._get_data_path(
        'example_multicell_design_stratified_sampling_go_dark_heavy_up.json'
    )
    with open(json_path, 'r') as f:
      selected_design = api.Design.load_from_json(f.read())

    # 3. Run multicell experiment analysis.
    analysis_config = api.AnalysisConfig(
        design=selected_design,
        analysis_start_date=pd.Timestamp('2020-04-01'),
        analysis_end_date=pd.Timestamp('2020-04-30'),
        alpha=0.1,
    )

    analysis_result = analysis.analyze(analysis_data, analysis_config)
    self.assertIn('cell_1', analysis_result.results)
    self.assertIn('cell_2', analysis_result.results)

    # Verify Cell 1 (GO_DARK) analysis results.
    metrics_1 = analysis_result.results['cell_1']
    np.testing.assert_allclose(
        metrics_1.lift.point_estimate, 147057.25, rtol=0.01
    )
    np.testing.assert_allclose(metrics_1.lift.p_value, 0.002, rtol=0.01)
    np.testing.assert_allclose(
        metrics_1.percent_lift.point_estimate, 0.5973, rtol=0.01
    )

    # Verify Cell 2 (HEAVY_UP) analysis results.
    metrics_2 = analysis_result.results['cell_2']
    np.testing.assert_allclose(
        metrics_2.lift.point_estimate, 378203.7, rtol=0.01
    )
    np.testing.assert_allclose(metrics_2.lift.p_value, 0.002, rtol=0.01)
    np.testing.assert_allclose(
        metrics_2.percent_lift.point_estimate, 1.4908, rtol=0.01
    )

  def test_multicell_stratified_sampling_design_holdback_holdback(self):
    # 1. Load design data.
    design_data = pd.read_csv(
        self._get_data_path(
            'example_design_data_multi_cell_go_dark_heavy_up.csv'
        )
    )
    design_data['date'] = pd.to_datetime(design_data['date'])
    design_data['location'] = design_data['location'].astype(str)

    # 2. Run multicell experiment design.
    design_config = api.DesignConfig(
        experiment_duration=30,
        experiment_types={
            'cell_1': api.ExperimentType.HOLDBACK,
            'cell_2': api.ExperimentType.HOLDBACK,
        },
        methodology=api.Methodology.TBR,
        geo_assignment_rule=api.GeoAssignmentRule.STRATIFIED_SAMPLING,
        cell_count=2,
    )

    constraints = api.Constraints(
        excluded_geos={'105'},
        budget_constraint={
            'cell_1': api.Budget(budget=50000),
            'cell_2': api.Budget(budget=50000),
        },
    )
    design_set = design.run_design(design_data, design_config, constraints)

    self.assertNotEmpty(design_set.designs)
    design_id = design_set.design_metrics.design_id.iloc[0]
    selected_design = design_set.designs[design_id]

    # Verify design results.
    self.assertLen(selected_design.control_geos, 81)

    # Cell 1 (HOLDBACK) assertions.
    self.assertLen(selected_design.designs['cell_1'].treatment_geos, 16)
    np.testing.assert_allclose(
        selected_design.designs['cell_1'].minimum_detectable_effect,
        0.0357,
        rtol=0.01,
    )
    np.testing.assert_allclose(
        selected_design.designs['cell_1'].p_value, 0.1909, rtol=0.01
    )
    np.testing.assert_allclose(
        selected_design.designs['cell_1'].budget, 9310.8, rtol=0.01
    )
    treatment_conversion_pct_1 = self._compute_treatment_conversion_pct(
        selected_design, 'cell_1'
    )
    np.testing.assert_allclose(treatment_conversion_pct_1, 0.144, rtol=0.01)

    # Cell 2 (HOLDBACK) assertions.
    self.assertLen(selected_design.designs['cell_2'].treatment_geos, 16)
    np.testing.assert_allclose(
        selected_design.designs['cell_2'].minimum_detectable_effect,
        0.0342,
        rtol=0.01,
    )
    np.testing.assert_allclose(
        selected_design.designs['cell_2'].p_value, 0.9764, rtol=0.01
    )
    np.testing.assert_allclose(
        selected_design.designs['cell_2'].budget, 9077.2, rtol=0.01
    )
    treatment_conversion_pct_2 = self._compute_treatment_conversion_pct(
        selected_design, 'cell_2'
    )
    np.testing.assert_allclose(treatment_conversion_pct_2, 0.148, rtol=0.01)

  def test_multicell_stratified_sampling_analysis_holdback_holdback(self):
    # 1. Load analysis data.
    analysis_data = pd.read_csv(
        self._get_data_path(
            'example_analysis_data_multi_cell_holdback_holdback.csv'
        )
    )
    analysis_data['date'] = pd.to_datetime(analysis_data['date'])
    analysis_data['location'] = analysis_data['location'].astype(str)

    # 2. Load design from pre-saved JSON.
    json_path = self._get_data_path(
        'example_multicell_design_stratified_sampling_holdback_holdback.json'
    )
    with open(json_path, 'r') as f:
      selected_design = api.Design.load_from_json(f.read())

    # 3. Run multicell experiment analysis.
    analysis_config = api.AnalysisConfig(
        design=selected_design,
        analysis_start_date=pd.Timestamp('2020-04-01'),
        analysis_end_date=pd.Timestamp('2020-04-30'),
        alpha=0.1,
    )

    analysis_result = analysis.analyze(analysis_data, analysis_config)
    self.assertIn('cell_1', analysis_result.results)
    self.assertIn('cell_2', analysis_result.results)

    # Verify Cell 1 (HOLDBACK) analysis results.
    metrics_1 = analysis_result.results['cell_1']
    np.testing.assert_allclose(
        metrics_1.lift.point_estimate, 154028.64, rtol=0.01
    )
    np.testing.assert_allclose(metrics_1.lift.p_value, 0.002, rtol=0.01)
    np.testing.assert_allclose(
        metrics_1.percent_lift.point_estimate, 1.6417, rtol=0.01
    )

    # Verify Cell 2 (HOLDBACK) analysis results.
    metrics_2 = analysis_result.results['cell_2']
    np.testing.assert_allclose(
        metrics_2.lift.point_estimate, 159344.88, rtol=0.01
    )
    np.testing.assert_allclose(metrics_2.lift.p_value, 0.002, rtol=0.01)
    np.testing.assert_allclose(
        metrics_2.percent_lift.point_estimate, 1.7058, rtol=0.01
    )


if __name__ == '__main__':
  absltest.main()
