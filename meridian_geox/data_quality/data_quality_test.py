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

"""Tests for the data quality check functionality."""

import datetime
from absl.testing import absltest
from meridian_geox import api
from meridian_geox.data_quality import data_quality
import pandas as pd


class DataQualityTest(absltest.TestCase):

  def test_check_design_data_quality_too_many_geos(self):
    config = api.QualityCheckConfig()
    design_config = api.DesignConfig(
        experiment_duration=datetime.timedelta(days=1)
    )
    locations = [f'G{i}' for i in range(501)]
    data = pd.DataFrame({
        'location': locations,
        'date': ['2023-01-01'] * 501,
        'conversions': [10.0] * 501,
    })
    result = data_quality.check_design_data_quality(data, design_config, config)
    self.assertIn('Too Many Geos', result.quality_metrics['metric'].values)

  def test_check_design_data_quality_missing_conversion_days_exceeds(self):
    config = api.QualityCheckConfig()
    design_config = api.DesignConfig(
        experiment_duration=datetime.timedelta(days=1)
    )
    # 10 expected days: 2023-01-01 to 2023-01-10.
    # Provide only 6 unique dates (Jan 1-5, and Jan 10).
    # missing pct: 1.0 - 6/10 = 0.40 (40%) > threshold (30%).
    dates = pd.date_range('2023-01-01', periods=5).tolist()
    data = pd.DataFrame({
        'location': ['G1'] * 5 + ['G2', 'G2'],
        'date': (
            dates + [pd.Timestamp('2023-01-01'), pd.Timestamp('2023-01-10')]
        ),
        'conversions': [10.0] * 7,
    })
    result = data_quality.check_design_data_quality(data, design_config, config)
    self.assertIn(
        'Percentage of missing conversion days',
        result.quality_metrics['metric'].values,
    )
    row = result.quality_metrics[
        result.quality_metrics['metric']
        == 'Percentage of missing conversion days'
    ].iloc[0]
    self.assertGreater(row['value'], 0.3)

  def test_check_design_data_quality_missing_conversion_days_under_threshold(
      self,
  ):
    config = api.QualityCheckConfig()
    design_config = api.DesignConfig(
        experiment_duration=datetime.timedelta(days=1)
    )
    # 10 expected days: 2023-01-01 to 2023-01-10.
    # Provide 9 unique dates.
    # missing pct: 1.0 - 9/10 = 0.10 (10%) <= threshold (30%).
    dates = pd.date_range('2023-01-01', periods=9).tolist()
    data = pd.DataFrame({
        'location': ['G1'] * 9 + ['G2'],
        'date': dates + [pd.Timestamp('2023-01-10')],
        'conversions': [10.0] * 10,
    })
    result = data_quality.check_design_data_quality(data, design_config, config)
    self.assertNotIn(
        'Percentage of missing conversion days',
        result.quality_metrics['metric'].values,
    )

  def test_check_analysis_data_quality_missing_conversion_days_pretest_only(
      self,
  ):
    config = api.QualityCheckConfig()
    # 10 days total. pretest period is before Jan 8 (7 expected days: Jan 1
    # to Jan 7).
    # Unique dates in pretest is 4.
    # 1.0 - 4/7 = 42.8% missing > 30% -> warning triggers.
    pretest_dates = [
        '2023-01-01',
        '2023-01-02',
        '2023-01-03',
        '2023-01-07',
    ]
    test_dates = ['2023-01-08', '2023-01-09', '2023-01-10']
    data = pd.DataFrame({
        'location': ['G1'] * len(pretest_dates + test_dates),
        'date': pretest_dates + test_dates,
        'conversions': [10.0] * len(pretest_dates + test_dates),
        'spend': [1.0] * len(pretest_dates + test_dates),
    })

    design_config = api.DesignConfig(
        experiment_duration=datetime.timedelta(days=3),
        experiment_types={'cell_1': api.ExperimentType.GO_DARK},
    )
    design = api.Design(
        designs={
            'cell_1': api.PerCellDesign(
                treatment_geos={'G1'},
                minimum_detectable_effect=0.5,
                design_implied_cpic=1.0,
                p_value=0.05,
                budget=100.0,
            )
        },
        control_geos=set(),
        excluded_geos=set(),
        design_config=design_config,
    )
    analysis_config = api.AnalysisConfig(
        design=design,
        analysis_start_date=pd.Timestamp('2023-01-08'),
        analysis_end_date=pd.Timestamp('2023-01-10'),
    )

    result = data_quality.check_analysis_data_quality(
        data, analysis_config, config
    )
    self.assertIn(
        'Percentage of missing conversion days',
        result.quality_metrics['metric'].values,
    )
    row = result.quality_metrics[
        result.quality_metrics['metric']
        == 'Percentage of missing conversion days'
    ].iloc[0]
    self.assertGreater(row['value'], 0.3)

  def test_check_analysis_data_quality_missing_spend_days_pretest_only(self):
    config = api.QualityCheckConfig()
    # 10 days total. pretest period is before Jan 8 (7 expected days: Jan 1
    # to Jan 7).
    # Unique dates in pretest is 4.
    # 1.0 - 4/7 = 42.8% missing > 30% -> warning triggers.
    pretest_dates = [
        '2023-01-01',
        '2023-01-02',
        '2023-01-03',
        '2023-01-07',
    ]
    test_dates = ['2023-01-08', '2023-01-09', '2023-01-10']
    data = pd.DataFrame({
        'location': ['G1'] * len(pretest_dates + test_dates),
        'date': pretest_dates + test_dates,
        'conversions': [10.0] * len(pretest_dates + test_dates),
        'spend': [1.0] * len(pretest_dates + test_dates),
    })

    # If GO_DARK: checks pretest period -> warning triggers.
    design_config_go_dark = api.DesignConfig(
        experiment_duration=datetime.timedelta(days=3),
        experiment_types={'cell_1': api.ExperimentType.GO_DARK},
    )
    design_go_dark = api.Design(
        designs={
            'cell_1': api.PerCellDesign(
                treatment_geos={'G1'},
                minimum_detectable_effect=0.5,
                design_implied_cpic=1.0,
                p_value=0.05,
                budget=100.0,
            )
        },
        control_geos=set(),
        excluded_geos=set(),
        design_config=design_config_go_dark,
    )
    analysis_config_go_dark = api.AnalysisConfig(
        design=design_go_dark,
        analysis_start_date=pd.Timestamp('2023-01-08'),
        analysis_end_date=pd.Timestamp('2023-01-10'),
    )

    result_go_dark = data_quality.check_analysis_data_quality(
        data,
        analysis_config_go_dark,
        config,
    )
    self.assertIn(
        'Percentage of missing spend days (cell_1)',
        result_go_dark.quality_metrics['metric'].values,
    )

    # If HOLDBACK: skips the check -> no warning.
    design_config_holdback = api.DesignConfig(
        experiment_duration=datetime.timedelta(days=3),
        experiment_types={'cell_1': api.ExperimentType.HOLDBACK},
    )
    design_holdback = api.Design(
        designs={
            'cell_1': api.PerCellDesign(
                treatment_geos={'G1'},
                minimum_detectable_effect=0.5,
                design_implied_cpic=1.0,
                p_value=0.05,
                budget=100.0,
            )
        },
        control_geos=set(),
        excluded_geos=set(),
        design_config=design_config_holdback,
    )
    analysis_config_holdback = api.AnalysisConfig(
        design=design_holdback,
        analysis_start_date=pd.Timestamp('2023-01-08'),
        analysis_end_date=pd.Timestamp('2023-01-10'),
    )

    result_holdback = data_quality.check_analysis_data_quality(
        data,
        analysis_config_holdback,
        config,
    )
    self.assertNotIn(
        'Percentage of missing spend days (cell_1)',
        result_holdback.quality_metrics['metric'].values,
    )

  def test_check_design_data_quality_missing_spend_days_pretest_design_phase(
      self,
  ):
    config = api.QualityCheckConfig()
    design_config = api.DesignConfig(
        experiment_duration=datetime.timedelta(days=3),
        experiment_types={'cell_1': api.ExperimentType.GO_DARK},
    )
    # In design phase, the entire data is pretest.
    # 10 expected days (Jan 1-10). Only 5 unique dates present.
    # missing pct: 50% > 30% -> warning triggers.
    dates = [
        '2023-01-01',
        '2023-01-02',
        '2023-01-03',
        '2023-01-04',
        '2023-01-10',
    ]
    data = pd.DataFrame({
        'location': ['G1'] * len(dates),
        'date': dates,
        'conversions': [10.0] * len(dates),
        'spend': [1.0] * len(dates),
    })
    result = data_quality.check_design_data_quality(
        data,
        design_config,
        config,
    )
    self.assertIn(
        'Percentage of missing spend days (cell_1)',
        result.quality_metrics['metric'].values,
    )

  def test_check_design_data_quality_duplicate_entries(self):
    config = api.QualityCheckConfig()
    design_config = api.DesignConfig(
        experiment_duration=datetime.timedelta(days=1)
    )
    data = pd.DataFrame({
        'location': ['G1', 'G1', 'G1', 'G2'],
        'date': ['2023-01-01', '2023-01-01', '2023-01-01', '2023-01-02'],
        'conversions': [10.0, 10.0, 10.0, 10.0],
    })
    result = data_quality.check_design_data_quality(data, design_config, config)
    self.assertIn('Duplicate Entries', result.quality_metrics['metric'].values)
    row = result.quality_metrics[
        result.quality_metrics['metric'] == 'Duplicate Entries'
    ].iloc[0]
    self.assertEqual(row['value'], 2)

  def test_check_design_data_quality_exclude_geos_no_response(self):
    config = api.QualityCheckConfig(exclude_geos_no_response=True)
    design_config = api.DesignConfig(
        experiment_duration=datetime.timedelta(days=1)
    )
    data = pd.DataFrame({
        'location': ['G1', 'G2'],
        'date': ['2023-01-01', '2023-01-01'],
        'conversions': [0.0, 10.0],
        'spend': [10.0, 10.0],
    })
    result = data_quality.check_design_data_quality(data, design_config, config)
    self.assertIn(
        'Spend > 0 and no conversions', result.quality_metrics['metric'].values
    )
    self.assertIn('G1', result.outlier_geos)

  def test_check_design_data_quality_exclude_geos_no_response_false(self):
    config = api.QualityCheckConfig(exclude_geos_no_response=False)
    design_config = api.DesignConfig(
        experiment_duration=datetime.timedelta(days=1)
    )
    data = pd.DataFrame({
        'location': ['G1', 'G2'],
        'date': ['2023-01-01', '2023-01-01'],
        'conversions': [0.0, 10.0],
        'spend': [10.0, 10.0],
    })
    result = data_quality.check_design_data_quality(data, design_config, config)
    self.assertIn('G1', result.outlier_geos)
    self.assertIn(
        'Spend > 0 and no conversions', result.quality_metrics['metric'].values
    )

  def test_check_design_data_quality_high_zero_response(self):
    config = api.QualityCheckConfig()
    design_config = api.DesignConfig(
        experiment_duration=datetime.timedelta(days=1)
    )
    data = pd.DataFrame({
        'location': ['G1', 'G1', 'G2', 'G2'],
        'date': ['2023-01-01', '2023-01-02', '2023-01-01', '2023-01-02'],
        'conversions': [0.0, 0.0, 10.0, 0.0],
    })
    result = data_quality.check_design_data_quality(data, design_config, config)
    self.assertIn(
        'Percentage of zero response', result.quality_metrics['metric'].values
    )

  def test_check_design_data_quality_multicell_spend(self):
    config = api.QualityCheckConfig()
    design_config = api.DesignConfig(
        experiment_duration=datetime.timedelta(days=3),
        cell_count=2,
        experiment_types={
            'cell_1': api.ExperimentType.GO_DARK,
            'cell_2': api.ExperimentType.HEAVY_UP,
        },
    )
    pretest_dates = ['2023-01-01', '2023-01-02', '2023-01-07']
    data = pd.DataFrame({
        'location': ['G1'] * len(pretest_dates),
        'date': pretest_dates,
        'conversions': [10.0] * len(pretest_dates),
        'spend_cell_1': [5.0] * len(pretest_dates),
        'spend_cell_2': [15.0] * len(pretest_dates),
    })
    result = data_quality.check_design_data_quality(
        data,
        design_config,
        config,
    )
    self.assertIn(
        'Percentage of missing spend days (cell_1)',
        result.quality_metrics['metric'].values,
    )
    self.assertIn(
        'Percentage of missing spend days (cell_2)',
        result.quality_metrics['metric'].values,
    )

  def test_check_design_data_quality_with_outlier_dates(self):
    config = api.QualityCheckConfig()
    design_config = api.DesignConfig(
        experiment_duration=datetime.timedelta(days=3)
    )
    dates = (
        pd.date_range('2023-01-01', periods=10).strftime('%Y-%m-%d').tolist()
    )
    conversions = [10.0] * 10
    conversions[5] = 1000.0

    data = pd.DataFrame({
        'location': ['G1'] * 10,
        'date': dates,
        'conversions': conversions,
    })
    result = data_quality.check_design_data_quality(data, design_config, config)
    self.assertIn(pd.Timestamp('2023-01-06'), result.outlier_dates)
    self.assertIn(
        'Outlier pretest dates', result.quality_metrics['metric'].values
    )

  def test_check_design_data_quality_no_outlier_dates(self):
    config = api.QualityCheckConfig()
    design_config = api.DesignConfig(
        experiment_duration=datetime.timedelta(days=3)
    )
    dates = (
        pd.date_range('2023-01-01', periods=10).strftime('%Y-%m-%d').tolist()
    )
    conversions = [10.0 + 2.0 * i for i in range(10)]

    data = pd.DataFrame({
        'location': ['G1'] * 10,
        'date': dates,
        'conversions': conversions,
    })
    result = data_quality.check_design_data_quality(data, design_config, config)
    self.assertEmpty(result.outlier_dates)
    self.assertNotIn(
        'Outlier pretest dates', result.quality_metrics['metric'].values
    )

  def test_check_design_data_quality_exclude_outlier_dates_false(self):
    config = api.QualityCheckConfig(exclude_outlier_dates=False)
    design_config = api.DesignConfig(
        experiment_duration=datetime.timedelta(days=3)
    )
    dates = (
        pd.date_range('2023-01-01', periods=10).strftime('%Y-%m-%d').tolist()
    )
    conversions = [10.0] * 10
    conversions[5] = 1000.0

    data = pd.DataFrame({
        'location': ['G1'] * 10,
        'date': dates,
        'conversions': conversions,
    })
    result = data_quality.check_design_data_quality(data, design_config, config)
    self.assertIn(pd.Timestamp('2023-01-06'), result.outlier_dates)
    self.assertIn(
        'Outlier pretest dates', result.quality_metrics['metric'].values
    )

  def test_check_analysis_data_quality_ignores_outlier_dates(self):
    config = api.QualityCheckConfig(exclude_outlier_dates=True)
    dates = (
        pd.date_range('2023-01-01', periods=10).strftime('%Y-%m-%d').tolist()
    )
    conversions = [10.0] * 10
    conversions[5] = 1000.0

    data = pd.DataFrame({
        'location': ['G1'] * 10,
        'date': dates,
        'conversions': conversions,
        'spend': [1.0] * 10,
    })

    design_config = api.DesignConfig(
        experiment_duration=datetime.timedelta(days=3),
        experiment_types={'cell_1': api.ExperimentType.GO_DARK},
    )
    design = api.Design(
        designs={
            'cell_1': api.PerCellDesign(
                treatment_geos={'G1'},
                minimum_detectable_effect=0.5,
                design_implied_cpic=1.0,
                p_value=0.05,
                budget=100.0,
            )
        },
        control_geos=set(),
        excluded_geos=set(),
        design_config=design_config,
    )
    analysis_config = api.AnalysisConfig(
        design=design,
        analysis_start_date=pd.Timestamp('2023-01-08'),
        analysis_end_date=pd.Timestamp('2023-01-10'),
    )

    result = data_quality.check_analysis_data_quality(
        data, analysis_config, config
    )
    self.assertEmpty(result.outlier_dates)
    self.assertNotIn(
        'Outlier pretest dates', result.quality_metrics['metric'].values
    )


if __name__ == '__main__':
  absltest.main()
