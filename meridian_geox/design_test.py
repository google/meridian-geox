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

import itertools

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
from meridian_geox import api
from meridian_geox import design
from meridian_geox import generate_candidates
import numpy as np
import pandas as pd


class DesignTest(parameterized.TestCase):

  @parameterized.named_parameters(
      dict(
          testcase_name='two_sided',
          test_type=api.TestType.TWO_SIDED,
      ),
      dict(
          testcase_name='one_sided',
          test_type=api.TestType.ONE_SIDED,
      ),
  )
  def test_tbr_random_design(self, test_type):
    # Create dummy data with 20 locations and 30 days.
    dates = pd.date_range(start='2023-01-01', periods=30)
    locations = [f'geo_{i}' for i in range(20)]
    data_list = []

    for date, loc in itertools.product(dates, locations):
      data_list.append({
          api.DATE: date,
          api.LOCATION: loc,
          api.CONVERSIONS: (
              np.random.rand() * 100 + 10
          ),  # Ensure positive conversions
      })
    data = pd.DataFrame(data_list)

    design_config = api.DesignConfig(
        experiment_duration=5,
        methodology=api.Methodology.TBR,
        geo_assignment_rule=api.GeoAssignmentRule.RANDOM,
        n_candidates=10,
        n_ranked_candidates=5,
        n_aa_test_iterations=10,
        design_output_count=2,
        seed=42,
        test_type=test_type,
    )

    constraints = api.Constraints()

    result = design.run_design(data, design_config, constraints)

    self.assertIsInstance(result, api.DesignSet)
    self.assertNotEmpty(result.designs)
    self.assertLessEqual(len(result.designs), 2)

    for _, design_obj in result.designs.items():
      self.assertIsInstance(design_obj, api.Design)

      self.assertIn('cell_1', design_obj.treatment_geos)
      treatment_geos = design_obj.treatment_geos['cell_1']
      self.assertNotEmpty(treatment_geos)
      self.assertNotEmpty(design_obj.control_geos)

      # Check treatment and control are disjoint.
      self.assertTrue(treatment_geos.isdisjoint(design_obj.control_geos))

      # Check that union of treatment and control is subset of all locations.
      all_design_geos = treatment_geos.union(design_obj.control_geos)
      self.assertTrue(all_design_geos.issubset(set(locations)))

    self.assertFalse(result.design_metrics.empty)
    self.assertIn('mde_pct', result.design_metrics.columns)
    self.assertIn('p_value', result.design_metrics.columns)
    self.assertIn('cell_id', result.design_metrics.columns)
    self.assertIn('design_methodology', result.design_metrics.columns)
    self.assertTrue((result.design_metrics['cell_id'] == 'cell_1').all())
    self.assertTrue(
        (result.design_metrics['design_methodology'] == 'RANDOM-TBR').all()
    )

  def test_random_design_min_geos_requirement(self):
    dates = pd.date_range(start='2023-01-01', periods=6)
    design_config = api.DesignConfig(
        experiment_duration=2,
        geo_assignment_rule=api.GeoAssignmentRule.RANDOM,
    )
    constraints = api.Constraints()

    # 3 geos: 2 treatment + 2 control = 4 required for RANDOM.
    locations = ['geo_1', 'geo_2', 'geo_3']
    data_list = []
    for d, l in itertools.product(dates, locations):
      data_list.append({api.DATE: d, api.LOCATION: l, api.CONVERSIONS: 10.0})
    data_3 = pd.DataFrame(data_list)
    with self.assertRaisesRegex(ValueError, 'Not enough geos.'):
      design.run_design(data_3, design_config, constraints)

    # 4 geos, but max_conversions_percent=0.8 means n_treated=3,
    # so n_control=1, which is < 2.
    locations_2 = ['geo_1', 'geo_2', 'geo_3', 'geo_4']
    data_list_2 = []
    for d, l in itertools.product(dates, locations_2):
      data_list_2.append({api.DATE: d, api.LOCATION: l, api.CONVERSIONS: 10.0})
    data_2 = pd.DataFrame(data_list_2)
    constraints_2 = api.Constraints(max_conversions_percent=0.8)
    with self.assertRaisesRegex(ValueError, 'Not enough geos.'):
      design.run_design(data_2, design_config, constraints_2)

  def test_design_small_n_geos(self):
    # 4 geos.
    dates = pd.date_range(start='2023-01-01', periods=10)
    locations = ['geo_1', 'geo_2', 'geo_3', 'geo_4']
    data_list = []
    for d, l in itertools.product(dates, locations):
      data_list.append({api.DATE: d, api.LOCATION: l, api.CONVERSIONS: 10.0})
    data = pd.DataFrame(data_list)
    design_config = api.DesignConfig(experiment_duration=2, n_candidates=5)
    constraints = api.Constraints(max_conversions_percent=0.5)

    result = design.run_design(data, design_config, constraints)

    for _, design_obj in result.designs.items():
      # n_treated should be at least 2 and at most n_geos - 2.
      # With 4 geos, 0.5 * 4 = 2 treated.
      t_geos = design_obj.treatment_geos['cell_1']
      c_geos = design_obj.control_geos
      self.assertLen(t_geos, 2)
      self.assertLen(c_geos, 2)

  def test_design_excluded_geos(self):
    dates = pd.date_range(start='2023-01-01', periods=30)
    locations = ['geo_1', 'geo_2', 'geo_3', 'geo_4', 'geo_5']
    data_list = []
    for date, loc in itertools.product(dates, locations):
      data_list.append({
          api.DATE: date,
          api.LOCATION: loc,
          api.CONVERSIONS: 100.0,
      })
    data = pd.DataFrame(data_list)

    design_config = api.DesignConfig(experiment_duration=5, n_candidates=10)
    constraints = api.Constraints(
        excluded_geos={'geo_1'}, max_conversions_percent=0.5
    )

    result = design.run_design(data, design_config, constraints)

    for _, design_obj in result.designs.items():
      # geo_1 should not be in treatment or control.
      self.assertNotIn('geo_1', design_obj.treatment_geos['cell_1'])
      self.assertNotIn('geo_1', design_obj.control_geos)
      # geo_1 should be in excluded_geos.
      self.assertIn('geo_1', design_obj.excluded_geos)

  @parameterized.named_parameters(
      dict(
          testcase_name='random',
          rule=api.GeoAssignmentRule.RANDOM,
      ),
      dict(
          testcase_name='stratified_sampling',
          rule=api.GeoAssignmentRule.STRATIFIED_SAMPLING,
      ),
  )
  def test_design_reproducibility(self, rule):
    dates = pd.date_range(start='2023-01-01', periods=30)
    locations = [f'geo_{i}' for i in range(10)]
    data_list = []
    for d, (i, l) in itertools.product(dates, enumerate(locations)):
      # Add deterministic variation to help with stratified sampling.
      val = 100.0 + i * 10.0 + (d.day % 7) * 5.0
      data_list.append({api.DATE: d, api.LOCATION: l, api.CONVERSIONS: val})
    data = pd.DataFrame(data_list)

    config1 = api.DesignConfig(
        experiment_duration=5,
        seed=42,
        n_candidates=20,
        n_ranked_candidates=10,
        geo_assignment_rule=rule,
    )
    config2 = api.DesignConfig(
        experiment_duration=5,
        seed=42,
        n_candidates=20,
        n_ranked_candidates=10,
        geo_assignment_rule=rule,
    )

    result1 = design.run_design(data, config1, api.Constraints())
    result2 = design.run_design(data, config2, api.Constraints())

    # Check that design metrics are identical.
    pd.testing.assert_frame_equal(
        result1.design_metrics.drop(columns=['design_id']),
        result2.design_metrics.drop(columns=['design_id']),
    )

  def test_cluster_geos(self):
    training_period = [
        pd.Timestamp('2024-01-01'),
        pd.Timestamp('2024-01-02'),
        pd.Timestamp('2024-01-03'),
        pd.Timestamp('2024-01-04'),
        pd.Timestamp('2024-01-05'),
    ]
    # long format data.
    data_list = []
    for date in training_period:
      for i, geo in enumerate(['G0', 'G1', 'G2', 'G3']):
        val = [1.0, 1.0, 2.0, 2.0][i]
        data_list.append(
            {api.DATE: date, api.LOCATION: geo, api.CONVERSIONS: val}
        )
    conversions_data = pd.DataFrame(data_list)
    design_config = api.DesignConfig(
        experiment_duration=1, num_strata=2, k_means_iterations=10
    )
    # Create selection_train data matching conversions_data
    # Shape should be (n_dates, n_geos) = (5, 4)
    selection_train_values = []
    for _ in range(5):
      selection_train_values.append([1.0, 1.0, 2.0, 2.0])
    selection_train = jnp.array(selection_train_values)

    key = jax.random.PRNGKey(42)
    processed_data = design.ProcessedData(
        selection_train=selection_train,
        selection_eval=jnp.array([]),
        estimation_train=jnp.array([]),
        estimation_eval=jnp.array([]),
        training_period=training_period,
        filtered_data=conversions_data,
    )
    result = generate_candidates.cluster_geos(
        processed_data, design_config, key
    )
    means = result.means
    labels = result.labels
    self.assertEqual(means.shape, (2, 5))
    self.assertEqual(labels.shape, (4,))
    self.assertContainsSubset(set(np.asarray(labels)), {0, 1})
    np.testing.assert_array_equal(labels, jnp.array([0, 0, 1, 1]))

  def test_design_included_control_geos(self):
    dates = pd.date_range(start='2023-01-01', periods=30)
    locations = [f'geo_{i}' for i in range(10)]
    data_list = []
    for d, l in itertools.product(dates, locations):
      data_list.append({
          api.DATE: d,
          api.LOCATION: l,
          api.CONVERSIONS: 100.0,
      })
    data = pd.DataFrame(data_list)

    design_config = api.DesignConfig(
        experiment_duration=5, n_candidates=10, seed=42
    )
    # Force geo_0 and geo_1 to be in control.
    included_control = {'geo_0', 'geo_1'}
    constraints = api.Constraints(
        included_control_geos=included_control, max_conversions_percent=0.5
    )

    result = design.run_design(data, design_config, constraints)

    self.assertNotEmpty(result.designs)
    for _, design_obj in result.designs.items():
      for geo in included_control:
        self.assertIn(geo, design_obj.control_geos)
        self.assertNotIn(geo, design_obj.treatment_geos['cell_1'])

  def test_design_max_conversions_percent(self):
    dates = pd.date_range(start='2023-01-01', periods=30)
    # 10 locations.
    # geo_0 has 60% of volume.
    # geo_1..9 share 40% of volume.
    data_list = []
    for d in dates:
      data_list.append(
          {api.DATE: d, api.LOCATION: 'geo_0', api.CONVERSIONS: 60.0}
      )
      for i in range(1, 10):
        data_list.append(
            {api.DATE: d, api.LOCATION: f'geo_{i}', api.CONVERSIONS: 40.0 / 9}
        )
    data = pd.DataFrame(data_list)

    # Max conversions percent = 0.5.
    # geo_0 (60%) must be excluded from treatment.
    design_config = api.DesignConfig(
        experiment_duration=5, n_candidates=10, seed=42
    )
    constraints = api.Constraints(max_conversions_percent=0.5)

    result = design.run_design(data, design_config, constraints)

    self.assertNotEmpty(result.designs)
    for _, design_obj in result.designs.items():
      self.assertNotIn('geo_0', design_obj.treatment_geos['cell_1'])
      self.assertIn('geo_0', design_obj.control_geos)

  def test_filter_results_by_aa_test(self):
    design_config = api.DesignConfig(
        experiment_duration=5,
        alpha=0.1,
        design_output_count=2,
    )
    scored_candidates = design.ScoredCandidates(
        candidates=jnp.array([[0, 0, 1, 1], [1, 1, 0, 0], [1, 0, 1, 0]]),
        mde_abs=jnp.array([1.0, 2.0, 3.0]),
        mde_pct=jnp.array([0.1, 0.2, 0.3]),
        p_values=jnp.array([0.05, 0.15, 0.2]),  # Candidate 0 fails (p < alpha)
        r2_scores=jnp.array([0.9, 0.8, 0.7]),
    )

    # p_values >= alpha (0.1).
    # 0.05 >= 0.1 -> False
    # 0.15 >= 0.1 -> True
    # 0.2 >= 0.1 -> True

    filtered_results = design._filter_results_by_aa_test(
        scored_candidates, design_config
    )

    self.assertLen(filtered_results.candidates, 2)
    np.testing.assert_array_equal(
        filtered_results.candidates, scored_candidates.candidates[1:]
    )
    np.testing.assert_array_equal(
        filtered_results.p_values, scored_candidates.p_values[1:]
    )

  def test_filter_results_by_aa_test_all_fail(self):
    design_config = api.DesignConfig(
        experiment_duration=5,
        alpha=0.1,
    )
    scored_candidates = design.ScoredCandidates(
        candidates=jnp.array([[0, 0, 1, 1]]),
        mde_abs=jnp.array([1.0]),
        mde_pct=jnp.array([0.1]),
        p_values=jnp.array([0.05]),  # Fails (p < alpha)
        r2_scores=jnp.array([0.9]),
    )

    with self.assertRaisesRegex(ValueError, 'No designs passed the A/A test'):
      design._filter_results_by_aa_test(scored_candidates, design_config)

  def test_design_json_serialization(self):
    constraints = api.Constraints(
        included_treatment_geos={'G1', 'G2'},
        excluded_dates={
            pd.Timestamp('2024-01-01'),
            pd.Timestamp('2024-01-02'),
        },
        max_conversions_percent=0.5,
    )
    design_config = api.DesignConfig(
        experiment_duration=5,
        experiment_types=[
            api.ExperimentType.HOLDBACK,
            api.ExperimentType.GO_DARK,
        ],
        alpha=0.05,
    )
    original_design = api.Design(
        treatment_geos={'cell_1': {'G1', 'G2'}},
        control_geos={'G3', 'G4'},
        excluded_geos={'G5'},
        minimum_detectable_effect={'cell_1': 0.1},
        p_value={'cell_1': 0.5},
        budget={'cell_1': 1000.0},
        design_config=design_config,
        constraints=constraints,
        geo_stratum_labels=jnp.array([0, 1, 0, 1, 2]),
    )

    json_str = original_design.export_to_json()
    loaded_design = api.Design.load_from_json(json_str)

    self.assertEqual(loaded_design.design_config, original_design.design_config)
    self.assertEqual(loaded_design.constraints, original_design.constraints)
    self.assertEqual(
        loaded_design.treatment_geos, original_design.treatment_geos
    )
    self.assertEqual(loaded_design.control_geos, original_design.control_geos)
    self.assertEqual(loaded_design.excluded_geos, original_design.excluded_geos)
    self.assertEqual(
        loaded_design.minimum_detectable_effect,
        original_design.minimum_detectable_effect,
    )
    self.assertEqual(loaded_design.p_value, original_design.p_value)
    self.assertEqual(loaded_design.budget, original_design.budget)

    np.testing.assert_array_equal(
        loaded_design.geo_stratum_labels, original_design.geo_stratum_labels
    )

  def test_concat_design_reports(self):
    design_config = api.DesignConfig(experiment_duration=5)
    constraints = api.Constraints()

    d1_id = 'd1'
    d1 = api.Design(
        treatment_geos={'cell_1': {'geo_1'}},
        control_geos={'geo_2'},
        excluded_geos=set(),
        design_config=design_config,
        constraints=constraints,
    )
    ds1_metrics = pd.DataFrame([{
        'design_id': d1_id,
        'mde_pct': 0.8,
        'design_methodology': 'RANDOM-TBR',
    }])
    ds1 = api.DesignSet(
        designs={d1_id: d1},
        design_metrics=ds1_metrics,
        design_data=pd.DataFrame(),
    )

    d2_id = 'd2'
    d2 = api.Design(
        treatment_geos={'cell_1': {'geo_3'}},
        control_geos={'geo_4'},
        excluded_geos=set(),
        design_config=design_config,
        constraints=constraints,
    )
    ds2_metrics = pd.DataFrame([{
        'design_id': d2_id,
        'mde_pct': 0.9,
        'design_methodology': 'RANDOM-TBR',
    }])
    ds2 = api.DesignSet(
        designs={d2_id: d2},
        design_metrics=ds2_metrics,
        design_data=pd.DataFrame(),
    )

    # Concatenate and keep top 1.
    result = design.concat_design_reports([ds1, ds2], design_output_count=1)

    self.assertLen(result.designs, 1)
    self.assertIn(d1_id, result.designs)
    self.assertEqual(result.design_metrics.iloc[0]['design_id'], d1_id)
    self.assertEqual(result.design_metrics.iloc[0]['mde_pct'], 0.8)
    self.assertEqual(
        result.design_metrics.iloc[0]['design_methodology'], 'RANDOM-TBR'
    )

  def test_compare_designs(self):
    data = pd.DataFrame()
    config1 = api.DesignConfig(experiment_duration=5, seed=42)
    config2 = api.DesignConfig(experiment_duration=5, seed=43)
    requirements = [(config1, api.Constraints()), (config2, api.Constraints())]

    d1_id = 'd1'
    d1 = api.Design(
        treatment_geos={'cell_1': {'geo_1'}},
        control_geos={'geo_2'},
        excluded_geos=set(),
    )
    ds1_metrics = pd.DataFrame([{
        'design_id': d1_id,
        'mde_pct': 0.8,
        'design_methodology': 'RANDOM-TBR',
    }])
    ds1 = api.DesignSet(
        designs={d1_id: d1},
        design_metrics=ds1_metrics,
        design_data=pd.DataFrame(),
    )

    d2_id = 'd2'
    d2 = api.Design(
        treatment_geos={'cell_1': {'geo_3'}},
        control_geos={'geo_4'},
        excluded_geos=set(),
    )
    ds2_metrics = pd.DataFrame([{
        'design_id': d2_id,
        'mde_pct': 0.7,
        'design_methodology': 'RANDOM-TBR',
    }])
    ds2 = api.DesignSet(
        designs={d2_id: d2},
        design_metrics=ds2_metrics,
        design_data=pd.DataFrame(),
    )

    with absltest.mock.patch.object(
        design, 'run_design', side_effect=[ds1, ds2]
    ) as mock_design:
      result = design.compare_designs(data, requirements, design_output_count=2)

    self.assertEqual(mock_design.call_count, 2)
    self.assertIsInstance(result, api.DesignSet)
    self.assertLen(result.designs, 2)
    self.assertLen(result.design_metrics, 2)
    # Check that they are sorted by mde_pct (0.7 < 0.8).
    self.assertEqual(result.design_metrics.iloc[0]['design_id'], d2_id)
    self.assertEqual(result.design_metrics.iloc[0]['mde_pct'], 0.7)
    self.assertEqual(result.design_metrics.iloc[1]['design_id'], d1_id)
    self.assertEqual(result.design_metrics.iloc[1]['mde_pct'], 0.8)

  @parameterized.named_parameters(
      dict(
          testcase_name='holdback',
          experiment_types=api.ExperimentType.HOLDBACK,
          budget_constraint=None,
          budget_percent_constraint=None,
          has_spend=True,
          expected_budget=0.1 * 100 * 1.0,  # mde_pct * volume * cpic
      ),
      dict(
          testcase_name='go_dark_no_constraints',
          experiment_types=api.ExperimentType.GO_DARK,
          budget_constraint=None,
          budget_percent_constraint=None,
          has_spend=True,
          expected_budget=50.0,  # treatment_geo_cost
      ),
      dict(
          testcase_name='heavy_up_absolute_budget',
          experiment_types=api.ExperimentType.HEAVY_UP,
          budget_constraint=1000.0,
          budget_percent_constraint=None,
          has_spend=True,
          expected_budget=1000.0,
      ),
      dict(
          testcase_name='go_dark_budget_percent',
          experiment_types=api.ExperimentType.GO_DARK,
          budget_constraint=None,
          budget_percent_constraint=0.5,
          has_spend=True,
          expected_budget=50.0 * 0.5,
      ),
      dict(
          testcase_name='go_dark_missing_spend',
          experiment_types=api.ExperimentType.GO_DARK,
          budget_constraint=None,
          budget_percent_constraint=None,
          has_spend=False,
          expected_budget=0.0,
      ),
  )
  def test_get_design_summary_budget(
      self,
      experiment_types,
      budget_constraint,
      budget_percent_constraint,
      has_spend,
      expected_budget,
  ):
    scored_candidates = design.ScoredCandidates(
        # 2 control, 2 treated geos.
        candidates=jnp.array([[0, 0, 1, 1]]),
        mde_abs=jnp.array([10.0]),
        mde_pct=jnp.array([0.1]),
        p_values=jnp.array([0.5]),
        r2_scores=jnp.array([0.9]),
    )
    design_config = api.DesignConfig(
        experiment_duration=5,
        experiment_types=experiment_types,
        cost_per_incremental_conversion=1.0,
        design_output_count=1,
    )
    constraints = api.Constraints(
        budget=budget_constraint,
        budget_percent=budget_percent_constraint,
    )
    geos = ['geo_1', 'geo_2', 'geo_3', 'geo_4']
    geo_stratum_labels = jnp.array([0, 0, 1, 1])

    # 2 treated geos (geo_3 and geo_4) with 50 conversions and 25 spend each.
    # Total treated volume = 100, total treated spend = 50.
    estimation_eval = jnp.array([[0, 0, 50, 50]])
    estimation_eval_spend = jnp.array([[0, 0, 25, 25]]) if has_spend else None

    processed_data = design.ProcessedData(
        selection_train=jnp.array([]),
        selection_eval=jnp.array([]),
        estimation_train=jnp.array([]),
        estimation_eval=estimation_eval,
        training_period=[],
        filtered_data=pd.DataFrame(),
        estimation_eval_spend=estimation_eval_spend,
    )

    result = design._get_design_summary(
        scored_candidates,
        design_config,
        constraints,
        geos,
        geo_stratum_labels,
        processed_data,
    )

    self.assertLen(result.designs, 1)
    design_id = result.design_metrics.iloc[0]['design_id']
    design_obj = result.designs[design_id]
    self.assertEqual(result.design_metrics.iloc[0]['cell_id'], 'cell_1')
    self.assertIn('cell_1', design_obj.minimum_detectable_effect)
    self.assertIn('cell_1', design_obj.p_value)
    self.assertIn('cell_1', design_obj.budget)
    self.assertIn('design_methodology', result.design_metrics.columns)
    self.assertEqual(
        result.design_metrics.iloc[0]['design_methodology'], 'RANDOM-TBR'
    )
    self.assertAlmostEqual(
        result.design_metrics.iloc[0]['budget'], expected_budget, places=5
    )


if __name__ == '__main__':
  absltest.main()
