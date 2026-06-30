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

"""Data quality check library API."""

import logging
import re
from typing import Optional

from meridian_geox import api
import pandas as pd


# Warning thresholds as constants
_MAX_GEOS = 500
_MAX_ZERO_RESPONSE_PCT = 0.5
_MAX_MISSING_DAYS_PCT = 0.3
_MAX_DUPLICATE_ENTRIES = 0


def _check_data_quality(
    data: pd.DataFrame,
    quality_check_config: api.QualityCheckConfig,
    is_design: bool,
    experiment_types: Optional[dict[str, api.ExperimentType]] = None,
    pretest_end_date: Optional[pd.Timestamp] = None,
) -> api.QualityCheckResult:
  """Internal function to check the quality of the input data."""
  metrics = []
  outlier_geos = set()
  outlier_dates = set()

  # Ensure working with copies and datetime
  data = data.copy()
  data[api.DATE] = pd.to_datetime(data[api.DATE])

  unique_geos = data[api.LOCATION].nunique()

  # Too Many Geos
  if unique_geos > _MAX_GEOS:
    msg = (
        f'Exceeds {_MAX_GEOS} geographic units. High-cardinality data can '
        'introduce excessive noise and contamination issues, reducing the '
        'effectiveness of the analysis. We recommend using coarser '
        'geographic levels for more reliable results.'
    )
    logging.warning(msg)
    metrics.append({
        'metric': 'Too Many Geos',
        'value': unique_geos,
        'message': msg,
        'threshold': _MAX_GEOS,
    })

  # Define pretest data
  if is_design:
    pretest_data = data
  else:
    if pretest_end_date is not None:
      pretest_data = data[data[api.DATE] <= pretest_end_date]
    else:
      pretest_data = data

  if not pretest_data.empty:
    # Percentage of missing conversion days
    min_date = pretest_data[api.DATE].min()
    max_date = pretest_data[api.DATE].max()
    expected_days = (max_date - min_date).days + 1
    actual_days = (
        pretest_data.groupby(api.DATE)[api.CONVERSIONS].sum().gt(0).sum()
    )
    missing_days_pct = (
        1.0 - (actual_days / expected_days) if expected_days > 0 else 0.0
    )
    missing_days_pct = round(missing_days_pct, 10)

    if missing_days_pct > _MAX_MISSING_DAYS_PCT:
      msg = (
          'Missing conversion days percentage exceeds 30% during the pretest'
          ' period.'
      )
      logging.warning(msg)
      metrics.append({
          'metric': 'Percentage of missing conversion days',
          'value': missing_days_pct,
          'message': msg,
          'threshold': _MAX_MISSING_DAYS_PCT,
      })

    # Percentage of missing spend days - only if spend is present.
    # Applied only for the pretest period when experiment_type is Go_dark or
    # Heavy_Up.
    spend_cols = [
        col
        for col in data.columns
        if col == api.SPEND or re.match(api.MULTICELL_SPEND_REGEX, col)
    ]
    if spend_cols and experiment_types is not None:
      for cell_name, experiment_type in experiment_types.items():
        if experiment_type in (
            api.ExperimentType.GO_DARK,
            api.ExperimentType.HEAVY_UP,
        ):
          if cell_name == api.CELL_1 and api.SPEND in data.columns:
            col = api.SPEND
          else:
            col = f'spend_{cell_name}'

          if col in pretest_data.columns:
            spend_by_date = pretest_data.groupby(api.DATE)[col].sum()
            spend_actual_days = spend_by_date.gt(0).sum()
            spend_missing_days_pct = 1.0 - (
                spend_actual_days / expected_days if expected_days > 0 else 0.0
            )
            spend_missing_days_pct = round(spend_missing_days_pct, 10)

            if spend_missing_days_pct > _MAX_MISSING_DAYS_PCT:
              msg = (
                  f'Missing spend days percentage for {cell_name} exceeds 30%'
                  ' during the pretest period.'
              )
              logging.warning(msg)
              metrics.append({
                  'metric': f'Percentage of missing spend days ({cell_name})',
                  'value': spend_missing_days_pct,
                  'message': msg,
                  'threshold': _MAX_MISSING_DAYS_PCT,
              })

  # Duplicate Entries check (on all data)
  if not data.empty:
    duplicates = data.duplicated(subset=[api.DATE, api.LOCATION]).sum()
    if duplicates > _MAX_DUPLICATE_ENTRIES:
      msg = (
          'Entries have duplicate dates per geo exceeding '
          f'{_MAX_DUPLICATE_ENTRIES}. Identified duplicated entries will be'
          ' aggregated.'
      )
      logging.warning(msg)
      metrics.append({
          'metric': 'Duplicate Entries',
          'value': duplicates,
          'message': msg,
          'threshold': _MAX_DUPLICATE_ENTRIES,
      })

  # Spend > 0 and No Conversions - design phase only
  if is_design:
    spend_cols = [
        col
        for col in data.columns
        if col == api.SPEND or re.match(api.MULTICELL_SPEND_REGEX, col)
    ]
    if spend_cols:
      geo_data = data.groupby(api.LOCATION)[
          spend_cols + [api.CONVERSIONS]
      ].sum()
      total_spend_per_location = geo_data[spend_cols].sum(axis=1)
      zero_conversion_geos = geo_data[
          (total_spend_per_location > 0) & (geo_data[api.CONVERSIONS] <= 0)
      ].index.tolist()
      if zero_conversion_geos:
        if quality_check_config.exclude_geos_no_response:
          outlier_geos.update(zero_conversion_geos)
          message = (
              'Found geos with spend > 0 and no response. Moved to outlier'
              ' geos.'
          )
        else:
          message = 'Found geos with spend > 0 and no response.'
        logging.warning(message)
        metrics.append({
            'metric': 'Spend > 0 and no conversions',
            'value': len(zero_conversion_geos),
            'message': message,
            'threshold': None,
        })

  # Percentage of high zero response during the pretest period only.
  if not pretest_data.empty:
    pivoted_conversions = pretest_data.pivot_table(
        index=api.DATE,
        columns=api.LOCATION,
        values=api.CONVERSIONS,
        aggfunc='sum',
    ).fillna(0)
    zero_response_pct = (pivoted_conversions == 0).values.mean()
  else:
    zero_response_pct = 0.0

  if zero_response_pct > _MAX_ZERO_RESPONSE_PCT:
    msg = 'Zero conversion percentage exceeds 50% during the pretest period.'
    logging.warning(msg)
    metrics.append({
        'metric': 'Percentage of zero response',
        'value': zero_response_pct,
        'message': msg,
        'threshold': _MAX_ZERO_RESPONSE_PCT,
    })

  quality_metrics = pd.DataFrame(
      metrics, columns=['metric', 'value', 'message', 'threshold']
  )
  if quality_metrics.empty:
    quality_metrics = pd.DataFrame(
        columns=['metric', 'value', 'message', 'threshold']
    )

  return api.QualityCheckResult(
      quality_metrics=quality_metrics,
      outlier_geos=outlier_geos,
      outlier_dates=outlier_dates,
  )


def check_design_data_quality(
    data: pd.DataFrame,
    design_config: api.DesignConfig,
    quality_check_config: api.QualityCheckConfig,
) -> api.QualityCheckResult:
  """Checks the quality of the input data for the design phase."""
  return _check_data_quality(
      data=data,
      quality_check_config=quality_check_config,
      is_design=True,
      experiment_types=design_config.experiment_types,
  )


def check_analysis_data_quality(
    data: pd.DataFrame,
    analysis_config: api.AnalysisConfig,
    quality_check_config: api.QualityCheckConfig,
) -> api.QualityCheckResult:
  """Checks the quality of the input data for the analysis/reporting phase."""
  if analysis_config.design.design_config is None:
    raise ValueError('Design config is required in analysis_config.')

  pretest_end_date = analysis_config.pretest_end_date
  if pretest_end_date is None:
    pretest_end_date = analysis_config.analysis_start_date - pd.Timedelta(
        days=1
    )

  return _check_data_quality(
      data=data,
      quality_check_config=quality_check_config,
      is_design=False,
      experiment_types=analysis_config.design.design_config.experiment_types,
      pretest_end_date=pretest_end_date,
  )
