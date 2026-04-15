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

import dataclasses
import pandas as pd


@dataclasses.dataclass
class QualityCheckConfig:
  """Parameters for checking the quality of the input data."""

  # TODO: Add additional basic checks/flags/thresholds.
  check_outlier_geos: bool = True
  check_outlier_dates: bool = True
  check_missing_conversions: bool = True
  check_missing_spend: bool = True


@dataclasses.dataclass
class QualityCheckResult:
  """Result of the quality check."""

  # A dataframe with quality metrics such as missing conversion and spend
  # percentages.
  # TODO: Add more quality metrics.
  quality_metrics: pd.DataFrame = dataclasses.field(repr=False)
  outlier_geos: set[str] = dataclasses.field(default_factory=set)
  outlier_dates: set[pd.Timestamp] = dataclasses.field(default_factory=set)
  # TODO: Add visualization method.


def check_data_quality(
    data: pd.DataFrame, quality_check_config: QualityCheckConfig
) -> QualityCheckResult:
  """Checks the quality of the input data."""
  raise NotImplementedError
