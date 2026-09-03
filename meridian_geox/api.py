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

"""API types and enums for the GeoX library."""

import dataclasses
import datetime
import enum
from typing import Annotated, Any, Optional, TypeVar, Union

import jax.numpy as jnp
import numpy as np
import pandas as pd
import pandera.pandas as pa
import pydantic

# pytype: disable=invalid-annotation

# Column names.
DATE = "date"
LOCATION = "location"
CONVERSIONS = "conversions"
SPEND = "spend"
CELL_1 = "cell_1"
MULTICELL_SPEND_REGEX = r"^spend_cell_[1-9]\d*$"


def _validate_timestamp(v: Any) -> pd.Timestamp:
  if isinstance(v, pd.Timestamp):
    return v
  return pd.Timestamp(v)


def _serialize_timestamp(v: pd.Timestamp) -> str:
  return v.isoformat()


def _serialize_set(v: set[Any]) -> list[Any]:
  return sorted(list(v))


Timestamp = Annotated[
    pd.Timestamp,
    pydantic.BeforeValidator(_validate_timestamp),
    pydantic.PlainSerializer(
        _serialize_timestamp, return_type=str, when_used="json"
    ),
]


T = TypeVar("T")
SortedSet = Annotated[
    set[T],
    pydantic.PlainSerializer(
        _serialize_set, return_type=list, when_used="json"
    ),
]


def _validate_jnp_array(v: Any) -> jnp.ndarray:
  if isinstance(v, (jnp.ndarray, np.ndarray)):
    return jnp.array(v)
  return jnp.array(v)


def _serialize_jnp_array(v: Any) -> list[Any]:
  if hasattr(v, "tolist"):
    return v.tolist()
  return list(v)


JnpArray = Annotated[
    jnp.ndarray,
    pydantic.BeforeValidator(_validate_jnp_array),
    pydantic.PlainSerializer(
        _serialize_jnp_array, return_type=list, when_used="json"
    ),
]


def _validate_duration(v: Any) -> datetime.timedelta:
  """Validates that duration is specified in full days or weeks."""
  if isinstance(v, datetime.timedelta):
    td = v
  elif isinstance(v, str):
    td = pd.to_timedelta(v).to_pytimedelta()
  else:
    raise TypeError("experiment_duration must be a datetime.timedelta object.")

  if td.seconds > 0 or td.microseconds > 0:
    raise ValueError(
        "experiment_duration must be in full days or weeks (no hours, minutes,"
        " seconds, or microseconds allowed)."
    )
  return td


def _validate_dataframe(v: Any) -> pd.DataFrame:
  if isinstance(v, pd.DataFrame):
    df = v.copy()
  else:
    df = pd.DataFrame(**v)
  df.columns = df.columns.astype(str).str.lower()
  if DATE in df.columns:
    df[DATE] = pd.to_datetime(df[DATE])
  return df


def _serialize_dataframe(v: pd.DataFrame) -> dict[str, Any]:
  return v.to_dict(orient="split")


DataFrame = Annotated[
    pd.DataFrame,
    pydantic.BeforeValidator(_validate_dataframe),
    pydantic.PlainSerializer(
        _serialize_dataframe, return_type=dict, when_used="json"
    ),
]


@dataclasses.dataclass
class QualityCheckConfig:
  """Parameters for checking the quality of the input data."""

  # This specific field is for design phase only.
  exclude_geos_no_response: bool = True
  exclude_outlier_dates: bool = True


@dataclasses.dataclass
class QualityCheckResult:
  """Result of the quality check."""

  quality_check_config: QualityCheckConfig
  quality_metrics: DataFrame = dataclasses.field(repr=False)
  outlier_geos: set[str] = dataclasses.field(default_factory=set)
  outlier_dates: set[Timestamp] = dataclasses.field(default_factory=set)


class DataSchema(pa.DataFrameModel):
  """Schema for geo data."""

  # Required: Date.
  date: pa.typing.Series[pd.Timestamp] = pa.Field(alias=DATE, nullable=False)
  # Required: Location Name (String).
  # Must not contain null/empty strings.
  location: pa.typing.Series[str] = pa.Field(
      alias=LOCATION, str_matches=r".+", nullable=False
  )
  # Required: Conversions (Numeric).
  conversions: pa.typing.Series[float] = pa.Field(
      alias=CONVERSIONS, nullable=False
  )
  # Optional: Spend (Numeric).
  # Must not be negative if provided.
  spend: Optional[pa.typing.Series[float]] = pa.Field(
      alias=SPEND, ge=0, nullable=False
  )
  # Optional: Spend per cell (Numeric).
  # Must not be negative if provided.
  spend_by_cell: Optional[pa.typing.Series[float]] = pa.Field(
      alias=r"^spend_cell_[1-9]\d*$", regex=True, ge=0, nullable=False
  )


class ExperimentType(enum.Enum):
  HOLDBACK = 1
  GO_DARK = 2
  HEAVY_UP = 3


class GeoAssignmentRule(enum.Enum):
  RANDOM = 1
  STRATIFIED_SAMPLING = 2


class Methodology(enum.Enum):
  TBR = 1
  SDID = 2


class TestType(enum.Enum):
  ONE_SIDED = 1
  TWO_SIDED = 2


class GeoGroup(enum.Enum):
  CONTROL = 1
  TREATMENT = 2
  EXCLUDED = 3


@pydantic.dataclasses.dataclass
class DesignConfig:
  """Parameters for designing a GeoX study."""

  __pydantic_config__ = pydantic.ConfigDict(arbitrary_types_allowed=True)

  # TODO: Figure out if need to support weekly granularity.
  experiment_duration: Annotated[
      datetime.timedelta,
      pydantic.BeforeValidator(_validate_duration),
      pydantic.Field(gt=datetime.timedelta(0)),
  ]
  # For multi-cell experiments, different types can be assigned per cell using
  # a dictionary; otherwise, a single provided type is applied to all cells.
  experiment_types: Union[
      ExperimentType,
      dict[str, ExperimentType],
  ] = ExperimentType.HOLDBACK
  # The methodologies to be considered for the experiment design.
  methodology: Methodology = Methodology.TBR
  geo_assignment_rule: GeoAssignmentRule = GeoAssignmentRule.STRATIFIED_SAMPLING
  # TODO: If needed, add per-methodology configs.
  cell_count: Annotated[int, pydantic.Field(gt=0)] = 1
  alpha: float = 0.1
  power: float = 0.8
  test_type: TestType = TestType.TWO_SIDED
  # The number of output design options.
  design_output_count: Annotated[int, pydantic.Field(gt=0)] = 10
  # Used to estimate budget requirements for Holdback experiment (cell). For
  # multi-cell experiments, different values can be assigned per Holdback cell
  # using a dictionary. If a single float is provided for a multi-cell
  # design, it will be applied to all Holdback cells.
  cost_per_incremental_conversion: Union[float, dict[str, float]] = 1.0

  @pydantic.model_validator(mode="after")
  def _normalize(self) -> "DesignConfig":
    """Normalizes the config to a standard format."""
    if isinstance(self.experiment_types, ExperimentType):
      self.experiment_types = {
          f"cell_{i}": self.experiment_types
          for i in range(1, self.cell_count + 1)
      }

    if isinstance(self.cost_per_incremental_conversion, float):
      cpic_val = self.cost_per_incremental_conversion
      self.cost_per_incremental_conversion = {
          cell: cpic_val
          for cell, et in self.experiment_types.items()
          if et == ExperimentType.HOLDBACK
      }
    return self

  # Advanced design search parameters.
  # Number of candidates for the fast scoring step.
  n_candidates: Annotated[int, pydantic.Field(gt=0)] = 100_000
  # Number of fully scored candidates.
  n_ranked_candidates: Annotated[int, pydantic.Field(gt=0)] = 100
  max_candidate_generation_retries: Annotated[int, pydantic.Field(gt=0)] = 10
  # Random number generator seed.
  seed: int = 42
  # Maximum allowed symmetric difference for slope check.
  slope_tolerance: float = 0.2
  # Minimum allowed R2 for design candidates.
  min_r2: float = 0.8
  # Number of strata for stratified sampling.
  num_strata: Annotated[int, pydantic.Field(gt=0)] = 4
  # Number of iterations for k-means clustering.
  k_means_iterations: Annotated[int, pydantic.Field(gt=0)] = 10


@pydantic.dataclasses.dataclass
class Budget:
  """Budget constraint for a single cell."""

  budget: Optional[float] = None
  budget_pct: Optional[float] = None

  def __post_init__(self):
    if (self.budget is not None) == (self.budget_pct is not None):
      raise ValueError(
          "Exactly one of 'budget' or 'budget_pct' must be provided."
      )


@pydantic.dataclasses.dataclass
class Constraints:
  """Constraints for designing a GeoX study."""

  __pydantic_config__ = pydantic.ConfigDict(arbitrary_types_allowed=True)

  # The geos to be included in the control group.
  included_control_geos: SortedSet[str] = dataclasses.field(default_factory=set)
  # The geos to be excluded from the experiment design.
  excluded_geos: SortedSet[str] = dataclasses.field(default_factory=set)
  # Dates to exclude from the experiment design.
  excluded_dates: SortedSet[Timestamp] = dataclasses.field(default_factory=set)
  # The budget constraint for the experiment design (per cell). For
  # multi-cell experiments, different values can be assigned per cell using
  # a dictionary; otherwise, a single provided budget is applied to all cells.
  budget_constraint: Union[Budget, dict[str, Budget], None] = None
  # The maximum conversion volume allowed for the treatment group. For
  # multi-cell designs, this percentage refers to the total for all
  # treatment cells.
  max_conversions_percent: Optional[float] = 0.3

  def normalize(self, experiment_types: dict[str, ExperimentType]):
    """Normalizes budget based on experiment types."""
    if isinstance(self.budget_constraint, Budget):
      self.budget_constraint = {
          cell: self.budget_constraint for cell in experiment_types
      }
    elif self.budget_constraint is None:
      self.budget_constraint = {}


@dataclasses.dataclass
class PerCellDesign:
  """Design results for a single cell."""

  __pydantic_config__ = pydantic.ConfigDict(arbitrary_types_allowed=True)

  treatment_geos: SortedSet[str]
  minimum_detectable_effect: float
  design_implied_cpic: float
  p_value: float
  budget: float
  # Counterfactual conversion time series. Includes date, observed, and
  # counterfactual conversions. This is used for plotting.
  counterfactual_conversions: Annotated[
      Optional[DataFrame], pydantic.Field(exclude=True)
  ] = dataclasses.field(default=None, repr=False)


@dataclasses.dataclass
class Design:
  """A design for a GeoX study."""

  __pydantic_config__ = pydantic.ConfigDict(arbitrary_types_allowed=True)

  # Results for each cell.
  designs: dict[str, PerCellDesign]
  control_geos: SortedSet[str]
  # Geos excluded from the design. Includes user manually excluded geos, outlier
  # geos detected by data quality checks (if configured to be removed
  # automatically).
  excluded_geos: SortedSet[str]
  # Dates excluded from the design. Includes user manually excluded dates,
  # and outlier dates detected by data quality checks (if configured to be
  # removed automatically).
  excluded_dates: SortedSet[Timestamp] = dataclasses.field(default_factory=set)
  design_config: Optional[DesignConfig] = None
  constraints: Optional[Constraints] = None
  quality_check_result: Optional[QualityCheckResult] = None
  # The stratum label of each geo, ordered by geo name.
  geo_stratum_labels: Optional[JnpArray] = dataclasses.field(
      default=None, repr=False
  )
  # The data used for the design. This is used for analysis.
  data: Optional[DataFrame] = dataclasses.field(default=None, repr=False)

  def export_to_json(self) -> str:
    """Exports the design to a JSON file."""
    return pydantic.TypeAdapter(Design).dump_json(self).decode()

  @classmethod
  def load_from_json(cls, json_str: str) -> "Design":
    """Loads the design from a JSON file."""
    return pydantic.TypeAdapter(cls).validate_json(json_str)


@dataclasses.dataclass
class DesignSet:
  """A set of designs for a GeoX study."""

  __pydantic_config__ = pydantic.ConfigDict(arbitrary_types_allowed=True)

  designs: dict[str, Design]
  # A dataframe with ranked designs and their corresponding metrics. This
  # includes an index to identify the design in the list of designs and metrics
  # such as cell ID, design rank score, budget, minimum detectable effect,
  # statistical power, and other robustness and representativeness metrics.
  design_metrics: DataFrame = dataclasses.field(repr=False)


@pydantic.dataclasses.dataclass
class AnalysisConfig:
  """Parameters for analyzing a GeoX study."""

  __pydantic_config__ = pydantic.ConfigDict(arbitrary_types_allowed=True)

  # The design (including geo splits, geo assignment rule, other design input
  # parameters) used for the experiment.
  design: Design
  analysis_start_date: Timestamp
  # If needed, extend this to include a cooldown period.
  analysis_end_date: Timestamp
  # The end date of the pretest period. If not provided, the pretest period
  # will be all dates before the analysis_start_date.
  pretest_end_date: Optional[Timestamp] = None
  # The dates to be excluded from the analysis.
  excluded_dates: SortedSet[Timestamp] = dataclasses.field(default_factory=set)
  # If not provided, will be inferred from the design config.
  alpha: Optional[float] = None
  # If not provided, will be inferred from the design config.
  test_type: Optional[TestType] = None

  @pydantic.model_validator(mode="after")
  def validate_dates(self) -> "AnalysisConfig":
    """Validates the pretest and test date ranges."""
    if self.analysis_start_date > self.analysis_end_date:
      raise ValueError(
          "analysis_start_date must be less than or equal to analysis_end_date."
      )
    if (
        self.pretest_end_date is not None
        and self.pretest_end_date >= self.analysis_start_date
    ):
      raise ValueError(
          "pretest_end_date must be strictly before analysis_start_date."
      )
    return self

  # Advanced analysis parameters
  # Number of initial placebo candidates generated before selection.
  n_placebo_candidates: int = 100_000
  # Number of top valid placebo candidates used for analysis.
  n_top_placebos: Annotated[int, pydantic.Field(gt=0)] = 500
  # Minimum out-of-sample R-squared score required for a placebo design to be
  # kept for analysis.
  min_placebo_r2: float = 0.6
  # Number of valid placebo candidates below which a warning will be logged.
  min_placebo_count_warning: Annotated[int, pydantic.Field(ge=0)] = 100
  # Number of valid placebo candidates below which an error will be raised.
  min_placebo_count_error: Annotated[int, pydantic.Field(ge=0)] = 10


@dataclasses.dataclass
class Estimate:
  """An estimate with its confidence interval."""

  __pydantic_config__ = pydantic.ConfigDict(arbitrary_types_allowed=True)

  point_estimate: float
  lower_bound: float
  upper_bound: float
  standard_deviation: float
  p_value: float


@dataclasses.dataclass
class DescriptiveMetrics:
  """Descriptive metrics for a single cell analysis."""

  __pydantic_config__ = pydantic.ConfigDict(arbitrary_types_allowed=True)
  # Represents the estimated BAU spend for the geos included in this cell's
  # analysis (the specific cell's treatment geos plus control geos). It does
  # not include spend from other treatment cells or non-experimental geos,
  # and therefore does not represent the advertiser's total national spend.
  estimated_bau_spend: Optional[float] = None


@dataclasses.dataclass
class AnalysisMetrics:
  """Metrics for a single cell analysis."""

  __pydantic_config__ = pydantic.ConfigDict(arbitrary_types_allowed=True)

  lift: Estimate
  percent_lift: Estimate
  # Cumulative time series of lift estimates.
  cumulative_lift: DataFrame = dataclasses.field(
      default_factory=pd.DataFrame, repr=False
  )
  # Counterfactual conversion time series. Includes date, observed,
  # counterfactual, and confidence intervals (test period only).
  counterfactual_conversions: pd.DataFrame = dataclasses.field(
      default_factory=pd.DataFrame, repr=False
  )
  # Pointwise difference between observed and counterfactual conversions.
  # Includes date, difference, and confidence intervals (test period only).
  pointwise_difference: pd.DataFrame = dataclasses.field(
      default_factory=pd.DataFrame, repr=False
  )
  # Incremental conversion per dollar. Conversion could be revenue or any other
  # KPI. When conversion value is used, iCPD is equivalent to iROAS. This is
  # only populated if spend data is available.
  icpd: Optional[Estimate] = None
  # Cumulative time series of iCPD.
  cumulative_icpd: Optional[DataFrame] = dataclasses.field(
      default=None, repr=False
  )
  descriptive_metrics: Optional[DescriptiveMetrics] = None
  analysis_metrics: Optional[DataFrame] = dataclasses.field(
      default=None, repr=False
  )


@dataclasses.dataclass
class AnalysisResult:
  """Analysis results for a GeoX study."""

  __pydantic_config__ = pydantic.ConfigDict(arbitrary_types_allowed=True)

  # Cell IDs and corresponding analysis metrics.
  results: dict[str, AnalysisMetrics]
  # The configuration used for the analysis.
  analysis_config: AnalysisConfig
  # Geos excluded from the analysis. Includes all geos excluded during the
  # design phase.
  excluded_geos: SortedSet[str] = dataclasses.field(default_factory=set)
  # Dates excluded from the analysis. Includes user manually excluded dates from
  # analysis config, and analysis-stage outlier dates (if configured to be
  # removed automatically).
  excluded_dates: SortedSet[Timestamp] = dataclasses.field(default_factory=set)
  quality_check_result: Optional[QualityCheckResult] = None
