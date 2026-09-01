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

"""meridian_geox API."""

# A new PyPI release will be pushed every time `__version__` is increased.
# When changing this, also update the CHANGELOG.md.
__version__ = '1.0.0'

from .analysis import analyze
from .analysis import plot_analysis
from .api import AnalysisConfig
from .api import AnalysisMetrics
from .api import AnalysisResult
from .api import Budget
from .api import Constraints
from .api import DataSchema
from .api import Design
from .api import DesignConfig
from .api import DesignSet
from .api import Estimate
from .api import ExperimentType
from .api import GeoAssignmentRule
from .api import GeoGroup
from .api import Methodology
from .api import QualityCheckConfig
from .api import QualityCheckResult
from .api import TestType
from .data_quality.data_quality import check_analysis_data_quality
from .data_quality.data_quality import check_design_data_quality
from .design import compare_designs
from .design import concat_design_reports
from .design import plot_design
from .design import run_design
