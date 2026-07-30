"""TVR checker: evaluation-only temporal-constraint metric (research plan v2).

In plan v2 the method is a *learned* temporal GNN discriminator; rules live
ONLY in this metric, never in the decoding method.  This package is the
rule-based TVR/CVR checker used to quantify temporal hallucination in TKGQA
reasoning paths (Phase 0 motivation + ongoing evaluation): granularity-aware
time points, an operator taxonomy for *labelling* questions, the timestamp-
augmented path format, and the validity checker itself.
"""

from .timepoint import TimePoint, parse_ts, GRANULARITIES
from .constraints import (
    OPERATORS,
    COMPLEXITY,
    TemporalConstraint,
    MULTITQ_QTYPE_MAP,
    build_constraint,
    resolve_before_after,
    resolve_first_last,
)
from .path_format import (
    temporal_path_to_string,
    quad_to_edge_str,
    wrap_path,
    extract_timestamps,
    PATH_START_TOKEN,
    PATH_END_TOKEN,
)
from .violation_checker import (
    CheckResult,
    is_temporally_valid,
    chronological_violation,
    temporal_validity_rate,
    chronological_violation_rate,
)

__all__ = [
    "TimePoint", "parse_ts", "GRANULARITIES",
    "OPERATORS", "COMPLEXITY", "TemporalConstraint", "MULTITQ_QTYPE_MAP",
    "build_constraint", "resolve_before_after", "resolve_first_last",
    "temporal_path_to_string", "quad_to_edge_str", "wrap_path", "extract_timestamps",
    "PATH_START_TOKEN", "PATH_END_TOKEN",
    "CheckResult", "is_temporally_valid", "chronological_violation",
    "temporal_validity_rate", "chronological_violation_rate",
]
