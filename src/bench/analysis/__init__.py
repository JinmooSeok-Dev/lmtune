from bench.analysis.aggregate import aggregate, requests_to_dataframe, session_totals_from_requests
from bench.analysis.compare import RunComparison, compare_runs
from bench.analysis.derived import DerivedSpec, compute_derived, resolve_builtin, safe_eval
from bench.analysis.distributions import ecdf, fit_zipf_s, histogram, variance_stats
from bench.analysis.metrics import percentiles, summarize_requests
from bench.analysis.nway import NWayTable, build_nway_table, nway_to_markdown, variance_across_runs
from bench.analysis.registry import MetricDef, direction_of, list_all, register

__all__ = [
    "DerivedSpec",
    "MetricDef",
    "NWayTable",
    "RunComparison",
    "aggregate",
    "build_nway_table",
    "compare_runs",
    "compute_derived",
    "direction_of",
    "ecdf",
    "fit_zipf_s",
    "histogram",
    "list_all",
    "nway_to_markdown",
    "percentiles",
    "register",
    "requests_to_dataframe",
    "resolve_builtin",
    "safe_eval",
    "session_totals_from_requests",
    "summarize_requests",
    "variance_across_runs",
    "variance_stats",
]
