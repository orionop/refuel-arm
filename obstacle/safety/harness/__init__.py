"""Benchmark harness — scenarios, metrics, runner."""
from .metrics import RunMetrics, compute_metrics
from .runner import RunLog, print_metrics_table, run_benchmark, run_scenario
from .scenarios import (
    Scenario,
    adversarial_head_on,
    all_scenarios,
    fast_dash,
    head_on,
    oblique,
    passing,
    static_in_path,
    vertical_drop,
)

__all__ = [
    "Scenario",
    "all_scenarios", "fast_dash", "head_on", "oblique", "passing",
    "static_in_path", "vertical_drop", "adversarial_head_on",
    "RunMetrics", "compute_metrics",
    "RunLog", "run_scenario", "run_benchmark", "print_metrics_table",
]
