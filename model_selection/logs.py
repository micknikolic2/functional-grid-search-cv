"""
Execution trace aggregation utilities.

This module provides helper functions for aggregating and filtering
Writer-based execution traces produced during functional GridSearchCV.

The utilities are pure and side-effect free: they operate only on the
structured execution_trace_ data produced during grid search and return
derived views suitable for inspection, testing, or reporting.
"""

# Import libraries
from typing import List, Dict, Any, Callable
from collections import defaultdict

# Containers for execution traces
LogEntry = Dict[str, Any]
ExecutionTrace = List[List[LogEntry]]

def flatten_logs(trace: ExecutionTrace) -> List[LogEntry]:
    """
    Flatten execution trace into a single list of log entries.
    """
    if trace is None:
        return []
    return [entry for logs in trace for entry in logs]


def logs_by_candidate(trace: ExecutionTrace) -> Dict[int, List[LogEntry]]:
    """
    Group logs by candidate_index.
    """
    grouped = defaultdict(list)
    for entry in flatten_logs(trace):
        cid = entry.get("candidate_index")
        if cid is not None:
            grouped[cid].append(entry)
    return dict(grouped)


def logs_by_fold(trace: ExecutionTrace) -> Dict[int, List[LogEntry]]:
    """
    Group logs by fold index.
    """
    grouped = defaultdict(list)
    for entry in flatten_logs(trace):
        fid = entry.get("split_index")
        if fid is not None:
            grouped[fid].append(entry)
    return dict(grouped)


def logs_by_candidate_and_fold(
    trace: ExecutionTrace,
) -> Dict[int, Dict[int, List[LogEntry]]]:
    """
    Group logs by candidate_index and split_index.
    """
    grouped = defaultdict(lambda: defaultdict(list))
    for entry in flatten_logs(trace):
        cid = entry.get("candidate_index")
        fid = entry.get("split_index")
        if cid is not None and fid is not None:
            grouped[cid][fid].append(entry)
    return {cid: dict(folds) for cid, folds in grouped.items()}


def filter_logs(
    trace: ExecutionTrace,
    predicate: Callable[[LogEntry], bool],
) -> List[LogEntry]:
    """
    Filter log entries using a predicate function.
    """
    return [e for e in flatten_logs(trace) if predicate(e)]


def events_of_type(trace: ExecutionTrace, event: str) -> List[LogEntry]:
    """
    Return all log entries matching a given event name.
    """
    return filter_logs(trace, lambda e: e.get("event") == event)


def slow_fold_events(trace: ExecutionTrace, factor: float = 2.0, min_seconds: float = 0.0) -> List[LogEntry]:
    """
    Identify folds/candidates whose fit_time is unusually long.

    The rule:
        fit_time > factor * median_fit_time and fit_time >= min_seconds

    A pure post-hoc detection. The raw fit_time must be logged as data.
    """
    entries = flatten_logs(trace)
    fit_times = [e.get("fit_time") for e in entries if isinstance(e.get("fit_time"), (int, float))]
    if not fit_times:
        return []

    med = float(np.median(fit_times))
    threshold = factor * med

    return [
        e for e in entries
        if isinstance(e.get("fit_time"), (int, float))
        and e["fit_time"] >= max(min_seconds, threshold)
    ]