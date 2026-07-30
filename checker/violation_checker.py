"""Rule-based temporal validity checker (Phase 0.4 of the experiment plan).

Given a generated reasoning path and the temporal constraint of a question,
decide whether the path *satisfies* the constraint.  This is the measurement
that produces TVR (Temporal Validity Rate) and CVR (Chronological Violation
Rate), the paper's motivation numbers.

Design notes
------------
* Interval predicates (before / after / equal / during) are checked against
  **every** timestamp on the path.
* Ordering predicates (first / last) cannot be judged from a path alone; they
  need the *candidate fact set* from the KG.  The caller passes ``candidate_ts``
  (all timestamps of facts matching the question's relation/entity signature),
  and the checker verifies the path's answer timestamp is the min (first) / max
  (last) of that set — optionally after applying the interval filter first
  (composite after_first / before_last).
* ``multi`` recurses over sub-constraints (conjunction).
* Chronological monotonicity across hops (CVR) is checked separately so a path
  can be interval-valid yet chronologically invalid.

Every check returns a :class:`CheckResult` carrying a boolean plus a reason,
so the report can bucket *why* paths fail.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

from .constraints import TemporalConstraint
from .path_format import extract_timestamps
from .timepoint import TimePoint


@dataclass
class CheckResult:
    valid: bool
    reason: str = ""

    def __bool__(self) -> bool:
        return self.valid


def _as_timepoints(path) -> list[TimePoint]:
    """Accept either a rendered path string or a list of TimePoints/strings."""
    if isinstance(path, str):
        return extract_timestamps(path)
    out = []
    for t in path:
        out.append(t if isinstance(t, TimePoint) else TimePoint.parse(t))
    return out


def _interval_ok(
    timestamps: Sequence[TimePoint],
    op: str,
    anchor: Optional[TimePoint],
    interval: Optional[tuple[TimePoint, TimePoint]],
) -> CheckResult:
    if op == "before":
        if anchor is None:
            return CheckResult(False, "before: missing anchor")
        bad = [t for t in timestamps if not t.strictly_before(anchor)]
        return CheckResult(not bad, "" if not bad else f"before({anchor}) violated by {[str(t) for t in bad]}")
    if op == "after":
        if anchor is None:
            return CheckResult(False, "after: missing anchor")
        bad = [t for t in timestamps if not t.strictly_after(anchor)]
        return CheckResult(not bad, "" if not bad else f"after({anchor}) violated by {[str(t) for t in bad]}")
    if op == "equal":
        if anchor is None:
            return CheckResult(False, "equal: missing anchor")
        bad = [t for t in timestamps if not t.equal_at(anchor)]
        return CheckResult(not bad, "" if not bad else f"equal({anchor}) violated by {[str(t) for t in bad]}")
    if op == "during":
        if interval is None:
            return CheckResult(False, "during: missing interval")
        lo, hi = interval
        bad = [t for t in timestamps if t.strictly_before(lo) or t.strictly_after(hi)]
        return CheckResult(not bad, "" if not bad else f"during([{lo},{hi}]) violated by {[str(t) for t in bad]}")
    return CheckResult(True, "")  # no interval predicate


def _ordering_ok(
    answer_ts: Optional[TimePoint],
    ordering: str,
    candidate_ts: Optional[Sequence[TimePoint]],
) -> CheckResult:
    if answer_ts is None:
        return CheckResult(False, f"{ordering}: answer timestamp unknown")
    if not candidate_ts:
        return CheckResult(False, f"{ordering}: candidate fact set unavailable (cannot judge)")
    cand = list(candidate_ts)
    target = min(cand) if ordering == "first" else max(cand)
    ok = answer_ts == target
    return CheckResult(ok, "" if ok else f"{ordering}: answer {answer_ts} != {target} (of {len(cand)} candidates)")


def is_temporally_valid(
    path,
    constraint: TemporalConstraint,
    candidate_ts: Optional[Sequence] = None,
    answer_ts=None,
) -> CheckResult:
    """Return whether ``path`` satisfies ``constraint``.

    Parameters
    ----------
    path:
        Rendered path string (``<PATH> ... </PATH>``) or a list of TimePoints.
    constraint:
        The question's :class:`TemporalConstraint`.
    candidate_ts:
        Timestamps of all KG facts matching the question signature; required for
        first/last ordering checks.
    answer_ts:
        Timestamp of the answer fact on this path; required for first/last.  If
        omitted it defaults to the last timestamp on the path.
    """
    timestamps = _as_timepoints(path)

    # multi: conjunction of sub-constraints (recurse).
    if constraint.op == "multi":
        if not constraint.sub:
            # equal_multi with no expanded sub-anchors: fall back to interval_op
            res = _interval_ok(timestamps, constraint.interval_op or "equal", constraint.anchor, constraint.interval)
            return res
        for sub in constraint.sub:
            r = is_temporally_valid(path, sub, candidate_ts, answer_ts)
            if not r.valid:
                return CheckResult(False, f"multi: {r.reason}")
        return CheckResult(True, "")

    # Resolve candidate timestamps as TimePoints once.
    cand_tp = _as_timepoints(candidate_ts) if candidate_ts is not None else None
    ans_tp = None
    if answer_ts is not None:
        ans_tp = answer_ts if isinstance(answer_ts, TimePoint) else TimePoint.parse(answer_ts)
    elif timestamps:
        ans_tp = timestamps[-1]

    # 1) interval predicate (before/after/equal/during) on all path timestamps.
    interval_op = constraint.interval_op
    if interval_op in ("before", "after", "equal", "during"):
        r = _interval_ok(timestamps, interval_op, constraint.anchor, constraint.interval)
        if not r.valid:
            return r
        # For composite (after_first/before_last), the ordering candidate set is
        # restricted to facts that also pass the interval filter.
        if cand_tp is not None and interval_op in ("before", "after"):
            cand_tp = [t for t in cand_tp if _interval_ok([t], interval_op, constraint.anchor, None).valid]

    # 2) ordering predicate (first/last) on the candidate set.
    if constraint.ordering in ("first", "last"):
        return _ordering_ok(ans_tp, constraint.ordering, cand_tp)

    return CheckResult(True, "")


def chronological_violation(path) -> CheckResult:
    """CVR helper: True (violation) if hop timestamps are non-monotone.

    Returns ``valid=True`` when the path is chronologically consistent
    (timestamps non-decreasing along hops), ``valid=False`` on a time-reversal.
    """
    ts = _as_timepoints(path)
    for a, b in zip(ts, ts[1:]):
        if b < a:
            return CheckResult(False, f"chronological reversal: {a} -> {b}")
    return CheckResult(True, "")


# --- aggregation -------------------------------------------------------------

def temporal_validity_rate(results: Sequence[CheckResult]) -> float:
    """TVR = fraction of paths satisfying the temporal constraint."""
    if not results:
        return float("nan")
    return sum(1 for r in results if r.valid) / len(results)


def chronological_violation_rate(paths: Sequence) -> float:
    """CVR = fraction of multi-hop paths with a time reversal."""
    multi = [p for p in paths if len(_as_timepoints(p)) >= 2]
    if not multi:
        return float("nan")
    return sum(1 for p in multi if not chronological_violation(p).valid) / len(multi)
