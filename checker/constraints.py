"""Temporal operator taxonomy and MultiTQ question-type mapping.

NOTE (plan v2): this taxonomy is used ONLY to *label* questions for the TVR
metric and the per-operator breakdown (H2).  The decoding method does not
enumerate operators — temporal constraints are learned by the GNN
discriminator.  The DFA enumerating these operators is kept only as an
ablation upper bound.

The metric defines the operator alphabet
``{equal, before, after, first, last, during, multi}``.  MultiTQ collapses
several of these into 6 ``qtype`` labels:

    equal          -> single-fact equality (op=equal)
    equal_multi    -> equality with several anchors / multi answers (op=equal, multi)
    before_after   -> a *before* OR *after* interval filter (resolved by surface form)
    first_last     -> a *first* OR *last* ordering selector (resolved by surface form)
    after_first    -> after(anchor) interval filter + first ordering  (composite)
    before_last    -> before(anchor) interval filter + last ordering  (composite)

Because MultiTQ folds before/after into one label and first/last into another,
we cannot read the concrete operator off ``qtype`` alone.  ``resolve_operator``
inspects the question surface form to disambiguate ("before"/"after",
"first"/"last").  This "one qtype -> family of operators, resolved per question"
step is exactly the temporal-operator labelling the plan asks for
(Phase 0.2 / 0.4) and is where granularity + implicit-anchor handling attaches.

A :class:`TemporalConstraint` is a *conjunction* of atomic predicates:

  * an optional **interval** predicate (before / after / equal / during) applied
    to *every* timestamp on a candidate path, and
  * an optional **ordering** selector (first / last) applied to the *candidate
    fact set* (the answer fact must have the min/max timestamp).

``op`` names the primary operator for reporting; ``ordering`` and ``interval``
carry the machine-checkable pieces.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from .timepoint import TimePoint

# Operator alphabet from the experiment plan (Phase 0.2).
OPERATORS = ("equal", "before", "after", "first", "last", "during", "multi")

# Complexity ordering used for hypothesis H2 (violation rate rises with
# operator complexity: simple < before/after < first/last < multi-constraint).
COMPLEXITY = {
    "equal": 0,
    "before": 1,
    "after": 1,
    "during": 1,
    "first": 2,
    "last": 2,
    "multi": 3,
}


@dataclass
class TemporalConstraint:
    """Machine-checkable temporal constraint extracted from a question.

    Attributes
    ----------
    op:
        Primary operator label, one of :data:`OPERATORS`.
    anchor:
        Reference time point for before/after/equal (may be resolved from an
        implicit event, see ``anchor_is_implicit``).
    interval:
        (lo, hi) inclusive bounds for ``during``.
    ordering:
        ``"first"`` / ``"last"`` selector applied on the candidate set, or None.
    interval_op:
        For composite types, the interval predicate applied *before* the
        ordering selector: one of ``before``/``after``/``equal``/``during``.
    granularity:
        day / month / year of the question.
    answer_type:
        ``entity`` or ``time`` (MultiTQ ``answer_type``).
    anchor_is_implicit:
        True when the anchor is an event/entity whose timestamp must be
        resolved from the KG (TIQ-style implicit time, or MultiTQ "before <X>").
    sub:
        Sub-constraints for ``op == "multi"`` (recursive).
    """

    op: str
    anchor: Optional[TimePoint] = None
    interval: Optional[tuple[TimePoint, TimePoint]] = None
    ordering: Optional[str] = None
    interval_op: Optional[str] = None
    granularity: str = "day"
    answer_type: str = "entity"
    anchor_is_implicit: bool = False
    sub: list["TemporalConstraint"] = field(default_factory=list)
    qtype: Optional[str] = None  # original MultiTQ label, for reporting

    def __post_init__(self):
        if self.op not in OPERATORS:
            raise ValueError(f"unknown operator {self.op!r}; must be one of {OPERATORS}")


# --- surface-form resolution -------------------------------------------------

_BEFORE_KW = (" before ", "prior to", "earlier than", "ahead of")
_AFTER_KW = (" after ", "following ", "later than", "subsequent to")
_FIRST_KW = ("first", "earliest", "initial")
_LAST_KW = ("last", "latest", "most recent", "final")


def _has(text: str, kws) -> bool:
    t = f" {text.lower()} "
    return any(k in t for k in kws)


def resolve_before_after(question: str) -> str:
    """Return 'before' or 'after' for a before_after / composite question."""
    b, a = _has(question, _BEFORE_KW), _has(question, _AFTER_KW)
    if b and not a:
        return "before"
    if a and not b:
        return "after"
    # Ambiguous / both present -> fall back on whichever appears first.
    lo = question.lower()
    ib = min((lo.find(k.strip()) for k in _BEFORE_KW if k.strip() in lo), default=1 << 30)
    ia = min((lo.find(k.strip()) for k in _AFTER_KW if k.strip() in lo), default=1 << 30)
    return "before" if ib <= ia else "after"


def resolve_first_last(question: str) -> str:
    """Return 'first' or 'last' for a first_last / composite question."""
    f, l = _has(question, _FIRST_KW), _has(question, _LAST_KW)
    if f and not l:
        return "first"
    if l and not f:
        return "last"
    lo = question.lower()
    i_f = min((lo.find(k) for k in _FIRST_KW if k in lo), default=1 << 30)
    i_l = min((lo.find(k) for k in _LAST_KW if k in lo), default=1 << 30)
    return "first" if i_f <= i_l else "last"


# --- MultiTQ qtype -> operator family ---------------------------------------
#
# Each entry declares the atomic pieces; the concrete operator is resolved from
# the question surface form at build time.  This is the "매핑 테이블" deliverable.
MULTITQ_QTYPE_MAP = {
    "equal": {
        "interval_op": "equal",
        "ordering": None,
        "primary": "equal",
        "note": "single time anchor equality (granularity-aware)",
    },
    "equal_multi": {
        "interval_op": "equal",
        "ordering": None,
        "primary": "multi",
        "note": "equality against multiple anchors / multi-answer; checked as multi(equal)",
    },
    "before_after": {
        "interval_op": "RESOLVE_BA",  # before | after from surface form
        "ordering": None,
        "primary": "RESOLVE_BA",
        "note": "interval filter only; before/after disambiguated by question text",
    },
    "first_last": {
        "interval_op": None,
        "ordering": "RESOLVE_FL",  # first | last from surface form
        "primary": "RESOLVE_FL",
        "note": "pure ordering selector; needs candidate fact set",
    },
    "after_first": {
        "interval_op": "after",
        "ordering": "first",
        "primary": "first",
        "note": "composite: after(anchor) then first among survivors",
    },
    "before_last": {
        "interval_op": "before",
        "ordering": "last",
        "primary": "last",
        "note": "composite: before(anchor) then last among survivors",
    },
}


def build_constraint(
    qtype: str,
    question: str,
    granularity: str = "day",
    answer_type: str = "entity",
    anchor: Optional[TimePoint] = None,
    interval: Optional[tuple[TimePoint, TimePoint]] = None,
    anchor_is_implicit: bool = False,
) -> TemporalConstraint:
    """Build a :class:`TemporalConstraint` from a MultiTQ question.

    ``anchor`` (for before/after/equal) is supplied by the caller after being
    extracted from the question / gold annotation, because resolving it may need
    the KG (implicit event anchors).  ``qtype`` selects the template; the
    surface form disambiguates before/after and first/last.
    """
    if qtype not in MULTITQ_QTYPE_MAP:
        raise ValueError(f"unknown MultiTQ qtype {qtype!r}")
    tmpl = MULTITQ_QTYPE_MAP[qtype]

    interval_op = tmpl["interval_op"]
    ordering = tmpl["ordering"]
    if interval_op == "RESOLVE_BA":
        interval_op = resolve_before_after(question)
    if ordering == "RESOLVE_FL":
        ordering = resolve_first_last(question)

    primary = tmpl["primary"]
    if primary == "RESOLVE_BA":
        primary = interval_op
    elif primary == "RESOLVE_FL":
        primary = ordering

    common = dict(
        granularity=granularity,
        answer_type=answer_type,
        anchor=anchor,
        interval=interval,
        anchor_is_implicit=anchor_is_implicit,
        qtype=qtype,
    )

    if primary == "multi":
        # equal_multi -> conjunction; concrete sub-anchors filled by caller.
        return TemporalConstraint(op="multi", interval_op="equal", **common)

    return TemporalConstraint(op=primary, interval_op=interval_op, ordering=ordering, **common)
