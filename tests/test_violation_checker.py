"""Unit tests for the temporal violation (TVR) checker.

Plan v2 checklist asks for ~10 cases per operator.  Tests are grouped per
operator: equal, before, after, first, last, during, multi, plus chronological
monotonicity (CVR) and end-to-end MultiTQ constraint building.
"""
import pytest

from checker.timepoint import TimePoint as TP
from checker.constraints import TemporalConstraint, build_constraint
from checker.path_format import temporal_path_to_string, wrap_path
from checker.violation_checker import (
    is_temporally_valid,
    chronological_violation,
    temporal_validity_rate,
    chronological_violation_rate,
)


def path(*quads):
    """Helper: build a wrapped path string from (s, r, o, ts) quads."""
    return wrap_path(temporal_path_to_string(quads))


# ----------------------------------------------------------------- equal
@pytest.mark.parametrize("ts,anchor,gran,ok", [
    ("2005-04-14", "2005-04", "month", True),
    ("2005-04-14", "2005", "year", True),
    ("2005-04-14", "2005-04-14", "day", True),
    ("2005-04-14", "2005-04-15", "day", False),
    ("2005-04-14", "2005-05", "month", False),
    ("2005-04-14", "2006", "year", False),
    ("2007-01-01", "2007", "year", True),
    ("2007-12-31", "2007", "year", True),
    ("2007-12-31", "2008", "year", False),
    ("2010-06", "2010-06", "month", True),
])
def test_equal(ts, anchor, gran, ok):
    c = TemporalConstraint(op="equal", interval_op="equal", anchor=TP.parse(anchor), granularity=gran)
    p = path(("X", "visit", "Y", ts))
    assert bool(is_temporally_valid(p, c)) == ok


# ---------------------------------------------------------------- before
@pytest.mark.parametrize("ts,anchor,ok", [
    ("2015-12-31", "2016", True),
    ("2015-01-01", "2016", True),
    ("2016-01-01", "2016", False),
    ("2016-06-01", "2016", False),
    ("2017-01-01", "2016", False),
    ("2015-04-24", "2015-04-25", True),
    ("2015-04-25", "2015-04-25", False),
    ("2014-12-31", "2015-01", True),
    ("2015-01-15", "2015-01", False),
    ("2010", "2011", True),
])
def test_before(ts, anchor, ok):
    c = TemporalConstraint(op="before", interval_op="before", anchor=TP.parse(anchor))
    assert bool(is_temporally_valid(path(("X", "r", "Y", ts)), c)) == ok


# ----------------------------------------------------------------- after
@pytest.mark.parametrize("ts,anchor,ok", [
    ("2017-01-01", "2016", True),
    ("2016-12-31", "2016", False),
    ("2016-06-01", "2016", False),
    ("2015-01-01", "2016", False),
    ("2015-04-26", "2015-04-25", True),
    ("2015-04-25", "2015-04-25", False),
    ("2015-02-01", "2015-01", True),
    ("2015-01-15", "2015-01", False),
    ("2012", "2011", True),
    ("2011-06", "2011", False),
])
def test_after(ts, anchor, ok):
    c = TemporalConstraint(op="after", interval_op="after", anchor=TP.parse(anchor))
    assert bool(is_temporally_valid(path(("X", "r", "Y", ts)), c)) == ok


# ----------------------------------------------------------------- first
def _first_last_case(ordering, answer_ts, candidates):
    c = TemporalConstraint(op=ordering, ordering=ordering)
    p = path(("X", "r", "Ans", answer_ts))
    cand = [TP.parse(t) for t in candidates]
    return is_temporally_valid(p, c, candidate_ts=cand, answer_ts=TP.parse(answer_ts))


@pytest.mark.parametrize("answer,cands,ok", [
    ("2005-01-01", ["2005-01-01", "2006-01-01", "2007-01-01"], True),
    ("2006-01-01", ["2005-01-01", "2006-01-01"], False),
    ("2005-01-01", ["2005-01-01"], True),
    ("2005-03", ["2005-03", "2005-06", "2007"], True),
    ("2005-06", ["2005-03", "2005-06"], False),
    ("2010-01-01", ["2010-01-01", "2010-01-02"], True),
    ("2010-01-02", ["2010-01-01", "2010-01-02"], False),
    ("2008", ["2008", "2009", "2010"], True),
    ("2009", ["2008", "2009", "2010"], False),
    ("2005-01-01", ["2007", "2006", "2005-01-01"], True),
])
def test_first(answer, cands, ok):
    assert bool(_first_last_case("first", answer, cands)) == ok


# ------------------------------------------------------------------ last
@pytest.mark.parametrize("answer,cands,ok", [
    ("2007-01-01", ["2005-01-01", "2006-01-01", "2007-01-01"], True),
    ("2006-01-01", ["2005-01-01", "2007-01-01", "2006-01-01"], False),
    ("2005-01-01", ["2005-01-01"], True),
    ("2007", ["2005-03", "2005-06", "2007"], True),
    ("2005-06", ["2005-03", "2005-06"], True),
    ("2010-01-02", ["2010-01-01", "2010-01-02"], True),
    ("2010-01-01", ["2010-01-01", "2010-01-02"], False),
    ("2010", ["2008", "2009", "2010"], True),
    ("2009", ["2008", "2009", "2010"], False),
    ("2011-12-31", ["2011-01-01", "2011-12-31"], True),
])
def test_last(answer, cands, ok):
    assert bool(_first_last_case("last", answer, cands)) == ok


def test_first_last_missing_candidates_is_invalid():
    c = TemporalConstraint(op="first", ordering="first")
    r = is_temporally_valid(path(("X", "r", "Y", "2005-01-01")), c, candidate_ts=None)
    assert not r.valid and "candidate" in r.reason


# ---------------------------------------------------------------- during
@pytest.mark.parametrize("ts,lo,hi,ok", [
    ("2010-06-01", "2010-01-01", "2010-12-31", True),
    ("2009-12-31", "2010-01-01", "2010-12-31", False),
    ("2011-01-01", "2010-01-01", "2010-12-31", False),
    ("2010-01-01", "2010-01-01", "2010-12-31", True),
    ("2010-12-31", "2010-01-01", "2010-12-31", True),
    ("2010-06", "2010", "2011", True),
    ("2012", "2010", "2011", False),
    ("2010-07-15", "2010-07", "2010-08", True),
    ("2010-09-01", "2010-07", "2010-08", False),
    ("2010-08-31", "2010-07", "2010-08", True),
])
def test_during(ts, lo, hi, ok):
    c = TemporalConstraint(op="during", interval_op="during",
                           interval=(TP.parse(lo), TP.parse(hi)))
    assert bool(is_temporally_valid(path(("X", "r", "Y", ts)), c)) == ok


# ---------------------------------------------- multi-hop interval (all ts)
def test_before_multihop_all_timestamps_checked():
    c = TemporalConstraint(op="before", interval_op="before", anchor=TP.parse("2016"))
    good = path(("A", "r1", "B", "2014-01-01"), ("B", "r2", "C", "2015-06-01"))
    bad = path(("A", "r1", "B", "2014-01-01"), ("B", "r2", "C", "2017-06-01"))
    assert is_temporally_valid(good, c).valid
    assert not is_temporally_valid(bad, c).valid


# ------------------------------------------------------------------ multi
def test_multi_conjunction():
    sub = [
        TemporalConstraint(op="after", interval_op="after", anchor=TP.parse("2010")),
        TemporalConstraint(op="before", interval_op="before", anchor=TP.parse("2015")),
    ]
    c = TemporalConstraint(op="multi", sub=sub)
    assert is_temporally_valid(path(("X", "r", "Y", "2012-01-01")), c).valid
    assert not is_temporally_valid(path(("X", "r", "Y", "2016-01-01")), c).valid
    assert not is_temporally_valid(path(("X", "r", "Y", "2009-01-01")), c).valid


def test_multi_empty_sub_falls_back_to_interval_op():
    c = TemporalConstraint(op="multi", interval_op="equal", anchor=TP.parse("2005"))
    assert is_temporally_valid(path(("X", "r", "Y", "2005-06-01")), c).valid
    assert not is_temporally_valid(path(("X", "r", "Y", "2006-06-01")), c).valid


# ----------------------------------------------- composite after_first etc.
def test_after_first_composite():
    # after(2010) filter, then first among survivors.
    c = TemporalConstraint(op="first", interval_op="after", ordering="first",
                           anchor=TP.parse("2010"))
    cands = [TP.parse(t) for t in ["2009-01-01", "2011-05-01", "2012-01-01"]]
    # answer 2011-05-01 is the earliest AFTER 2010 -> valid.
    r = is_temporally_valid(path(("X", "r", "Ans", "2011-05-01")), c,
                            candidate_ts=cands, answer_ts=TP.parse("2011-05-01"))
    assert r.valid
    # answer 2012 is not the first after 2010.
    r2 = is_temporally_valid(path(("X", "r", "Ans", "2012-01-01")), c,
                             candidate_ts=cands, answer_ts=TP.parse("2012-01-01"))
    assert not r2.valid


def test_before_last_composite():
    c = TemporalConstraint(op="last", interval_op="before", ordering="last",
                           anchor=TP.parse("2015"))
    cands = [TP.parse(t) for t in ["2013-01-01", "2014-06-01", "2016-01-01"]]
    # 2014-06-01 is the latest BEFORE 2015 -> valid.
    r = is_temporally_valid(path(("X", "r", "Ans", "2014-06-01")), c,
                            candidate_ts=cands, answer_ts=TP.parse("2014-06-01"))
    assert r.valid
    # answer 2016 fails the before filter itself.
    r2 = is_temporally_valid(path(("X", "r", "Ans", "2016-01-01")), c,
                             candidate_ts=cands, answer_ts=TP.parse("2016-01-01"))
    assert not r2.valid


# ------------------------------------------- chronological monotonicity (CVR)
@pytest.mark.parametrize("quads,monotone", [
    ([("A", "r", "B", "2014-01-01"), ("B", "r", "C", "2015-01-01")], True),
    ([("A", "r", "B", "2015-01-01"), ("B", "r", "C", "2014-01-01")], False),
    ([("A", "r", "B", "2014-01-01"), ("B", "r", "C", "2014-01-01")], True),
    ([("A", "r", "B", "2014")], True),
])
def test_chronological(quads, monotone):
    assert chronological_violation(path(*quads)).valid == monotone


def test_cvr_aggregation():
    paths = [
        path(("A", "r", "B", "2014-01-01"), ("B", "r", "C", "2015-01-01")),  # ok
        path(("A", "r", "B", "2015-01-01"), ("B", "r", "C", "2014-01-01")),  # reversal
    ]
    assert chronological_violation_rate(paths) == pytest.approx(0.5)


def test_tvr_aggregation():
    c = TemporalConstraint(op="before", interval_op="before", anchor=TP.parse("2016"))
    results = [
        is_temporally_valid(path(("X", "r", "Y", "2015-01-01")), c),
        is_temporally_valid(path(("X", "r", "Y", "2017-01-01")), c),
        is_temporally_valid(path(("X", "r", "Y", "2014-01-01")), c),
    ]
    assert temporal_validity_rate(results) == pytest.approx(2 / 3)


# ----------------------------------------- MultiTQ qtype -> constraint mapping
def test_build_constraint_before_after_resolution():
    c = build_constraint("before_after", "Before 2016, who attacked Iraq?",
                         granularity="day", anchor=TP.parse("2016"))
    assert c.op == "before" and c.interval_op == "before"
    c2 = build_constraint("before_after", "After Ethiopia, who did he negotiate with?",
                          granularity="day", anchor=TP.parse("2013"))
    assert c2.op == "after"


def test_build_constraint_first_last_resolution():
    c = build_constraint("first_last", "When did X last make a request?", granularity="year")
    assert c.op == "last" and c.ordering == "last"
    c2 = build_constraint("first_last", "Who was the first country X praised?", granularity="day")
    assert c2.op == "first" and c2.ordering == "first"


def test_build_constraint_composite():
    c = build_constraint("after_first", "After E, who was the first to visit Iraq?",
                         anchor=TP.parse("2010"))
    assert c.op == "first" and c.interval_op == "after" and c.ordering == "first"
    c2 = build_constraint("before_last", "Before Cambodia, who did X last meet?",
                          anchor=TP.parse("2012"))
    assert c2.op == "last" and c2.interval_op == "before" and c2.ordering == "last"


def test_build_constraint_equal_multi_is_multi():
    c = build_constraint("equal_multi", "Who visited China in the same month as Y?",
                         granularity="month")
    assert c.op == "multi"
