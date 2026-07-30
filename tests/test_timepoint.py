"""Unit tests for granularity-aware TimePoint."""
from checker.timepoint import TimePoint as TP


def test_parse_granularities():
    assert TP.parse("2016").granularity == "year"
    assert TP.parse("2016-03").granularity == "month"
    assert TP.parse("2016-03-04").granularity == "day"


def test_ordering():
    assert TP.parse("2005-01-01") < TP.parse("2005-01-02")
    assert TP.parse("2005") < TP.parse("2006")
    assert sorted([TP.parse("2007"), TP.parse("2005"), TP.parse("2006")])[0].year == 2005


def test_strictly_before_after_year_boundary():
    # A day in 2015 is strictly before the year 2016.
    assert TP.parse("2015-12-31").strictly_before(TP.parse("2016"))
    # A day inside 2016 is NOT strictly after the year 2016.
    assert not TP.parse("2016-06-01").strictly_after(TP.parse("2016"))
    # 2017 is strictly after year 2016.
    assert TP.parse("2017-01-01").strictly_after(TP.parse("2016"))


def test_equal_at_granularity():
    day = TP.parse("2005-04-14")
    assert day.equal_at(TP.parse("2005-04"))       # month match
    assert day.equal_at(TP.parse("2005"))          # year match
    assert not day.equal_at(TP.parse("2005-04-15"))  # day mismatch
    assert not day.equal_at(TP.parse("2005-05"))     # month mismatch


def test_str_roundtrip():
    assert str(TP.parse("2016")) == "2016"
    assert str(TP.parse("2016-03")) == "2016-03"
    assert str(TP.parse("2016-03-04")) == "2016-03-04"
