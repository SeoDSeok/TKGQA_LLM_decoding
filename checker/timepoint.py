"""Granularity-aware time points for TKGQA temporal reasoning.

MultiTQ (and ICEWS in general) uses ``YYYY-MM-DD`` timestamps, but questions
carry three granularities: ``day`` / ``month`` / ``year``.  A "before 2016"
constraint must treat 2016 as the whole year, while a "on 2016-03-04" constraint
is a single day.  ``TimePoint`` captures both the parsed date and the intended
granularity so that comparisons stay correct across granularities.

Key idea: a granular time point denotes an *interval* ``[lower, upper]`` on the
day axis (year 2016 -> [2016-01-01, 2016-12-31]).  ``before``/``after`` compare
against that interval's boundaries, while ``equal`` compares only the fields up
to the coarser of the two granularities.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import total_ordering
import re

GRANULARITIES = ("year", "month", "day")
_GRAN_RANK = {g: i for i, g in enumerate(GRANULARITIES)}

_DATE_RE = re.compile(r"^\s*(\d{4})(?:-(\d{1,2}))?(?:-(\d{1,2}))?\s*$")


def _days_in_month(year: int, month: int) -> int:
    if month == 2:
        leap = (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0)
        return 29 if leap else 28
    return 31 if month in (1, 3, 5, 7, 8, 10, 12) else 30


@total_ordering
@dataclass(frozen=True)
class TimePoint:
    """A date with an explicit granularity.

    Ordering (``<``, ``==``) is defined on the canonical *lower* day ordinal, so
    that ``TimePoint`` instances can be put in sorted lists.  For granularity-
    aware semantics use :meth:`lower`, :meth:`upper`, and :meth:`equal_at`.
    """

    year: int
    month: int = 1
    day: int = 1
    granularity: str = "day"

    # ----- construction ---------------------------------------------------
    @classmethod
    def parse(cls, s: str, granularity: str | None = None) -> "TimePoint":
        """Parse ``YYYY``, ``YYYY-MM`` or ``YYYY-MM-DD``.

        If ``granularity`` is not given it is inferred from how many fields are
        present in the string.
        """
        m = _DATE_RE.match(str(s))
        if not m:
            raise ValueError(f"unparseable timestamp: {s!r}")
        y = int(m.group(1))
        mo = m.group(2)
        d = m.group(3)
        if granularity is None:
            if d is not None:
                granularity = "day"
            elif mo is not None:
                granularity = "month"
            else:
                granularity = "year"
        month = int(mo) if mo is not None else 1
        day = int(d) if d is not None else 1
        return cls(y, month, day, granularity)

    # ----- interval semantics --------------------------------------------
    def lower(self) -> tuple[int, int, int]:
        """Inclusive lower bound of the interval this point denotes (y, m, d)."""
        if self.granularity == "year":
            return (self.year, 1, 1)
        if self.granularity == "month":
            return (self.year, self.month, 1)
        return (self.year, self.month, self.day)

    def upper(self) -> tuple[int, int, int]:
        """Inclusive upper bound of the interval this point denotes (y, m, d)."""
        if self.granularity == "year":
            return (self.year, 12, 31)
        if self.granularity == "month":
            return (self.year, self.month, _days_in_month(self.year, self.month))
        return (self.year, self.month, self.day)

    def _lower_ord(self) -> int:
        y, m, d = self.lower()
        return (y * 13 + m) * 32 + d

    def _upper_ord(self) -> int:
        y, m, d = self.upper()
        return (y * 13 + m) * 32 + d

    # ----- comparisons ----------------------------------------------------
    def __eq__(self, other) -> bool:
        if not isinstance(other, TimePoint):
            return NotImplemented
        return self._lower_ord() == other._lower_ord()

    def __lt__(self, other) -> bool:
        if not isinstance(other, TimePoint):
            return NotImplemented
        return self._lower_ord() < other._lower_ord()

    def __hash__(self) -> int:
        return hash(self._lower_ord())

    def strictly_before(self, anchor: "TimePoint") -> bool:
        """True iff this point lies entirely before ``anchor``'s interval."""
        return self._upper_ord() < anchor._lower_ord()

    def strictly_after(self, anchor: "TimePoint") -> bool:
        """True iff this point lies entirely after ``anchor``'s interval."""
        return self._lower_ord() > anchor._upper_ord()

    def equal_at(self, anchor: "TimePoint", granularity: str | None = None) -> bool:
        """Granularity-aware equality.

        Compares fields up to the *coarser* of the two granularities (or an
        explicit ``granularity`` override).  "2005-04-14" ``equal_at`` "2005-04"
        (month) is True; at day granularity it would be False.
        """
        if granularity is None:
            granularity = self.granularity if _GRAN_RANK[self.granularity] < _GRAN_RANK[anchor.granularity] else anchor.granularity
        rank = _GRAN_RANK[granularity]
        if self.year != anchor.year:
            return False
        if rank >= _GRAN_RANK["month"] and self.month != anchor.month:
            return False
        if rank >= _GRAN_RANK["day"] and self.day != anchor.day:
            return False
        return True

    def __str__(self) -> str:
        if self.granularity == "year":
            return f"{self.year:04d}"
        if self.granularity == "month":
            return f"{self.year:04d}-{self.month:02d}"
        return f"{self.year:04d}-{self.month:02d}-{self.day:02d}"


def parse_ts(s: str, granularity: str | None = None) -> TimePoint:
    return TimePoint.parse(s, granularity)
