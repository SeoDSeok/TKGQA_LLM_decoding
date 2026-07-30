"""Timestamp-augmented reasoning-path format (temporal extension of GCR).

GCR renders a path of triples ``(h, r, t)`` as::

    Obama -> visit -> Cape_Verde

(see ``gcr/src/utils/utils.py::path_to_string``).  For TKGQA we extend the KG
element to a quadruple ``(s, r, o, ts)`` and attach the timestamp to the
relation so the decoder can be constrained on it (Phase 0.2 / Phase 1)::

    <PATH> Obama -> visit [2013-03-28] -> Cape_Verde </PATH>

The bracketed token keeps timestamps lexically separable so a rule checker (and
later the DFA) can recover them regardless of how the tokenizer splits them.
"""

from __future__ import annotations

import re
from typing import Iterable

from .timepoint import TimePoint

PATH_START_TOKEN = "<PATH>"
PATH_END_TOKEN = "</PATH>"

# Timestamp is emitted inside square brackets right after the relation.
_TS_IN_PATH_RE = re.compile(r"\[\s*(\d{4}(?:-\d{1,2}){0,2})\s*\]")


def quad_to_edge_str(s: str, r: str, o: str, ts: str) -> str:
    """Render one temporal edge ``s -> r [ts] -> o``."""
    return f"{s} -> {r} [{ts}] -> {o}"


def temporal_path_to_string(quads: Iterable[tuple[str, str, str, str]]) -> str:
    """Render a path of quadruples into the timestamp-augmented format.

    ``quads`` is a sequence of ``(subject, relation, object, timestamp)``.
    Consecutive edges share the joining entity, matching GCR's rendering.
    """
    quads = list(quads)
    if not quads:
        return ""
    parts = []
    for i, (s, r, o, ts) in enumerate(quads):
        if i == 0:
            parts.append(f"{s} -> {r} [{ts}] -> {o}")
        else:
            parts.append(f" -> {r} [{ts}] -> {o}")
    return "".join(parts).strip()


def wrap_path(path_str: str) -> str:
    return f"{PATH_START_TOKEN}{path_str}{PATH_END_TOKEN}"


def extract_timestamps(path: str) -> list[TimePoint]:
    """Recover every timestamp appearing in a rendered path, in order."""
    out = []
    for m in _TS_IN_PATH_RE.finditer(path):
        out.append(TimePoint.parse(m.group(1)))
    return out


def extract_timestamp_strings(path: str) -> list[str]:
    return [m.group(1) for m in _TS_IN_PATH_RE.finditer(path)]
