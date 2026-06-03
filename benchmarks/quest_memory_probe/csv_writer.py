# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Schema-tolerant CSV writer for probe rows.

Header = union of all keys across rows. Missing values render as empty cells.
None renders as empty cell (NOT 'None'). All values stringified via str().
"""

from __future__ import annotations

import csv
from collections.abc import Iterable
from pathlib import Path
from typing import Any


def write_rows(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    rows = list(rows)
    if not rows:
        return
    keys: list[str] = []
    seen: set[str] = set()
    for r in rows:
        for k in r:
            if k not in seen:
                seen.add(k)
                keys.append(k)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(keys)
        for r in rows:
            w.writerow(["" if r.get(k) is None else r[k] for k in keys])
