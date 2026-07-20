"""
Loaders for real context-aware recommendation (CARS) benchmarks, matching
the datasets referenced in Section 7.1: movies, points of interest, and
music domains with heterogeneous contextual variables (time, location,
mood, social setting).

IMPORTANT: this repository does not ship any dataset file. For the double
-blind review period, no external network calls or dataset redistribution
are performed here; you must download the datasets yourself from their
official sources and point these loaders at your local copy. Expected
schemas are documented below for the three benchmarks most commonly used
in the CARS literature (Adomavicius and Tuzhilin 2011; Baltrunas, Ludwig,
and Ricci 2011):

  * LDOS-CoMoDa  (movies; context: time, location, mood, social setting,
    weather, ...) -- https://www.tinkerhost.eu/... (see original paper)
  * Frappe        (mobile apps; context: time, location (lat/lon), weather,
    daytime, weekday, isweekend, homework, cost, city)
  * STS / Foursquare-style POI datasets (points of interest; context:
    time-of-day, day-of-week, weather, companion, mood, transport mode)

Expected CSV schema (long/tidy format), one row per (user, item,
interaction, context) observation:

    user_id, item_id, rating, timestamp, <context_col_1>, <context_col_2>, ...

Each <context_col_k> may contain either:
  (a) a single categorical value (deterministic legacy format), which will
      be treated as a degenerate distribution (P(c)=1), reproducing
      classical deterministic CARS as required by Theorem 2 (reduction
      property) -- this is the correct baseline setting; or
  (b) a JSON-encoded dict of {value: probability} pairs, allowing you to
      inject externally estimated contextual uncertainty (e.g. from a
      positioning confidence score or a mood classifier's softmax output).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from ..context import ContextualVariable, ProbabilisticContextInstance


@dataclass
class CARSDataset:
    """A minimal in-memory container for a loaded CARS benchmark."""

    interactions: pd.DataFrame  # columns: user_id, item_id, rating, timestamp
    context_columns: List[str]
    name: str

    def context_instance_for_row(self, row: pd.Series) -> ProbabilisticContextInstance:
        """Builds a ProbabilisticContextInstance (Definition 2) from one
        interaction row, handling both deterministic and probabilistic
        (JSON-encoded) context columns as described in the module docstring.
        """
        variables = []
        for col in self.context_columns:
            raw = row[col]
            dist = _parse_context_cell(raw)
            variables.append(ContextualVariable(name=col, distribution=dist))
        return ProbabilisticContextInstance(categorical_variables=variables)


def _parse_context_cell(raw) -> Dict:
    """Parses one context cell into a {value: probability} distribution."""
    if isinstance(raw, dict):
        return dict(raw)
    if isinstance(raw, str):
        stripped = raw.strip()
        if stripped.startswith("{"):
            try:
                parsed = json.loads(stripped)
                return {k: float(v) for k, v in parsed.items()}
            except (json.JSONDecodeError, ValueError):
                pass
        return {stripped: 1.0}
    # Numeric / other scalar -> degenerate distribution.
    return {raw: 1.0}


def load_csv_dataset(
    path: str,
    context_columns: List[str],
    user_col: str = "user_id",
    item_col: str = "item_id",
    rating_col: str = "rating",
    timestamp_col: Optional[str] = "timestamp",
    name: Optional[str] = None,
) -> CARSDataset:
    """Generic loader for any tidy CSV following the schema documented in
    the module docstring. Use this directly for LDOS-CoMoDa, Frappe, or
    any other CARS dataset once converted to the expected long format.
    """
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Dataset file not found at '{csv_path}'. This repository does "
            f"not ship dataset files -- download the benchmark from its "
            f"official source and place it at this path, or pass a "
            f"different --data-path. See loaders.py docstring for schema."
        )
    df = pd.read_csv(csv_path)

    required = {user_col, item_col, rating_col, *context_columns}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Dataset at '{csv_path}' is missing expected columns: {missing}")

    rename = {user_col: "user_id", item_col: "item_id", rating_col: "rating"}
    if timestamp_col and timestamp_col in df.columns:
        rename[timestamp_col] = "timestamp"
    df = df.rename(columns=rename)

    return CARSDataset(interactions=df, context_columns=context_columns, name=name or csv_path.stem)


# Convenience presets documenting the expected context columns for the
# datasets cited in Section 7.1, to be used once the corresponding CSV
# (converted to the tidy schema above) is placed under `data/`.
DATASET_PRESETS = {
    "ldos_comoda": dict(
        context_columns=["time", "location", "mood", "social", "weather", "endEmo"],
        rating_col="rating",
    ),
    "frappe": dict(
        context_columns=["daytime", "weekday", "isweekend", "weather", "city", "homework", "cost"],
        rating_col="cnt",  # implicit-feedback count, treat as a relevance proxy
    ),
}


def load_preset(preset: str, path: str) -> CARSDataset:
    if preset not in DATASET_PRESETS:
        raise ValueError(f"Unknown preset '{preset}'. Available: {list(DATASET_PRESETS)}")
    cfg = DATASET_PRESETS[preset]
    return load_csv_dataset(path, name=preset, **cfg)
