# backend/state.py
"""
Simple in-memory state for FinBuddy.
Stores the latest cleaned transactions DataFrame for use across routes
(e.g., forecast, goals, strategies).

NOTE:
- This is process-local memory. Running multiple workers or using --reload
  will create separate processes with separate memory. For true sharing,
  persist to a DB or cache (e.g., Redis/Postgres/S3).
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from threading import RLock
from typing import Optional, Tuple, Dict, Any

import pandas as pd

# ---------- Data Model ----------

@dataclass
class MemoryMeta:
    """Lightweight metadata about the currently stored dataset."""
    source: str = "memory"          # e.g., "upload", "memory", "demo"
    rows: int = 0
    cols: int = 0
    last_updated: str = ""          # ISO timestamp
    notes: str = ""                 # optional message


# ---------- Module State ----------

_lock = RLock()
_stored_df: Optional[pd.DataFrame] = None
_meta: MemoryMeta = MemoryMeta(source="empty", rows=0, cols=0, last_updated="", notes="No data")


# ---------- Public API ----------

def set_stored_df(df: pd.DataFrame, *, source: str = "upload", notes: str = "") -> None:
    """
    Save a cleaned DataFrame in memory so other routes can use it.
    Expected columns (at minimum): Month (datetime), Income (float), Expense (float)
    Optional: Balance (string/float)
    """
    if df is None:
        raise ValueError("set_stored_df: df cannot be None")

    # Make a defensive copy
    local = df.copy()

    # Best-effort normalization of Month column to datetime (safe if already datetime)
    if "Month" in local.columns:
        local["Month"] = pd.to_datetime(local["Month"], errors="coerce")

    with _lock:
        global _stored_df, _meta
        _stored_df = local
        _meta = MemoryMeta(
            source=source,
            rows=int(len(local)),
            cols=int(len(local.columns)),
            last_updated=datetime.now(timezone.utc).isoformat(),
            notes=notes or "Stored via set_stored_df",
        )


def get_stored_df() -> Tuple[Optional[pd.DataFrame], MemoryMeta]:
    """
    Retrieve the stored DataFrame and its metadata.
    Returns (df, meta). df is None if nothing is stored yet.
    """
    with _lock:
        return _stored_df, _meta


def has_stored_df(min_rows: int = 1) -> bool:
    """Quick check for availability/health of the in-memory dataset."""
    with _lock:
        return _stored_df is not None and len(_stored_df) >= min_rows


def clear_stored_df(note: str = "Cleared by request") -> None:
    """Remove the dataset from memory."""
    with _lock:
        global _stored_df, _meta
        _stored_df = None
        _meta = MemoryMeta(source="empty", rows=0, cols=0, last_updated=datetime.now(timezone.utc).isoformat(), notes=note)


def preview(head: int = 5) -> Dict[str, Any]:
    """
    Return a lightweight JSON-safe preview for logging/diagnostics.
    Example shape:
    {
      "meta": {...},
      "columns": ["Month","Income","Expense","Balance"],
      "head": [ {...row1...}, ... ]
    }
    """
    with _lock:
        if _stored_df is None or len(_stored_df) == 0:
            return {"meta": asdict(_meta), "columns": [], "head": []}

        df = _stored_df.head(max(1, head))
        # Convert datetimes to ISO strings for safe JSON
        df = df.copy()
        for col in df.columns:
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                df[col] = df[col].dt.tz_localize("UTC", nonexistent="NaT", ambiguous="NaT", errors="ignore") \
                                   .dt.tz_convert("UTC") \
                                   .dt.strftime("%Y-%m-%dT%H:%M:%SZ")

        return {
            "meta": asdict(_meta),
            "columns": list(_stored_df.columns),
            "head": df.to_dict(orient="records"),
        }


__all__ = [
    "set_stored_df",
    "get_stored_df",
    "has_stored_df",
    "clear_stored_df",
    "preview",
    "MemoryMeta",
]
