# backend/modules/data_store.py
from typing import Any, Dict, Optional
import pandas as pd

# ---- Simple in-memory key–value store ----
# _store: Dict[str, Any] = {}


_store = {}
def set(key, value):  # <- add
    _store[key] = value

def get(key, default=None):  # <- add
    return _store.get(key, default)

def delete(key: str) -> None:
    _store.pop(key, None)

def keys():
    return list(_store.keys())

# ---- Legacy attribute-style fields (kept for older routes) ----
stored_df: Optional[pd.DataFrame] = None         # FORECAST (Month/Income/Expense)
stored_df_loans: Optional[pd.DataFrame] = None   # LOANS (date/credit/debit/Description)
last_upload_time: Optional[float] = None
