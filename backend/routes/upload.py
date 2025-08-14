
import time
import os
import io
import re
import pandas as pd
import pdfplumber
import traceback
from fastapi import APIRouter, UploadFile, File, HTTPException
import modules.data_store as data_store

UPLOAD_DIR = "uploaded_files"
os.makedirs(UPLOAD_DIR, exist_ok=True)

router = APIRouter()

@router.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    # Always reset cached frame so this upload becomes the source of truth
    data_store.stored_df = None

    filename = (file.filename or "").lower()
    contents = await file.read()

    try:
        # -------- Parse file into raw df --------
        if filename.endswith(".pdf"):
            with pdfplumber.open(io.BytesIO(contents)) as pdf:
                tables = []
                for page in pdf.pages:
                    table = page.extract_table()
                    if table:
                        tables.extend(table)
            print("📄 Raw PDF tables extracted:", len(tables))
            if not tables or not isinstance(tables[0], list) or len(tables[0]) < 2:
                raise HTTPException(400, "No usable table found in PDF bank statement.")
            header = tables[0]
            rows = [row for row in tables[1:] if len(row) == len(header)]
            if not rows:
                raise HTTPException(400, "PDF table rows are empty or malformed.")
            df = pd.DataFrame(rows, columns=header)
            print("📄 Cleaned PDF DataFrame:\n", df.head(3))

        elif filename.endswith(".csv"):
            df = pd.read_csv(io.BytesIO(contents))
        elif filename.endswith((".xls", ".xlsx")):
            df = pd.read_excel(io.BytesIO(contents))
        else:
            raise HTTPException(415, "Only PDF, CSV, XLS, or XLSX files are supported.")

        # -------- Normalize to canonical schema --------
        df = clean_and_standardize_df(df)

        # Hard stop if nothing usable (prevents fake flat charts)
        if (df.get("Income", 0) == 0).all() and (df.get("Expense", 0) == 0).all():
            raise HTTPException(400, "Could not detect income or expense data in this file.")

        # Forecast view
        req_forecast = ["Month", "Income", "Expense"]
        if not all(c in df.columns for c in req_forecast):
            raise HTTPException(400, f"Missing required forecast columns: {req_forecast}")
        fc = df[req_forecast].copy()

        # Transactions view
        req_loans = ["date", "credit", "debit", "Description"]
        if not all(c in df.columns for c in req_loans):
            raise HTTPException(400, f"Missing required transaction columns: {req_loans}")
        tx = df[req_loans].copy()

        # -------- Persist for all code paths --------
        data_store.stored_df = fc.copy()
        data_store.stored_df_loans = tx.copy()
        data_store.last_upload_time = time.time()

        if hasattr(data_store, "set"):
            data_store.set("stored_df", fc.copy())
            data_store.set("uploaded_df", df.copy())
            data_store.set("transactions_df", tx.copy())

        print("🧠 stored_df (forecast):\n", data_store.stored_df.head(3))
        print("🧠 stored_df_loans:\n", data_store.stored_df_loans.head(3))

        # Save to disk snapshot
        os.makedirs(UPLOAD_DIR, exist_ok=True)
        filepath = os.path.join(UPLOAD_DIR, "latest_upload.csv")
        df.to_csv(filepath, index=False)
        print("💾 File saved to disk as:", filepath)

        return {
            "ok": True,
            "message": "✅ File uploaded and ready.",
            "rows_forecast": int(len(fc)),
            "cols_forecast": list(fc.columns),
            "rows_transactions": int(len(tx)),
            "cols_transactions": list(tx.columns),
        }

    except HTTPException:
        raise
    except Exception as e:
        print("❌ Upload error:", e)
        traceback.print_exc()
        raise HTTPException(400, f"Upload failed: {e}")

@router.post("/upload/", include_in_schema=False)
async def upload_file_slash(file: UploadFile = File(...)):
    return await upload_file(file)


# ---------- Normalization ----------
def clean_and_standardize_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Produce a unified frame with:
      Month (datetime), Income, Expense (float),
      Description (str), and date/credit/debit for loans.
    """
    df = df.copy()

    # 1) Normalize headers
    df.columns = [re.sub(r"[^a-z0-9]+", "", str(col).lower()) for col in df.columns]
    print("🧩 Normalized columns:", df.columns.tolist())

    if df.columns.duplicated().any():
        df = df.loc[:, ~df.columns.duplicated()].copy()
        print("🧩 Deduped columns:", df.columns.tolist())

    # 2) Safer column finder: exact match first, then starts/endswith, never loose substrings
    def find_col(cands):
        cols = list(df.columns)
        lc = [c.lower() for c in cols]
        # exact
        for cand in cands:
            cand_l = cand.lower()
            for i, name in enumerate(lc):
                if name == cand_l:
                    return cols[i]
        # conservative contains: startswith/endswith only
        for cand in cands:
            cand_l = cand.lower()
            for i, name in enumerate(lc):
                if name.startswith(cand_l) or name.endswith(cand_l):
                    return cols[i]
        return None

    date_col   = find_col(["date", "month", "posteddate", "transactiondate", "statementdate"])
    desc_col   = find_col(["description","narration","details","merchant","memo","payee","transactiondetails","transactiondetail"])
    credit_col = find_col(["income","credit","deposit","inflow","amountcredit","credits","paidin"])
    debit_col  = find_col(["expense","debit","withdrawal","outflow","amountdebit","debits","paidout"])
    amount_col = find_col(["amount","amt","value"])

    # 3) Month
    if not date_col:
        df["Month"] = pd.date_range(start="2024-01-01", periods=len(df), freq="D" )
    else:
        df["Month"] = pd.to_datetime(df[date_col], errors="coerce" )

    # 4) Description
    if desc_col:
        col = df[desc_col]
        if isinstance(col, pd.DataFrame):
            col = col.iloc[:, 0]
        df["Description"] = col.astype(str)
    else:
        df["Description"] = ""

    # 5) Numeric coercion (handles $, commas, spaces, and (1,234.56) negatives)
    def to_number(series_like):
        s = pd.Series(series_like)
        s = s.astype(str).str.strip()
        s = s.str.replace(r"[,\s$]", "", regex=True)
        # parentheses negatives -> -value
        s = s.str.replace(r"^\((.*)\)$", r"-\1", regex=True)
        return pd.to_numeric(s, errors="coerce").fillna(0.0)

    # 6) Build Income/Expense
    if credit_col or debit_col:
        inc = to_number(df[credit_col]) if credit_col else pd.Series(0.0, index=df.index)
        exp = to_number(df[debit_col])  if debit_col  else pd.Series(0.0, index=df.index)
        df["Income"]  = inc
        df["Expense"] = exp
    elif amount_col:
        amt = to_number(df[amount_col])
        df["Income"]  = amt.where(amt > 0, 0.0)
        df["Expense"] = (-amt).where(amt < 0, 0.0)
    else:
        df["Income"]  = 0.0
        df["Expense"] = 0.0

    # 7) Loans/transactions view
    df["date"]   = df["Month"]
    df["credit"] = df["Income"].fillna(0.0)
    df["debit"]  = df["Expense"].fillna(0.0)

    # 8) Final clean + quick debug counts
    df = df.dropna(subset=["Month"])
    print("✅ Cleaned preview:\n", df[["Month","Income","Expense"]].head())
    nz_inc = int((df["Income"]  > 0).sum())
    nz_exp = int((df["Expense"] > 0).sum())
    print(f"🔎 Non-zero counts → Income: {nz_inc}, Expense: {nz_exp}")

    return df

@router.get("/upload/last")
async def get_last_upload_time():
    ts = getattr(data_store, "last_upload_time", None)
    return {"last_upload": None if not ts else time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(ts))}
