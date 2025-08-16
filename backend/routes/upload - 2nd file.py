
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
    # always reset cached frame so this upload becomes the source of truth
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

        # hard stop if nothing usable (prevents fake flat charts)
        if (df.get("Income", 0) == 0).all() and (df.get("Expense", 0) == 0).all():
            raise HTTPException(400, "Could not detect income or expense data in this file.")

        # Forecast view
        req_forecast = ["Month", "Income", "Expense"]
        if not all(c in df.columns for c in req_forecast):
            raise HTTPException(400, f"Missing required forecast columns: {req_forecast}")
        fc = df[req_forecast].copy()

        # Save for old code (attrs)
        data_store.stored_df = fc.copy()
        data_store.stored_df_loans = df[["date","credit","debit","Description"]].copy()

        # Save for new report endpoints (KV)
        data_store.set("stored_df", fc.copy())
        data_store.set("uploaded_df", df.copy())
        data_store.set("transactions_df", df.copy())

        # Emergency/cash balance
        cols_lc = {c.lower(): c for c in df.columns}
        for key in ["balance","closing balance","ending balance","available balance","current balance","eod balance","closingbalance"]:
            if key in cols_lc:
                s = pd.to_numeric(df[cols_lc[key]], errors="coerce").dropna()
                if not s.empty:
                    data_store.set("emergency_fund_balance", float(max(0.0, s.iloc[-1])))
                break

        # Credit card info
        def last_num(colname):
            s = pd.to_numeric(df[colname], errors="coerce").dropna()
            return float(s.iloc[-1]) if not s.empty else None

        bal_col = next((cols_lc[k] for k in cols_lc if k in ["cc_balance","credit card balance","card balance","outstanding","current balance (card)"]), None)
        lim_col = next((cols_lc[k] for k in cols_lc if k in ["cc_limit","credit limit","card limit"]), None)
        apr_col = next((cols_lc[k] for k in cols_lc if k in ["apr","interest rate","card apr"]), None)

        if bal_col:
            v = last_num(bal_col);  data_store.set("cc_balance", max(0.0, v or 0.0))
        if lim_col:
            v = last_num(lim_col);  data_store.set("cc_limit",  max(0.0, v or 0.0))
        if apr_col:
            v = last_num(apr_col)
            if v is not None:
                data_store.set("cc_apr", v/100.0 if v > 1 else v)

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

        # Save to disk
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

def clean_and_standardize_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [re.sub(r"[^a-z0-9]+", "", str(col).lower()) for col in df.columns]
    print("🧩 Normalized columns:", df.columns.tolist())

    if df.columns.duplicated().any():
        df = df.loc[:, ~df.columns.duplicated()].copy()
        print("🧩 Deduped columns:", df.columns.tolist())

    def find_col(cands):
        cols = list(df.columns)
        lc = [c.lower() for c in cols]
        for cand in cands:
            cand_l = cand.lower()
            for i, name in enumerate(lc):
                if name == cand_l:
                    return cols[i]
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

    if credit_col and desc_col and credit_col == desc_col:
        credit_col = None
    if debit_col and desc_col and debit_col == desc_col:
        debit_col = None

    if not date_col:
        df["Month"] = pd.date_range(start="2024-01-01", periods=len(df), freq="D")
    else:
        df["Month"] = pd.to_datetime(df[date_col], errors="coerce")

    if desc_col:
        col = df[desc_col]
        if isinstance(col, pd.DataFrame):
            col = col.iloc[:, 0]
        df["Description"] = col.astype(str)
    else:
        df["Description"] = ""

    def num(s):
        return pd.to_numeric(
            pd.Series(s).astype(str).str.replace(r"[^\d\.\-]", "", regex=True),
            errors="coerce",
        ).fillna(0.0)

    df["Income"]  = num(df[credit_col]) if credit_col else 0.0
    df["Expense"] = num(df[debit_col])  if debit_col  else 0.0

    if not credit_col and not debit_col and amount_col:
        amt = num(df[amount_col])
        df["Income"]  = amt.where(amt > 0, 0.0)
        df["Expense"] = (-amt).where(amt < 0, 0.0)

    df["date"]   = df["Month"]
    df["credit"] = df["Income"].fillna(0.0)
    df["debit"]  = df["Expense"].fillna(0.0)

    df = df.dropna(subset=["Month"])
    print("✅ Cleaned preview:\n", df[["Month","Income","Expense"]].head())
    print("🔗 Canonical for loans:\n", df[["date","credit","debit","Description"]].head())
    return df

@router.get("/upload/last")
async def get_last_upload_time():
    ts = getattr(data_store, "last_upload_time", None)
    return {"last_upload": None if not ts else time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(ts))}
