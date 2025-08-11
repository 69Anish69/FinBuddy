from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from routes.upload import router as upload_router
from routes.forecast import router as forecast_router
from routes.goals import router as goals_router
from routes.user import router as user_router
from routes.loans import router as loans_router
from routes.loan_docs import router as loan_docs_router
from routes.report import router as report_router

import os
import pandas as pd
import modules.data_store as data_store

# put this near the top to silence Tk errors
import os
os.environ.setdefault("MPLBACKEND", "Agg")
try:
    import matplotlib
    matplotlib.use("Agg", force=True)
except Exception:
    pass

# ...
from routes import assistant


app = FastAPI()

from fastapi.middleware.cors import CORSMiddleware

UPLOAD_SNAPSHOT = os.path.join("uploaded_files", "latest_upload.csv")

@app.on_event("startup")
def seed_memory_from_disk():
    try:
        if os.path.exists(UPLOAD_SNAPSHOT):
            # import inside to avoid circular import
            from routes.upload import clean_and_standardize_df

            raw = pd.read_csv(UPLOAD_SNAPSHOT)
            df = clean_and_standardize_df(raw)

            fc = df[["Month", "Income", "Expense"]].copy()
            tx = df[["date", "credit", "debit", "Description"]].copy()

            # legacy attrs (older routes)
            data_store.stored_df = fc.copy()
            data_store.stored_df_loans = tx.copy()

            # KV keys (report.py reads these)
            if hasattr(data_store, "set"):
                data_store.set("stored_df", fc.copy())
                data_store.set("uploaded_df", df.copy())
                data_store.set("transactions_df", tx.copy())

            print(f"🔁 Seeded memory from {UPLOAD_SNAPSHOT}: {len(fc)} rows")
        else:
            print("ℹ️ No snapshot found yet, upload once to create it.")
    except Exception as e:
        print("⚠️ Startup seed skipped:", e)


origins = [
    "http://localhost:3000", "http://127.0.0.1:3000",
    "http://localhost:8080", "http://127.0.0.1:8080",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["Content-Disposition"],  # useful for downloads
)


# include each APIRouter object exactly once
app.include_router(upload_router)
app.include_router(forecast_router)
app.include_router(goals_router)
app.include_router(user_router)
app.include_router(loans_router)
app.include_router(loan_docs_router)
app.include_router(report_router)
app.include_router(assistant.router)

@app.get("/")
def root():
    return {"message": "🚀 FinBuddy API running"}
