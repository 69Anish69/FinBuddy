import time
from fastapi import APIRouter, UploadFile, File
import pandas as pd
import fitz  # PyMuPDF
import io
import os
from modules.chatbot import set_documents
import modules.data_store as data_store
import logging

logger = logging.getLogger("upload_pipeline")
logging.basicConfig(level=logging.INFO)
router = APIRouter()
UPLOAD_DIR = "uploaded_files"
os.makedirs(UPLOAD_DIR, exist_ok=True)

router = APIRouter()

@router.post("/mlops_services/upload_pipeline/")
async def upload_file(file: UploadFile = File(...)):
    filename = file.filename.lower()
    contents = await file.read()

    try:
        if filename.endswith(".pdf"):
            with fitz.open(stream=contents, filetype="pdf") as doc:
                text = "\n".join([page.get_text() for page in doc])
            set_documents([text])
            return {"message": "✅ PDF processed and ready for chatbot."}

        elif filename.endswith(".csv"):
            df = pd.read_csv(io.BytesIO(contents))
        elif filename.endswith((".xls", ".xlsx")):
            df = pd.read_excel(io.BytesIO(contents))
        else:
            return {"error": "❌ Only PDF, CSV, XLS, or XLSX files are supported."}

        df = clean_and_standardize_df(df)

        # ✅ Save to memory
        data_store.stored_df = df
        data_store.last_upload_time = time.time()

        # ✅ Save to disk (optional)
        filepath = os.path.join(UPLOAD_DIR, "latest_upload.csv")
        df.to_csv(filepath, index=False)

        print("✅ File parsed and saved. Columns:", df.columns)
        return {"message": "✅ File uploaded and ready for forecasting."}

    except Exception as e:
        print("❌ Upload error:", e)
        return {"error": f"❌ Upload failed: {str(e)}"}

def clean_and_standardize_df(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [col.strip().capitalize() for col in df.columns]

    rename_map = {
        "Date": "Month",
        "Credit": "Income",
        "Debit": "Expense",
        "Amount": "Income",
    }
    df.rename(columns=rename_map, inplace=True)

    # Fallbacks
    if "Month" not in df.columns:
        df["Month"] = pd.date_range(start="2024-01-01", periods=len(df), freq="M").strftime("%b %Y")
    if "Income" not in df.columns:
        df["Income"] = 0
    if "Expense" not in df.columns:
        df["Expense"] = 0

    return df
