from fastapi import APIRouter, UploadFile, File
from modules.chatbot import set_documents
from modules.parser import parse_financial_data  # ✅ your smart parser
import io
import pandas as pd

router = APIRouter()
stored_df = None

@router.post("/upload/")
async def upload_file(file: UploadFile = File(...)):
    global stored_df
    filename = file.filename.lower()
    contents = await file.read()

    try:
        file_obj = io.BytesIO(contents)
        df = parse_financial_data(file_obj)
        stored_df = df

        # Optional: send raw text to chatbot if it's PDF
        if filename.endswith(".pdf"):
            set_documents(["\n".join(df["Description"].astype(str).tolist())])

        return {
            "message": "✅ File processed successfully!",
            "rows": len(df),
            "columns": list(df.columns)
        }

    except Exception as e:
        print("❌ Upload error:", e)
        return {"error": f"❌ Upload failed: {str(e)}"}
