from turtle import pd
from fastapi import UploadFile, File
from flask import app
from universal_parser import detect_file_type, read_csv_excel, extract_text_from_pdf, preprocess_dataframe

@app.post("/upload/")
async def upload_file(file: UploadFile = File(...)):
    content_type = detect_file_type(file.filename)

    if content_type and "pdf" in content_type:
        extracted_text = extract_text_from_pdf(await file.read())
        return {"text": extracted_text[:1000]}  # optional preview
    else:
        df = read_csv_excel(await file.read())
        processed = preprocess_dataframe(df)
        return {"summary": processed.head(5).to_dict() if isinstance(processed, pd.DataFrame) else processed}
