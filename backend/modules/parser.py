import pdfplumber
import pandas as pd
import io

def parse_financial_data(file):
    filename = file.name.lower()
    if filename.endswith(".pdf"):
        return parse_pdf(file)
    elif filename.endswith((".xlsx", ".xls", ".csv")):
        return parse_excel(file)
    else:
        raise ValueError("Unsupported file format")


def parse_pdf(file):
    with pdfplumber.open(file) as pdf:
        text = ''
        for page in pdf.pages:
            text += (page.extract_text() or '') + '\n'

    lines = [line.strip() for line in text.split("\n") if line.strip()]
    rows = []
    for line in lines:
        parts = line.split()
        if len(parts) >= 4:
            try:
                month = parts[0]
                income = float(parts[1].replace(",", "").replace("$", ""))
                expense = float(parts[2].replace(",", "").replace("$", ""))
                desc = " ".join(parts[3:])
                rows.append({
                    "Month": month,
                    "Income": income,
                    "Expense": expense,
                    "Description": desc
                })
            except:
                continue

    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError("Unable to parse useful financial data from the PDF")
    return df


def parse_excel(file):
    if file.name.endswith(".csv"):
        df = pd.read_csv(file)
    else:
        df = pd.read_excel(file)

    df.columns = [col.strip() for col in df.columns]

    # Try to rename common variants
    column_map = {
        "income": "Income",
        "amount in": "Income",
        "inflow": "Income",
        "debit": "Expense",
        "amount out": "Expense",
        "expense": "Expense",
        "month": "Month",
        "description": "Description",
        "details": "Description"
    }

    renamed = {}
    for col in df.columns:
        for key in column_map:
            if key in col.lower():
                renamed[col] = column_map[key]
                break

    df = df.rename(columns=renamed)

    required = {"Income", "Expense"}
    if not required.issubset(df.columns):
        raise ValueError("Missing expected columns: Income, Expense")

    return df
