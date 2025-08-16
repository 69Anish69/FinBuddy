# FinBuddy – Financial Assistant

[![Made with FastAPI](https://img.shields.io/badge/FastAPI-0.111+-009688?logo=fastapi&logoColor=white)](#)
[![Next.js](https://img.shields.io/badge/Next.js-14+-000000?logo=nextdotjs)](#)
[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)](#)
[![Prophet](https://img.shields.io/badge/Facebook%20Prophet-time%20series-6F42C1)](#)
[![MongoDB](https://img.shields.io/badge/MongoDB-Atlas%20or%20Local-47A248?logo=mongodb&logoColor=white)](#)
[![Azure Blob](https://img.shields.io/badge/Azure-Blob%20Storage-0078D4?logo=microsoftazure&logoColor=white)](#)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](#license)

> Upload a bank statement (PDF/CSV/XLSX) → OCR + Clean → Categorize & Detect anomalies → Forecast with Prophet → Get insights, loan strategies, and a one-click PDF report. Includes a voice assistant for hands-free navigation.

---

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Architecture](#architecture)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [Configuration](#configuration)
- [Data Ingestion & Mapping](#data-ingestion--mapping)
- [API Reference](#api-reference)
- [Voice Assistant (Quick Commands)](#voice-assistant-quick-commands)
- [Reports](#reports)
- [Docker Deployment](#docker-deployment)
- [Troubleshooting](#troubleshooting)
- [Roadmap](#roadmap)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgements](#acknowledgements)

---

## Overview

**FinBuddy** helps users understand and improve their financial health. It ingests bank statements, auto-categorizes expenses, flags unusual transactions, forecasts income/expense/savings with **Prophet**, suggests **loan repayment strategies**, and generates a **PDF report**. OCR is powered by **LayoutLM**, and artifacts can be saved to **MongoDB** and **Azure Blob Storage**.

---

## Key Features

- **Smart ingestion**: Upload **PDF/CSV/XLSX**; OCR + table extraction; robust column auto-mapping.
- **Auto-categorization** & **anomaly detection** for transactions.
- **Forecasts** with **Prophet** (Income / Expense / Savings) + scenario toggles (current / best / worst).
- **Goal planning**: Check feasibility and monthly contributions.
- **Loan strategies**: Snowball/avalanche comparisons; APR savings.
- **One‑click PDF report** with charts and insights.
- **Voice assistant**: Navigate, export, and query insights hands‑free.
- **Secure storage** (optional): **MongoDB** for metadata and **Azure Blob** for files.

---

## Architecture

**User → Frontend (Next.js/React) → Backend (FastAPI) ↔ OCR (LayoutLM) → MongoDB + Azure Blob → Frontend (Charts & Insights)**

```
[User]
   │
   ▼
[Next.js Frontend]  ──►  /upload  ──►  [FastAPI Backend]
   │                               ├─►  OCR/Table Extract (LayoutLM/Tesseract)
   │                               ├─►  Cleaning & Mapping (date/credit/debit)
   │                               ├─►  Categorization & Anomaly Detection
   │                               ├─►  Forecasting (Prophet)
   │                               ├─►  Loan/Goal Engines
   │                               └─►  Report (PDF)
   │
   ├── Web UI (Charts/Reports)
   └── Voice Assistant (commands → API)
                    │
                    ▼
          [MongoDB] & [Azure Blob] (optional persistence)
```

**Data flow**: Upload → Normalize → Analyze → Persist (optional) → Visualize/Report.

---

## Tech Stack

- **Frontend**: Next.js 14, React, shadcn/ui, Recharts, Tailwind
- **Backend**: FastAPI, Uvicorn, Pydantic
- **ML/Analytics**: Prophet, pandas, numpy, scikit-learn (categorization/anomalies)
- **OCR**: LayoutLM (Hugging Face) with optional Tesseract fallback
- **Storage**: MongoDB (Atlas/local), Azure Blob Storage
- **Reports**: reportlab (PDF), Matplotlib (charts)
- **Infra**: Docker & docker‑compose

---

## Project Structure

```
FinBuddy/
├─ frontend/                      # Next.js app
│  ├─ app/ (or pages/)            # Routes (Dashboard, Forecast, Goals, Reports)
│  ├─ components/                 # UI components (Navigation, Footer, Cards)
│  ├─ env.example                 # Frontend environment sample
│  └─ package.json
├─ backend/
│  ├─ main.py                     # FastAPI app, routers include, CORS
│  ├─ routes/
│  │  ├─ upload.py                # /upload endpoint
│  │  ├─ forecast.py              # /forecast endpoint
│  │  ├─ report.py                # /report endpoints
│  │  └─ insights.py              # /insights, /loans, /goals (optional)
│  ├─ modules/
│  │  ├─ data_store.py            # in‑memory df store + disk fallback
│  │  ├─ ocr.py                   # LayoutLM/Tesseract helpers
│  │  ├─ cleaning.py              # normalize & map columns
│  │  ├─ categorize.py            # rules/ML for expense categories
│  │  ├─ anomalies.py             # anomaly detection
│  │  └─ charts.py                # chart utils for reports
│  ├─ static/logo.png             # branding for report
│  ├─ requirements.txt
│  └─ .env.example                # Backend environment sample
├─ docker-compose.yml
├─ Dockerfile.backend
├─ Dockerfile.frontend
└─ README.md
```

---

## Getting Started

### Prerequisites
- **Node.js** ≥ 18
- **Python** 3.10+
- **MongoDB** (optional) local or Atlas
- **Azure Storage account** (optional, for file blobs)

### 1) Clone
```bash
git clone https://github.com/69Anish69/FinBuddy/tree/master
cd FinBuddy
```

### 2) Backend Setup
```bash
cd backend
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # then edit values
uvicorn main:app --reload --port 8000
```

### 3) Frontend Setup
```bash
cd ../frontend
npm i
cp env.example .env.local  # then edit values
npm run dev  # http://localhost:3000
```

Open: `http://localhost:3000` (frontend) – API at `http://127.0.0.1:8000` by default.

---

## Configuration

### Backend `.env` (example)
```
# CORS
ALLOWED_ORIGINS=http://localhost:3000

# Mongo (optional)
MONGO_URI=mongodb://localhost:27017
MONGO_DB=finbuddy

# Azure Blob (optional)
AZURE_STORAGE_CONNECTION_STRING=DefaultEndpointsProtocol=...
AZURE_BLOB_CONTAINER=finbuddy-upload

# OCR/ML
OCR_MODEL=layoutlm-base-uncased
USE_TESSERACT_FALLBACK=true

# App
LOG_LEVEL=INFO
FALLBACK_UPLOAD_PATH=./uploads/latest.csv
```

### Frontend `.env.local` (example)
```
NEXT_PUBLIC_API_URL=http://127.0.0.1:8000
NEXT_PUBLIC_VOICE_LANG=en-US
```

---

## Data Ingestion & Mapping

- Accepts **PDF/CSV/XLSX**.
- Normalizes columns to `date`, `credit`, `debit` (auto-mapping from e.g. *Month/Income/Expense*).
- Generates a cleaned frame with canonical names: **Month**, **Income**, **Expense**.
- In-memory storage: `modules/data_store.stored_df` (with disk fallback if server restarts).

**Tip:** If your bank exports `Description/Credit ($)/Debit ($)`, the mapper aligns them automatically. Logs may show lines like:
```
🧩 Normalized columns: ['date', 'description', 'credit', 'debit']
✅ Cleaned preview:
        Month  Income  Expense
0 2024-01-01  1234.0    82.53
```

---

## API Reference

### `POST /upload`
- **Body**: multipart/form-data, `file` (PDF/CSV/XLSX)
- **Returns**: `{ summary, columns_mapped, sample }`

**curl**
```bash
curl -X POST http://127.0.0.1:8000/upload \
  -F "file=@/path/to/statement.pdf"
```

### `POST /forecast`
- **JSON**: `{ "months": 6, "target": "Income" | "Expense" | "Savings", "scenario": "current" | "best" | "worst" }`
- **Returns**: `{ series, text, chart_png_b64 }`

**curl**
```bash
curl -X POST http://127.0.0.1:8000/forecast \
  -H "Content-Type: application/json" \
  -d '{"months":6,"target":"Expense","scenario":"current"}'
```

### `GET /report/overview`
- **Returns**: overall status + insights
```json
{
  "updated_at": "2025-08-15T20:10:00Z",
  "status": { "forecast":"ready", "goals":"ready", "loans":"ready", "risk":"ready" },
  "score": 72,
  "insights": { "ef_months":3.2, "apr_save_12m":412.5, "invest_uplift_pct":4.2, "goal_buffer_months":2 },
  "improvements": ["Cut food delivery by 15%","Negotiate internet plan"]
}
```

### `POST /report/generate`
- **Returns**: PDF file download
```bash
curl -OJ http://127.0.0.1:8000/report/generate
```

### `GET /health`
- Returns `{ "status":"ok" }`

---

## Voice Assistant (Quick Commands)

Try:
- “Go to forecast”
- “Export report as PDF”
- “Give suggestions to save”
- “Pay loan faster with $50 extra”
- “Help”

**Note (Next.js client):** Use only `NEXT_PUBLIC_*` vars in client code. Avoid direct `process.env.*` in browser runtime.

---

## Reports

- PDF includes: summary metrics, forecasts, anomaly highlights, loan plan, and next actions.
- Branding: `backend/static/logo.png` used as watermark header.
- Each uploaded file yields **dynamic** insights (no static templates).

---

## Docker Deployment

**docker-compose.yml (sample)**
```yaml
services:
  backend:
    build:
      context: ./backend
      dockerfile: ../Dockerfile.backend
    env_file:
      - ./backend/.env
    ports: ["8000:8000"]
    depends_on: [mongo]
  frontend:
    build:
      context: ./frontend
      dockerfile: ../Dockerfile.frontend
    env_file:
      - ./frontend/.env.local
    ports: ["3000:3000"]
  mongo:
    image: mongo:6
    ports: ["27017:27017"]
    volumes:
      - mongo_data:/data/db

volumes:
  mongo_data:
```

**Run**
```bash
docker compose up --build -d
```

---

## Troubleshooting

- **`NameError: name 'forecast' is not defined`**
  - Ensure routers are imported and included:
    ```python
from routes import upload, forecast, report, insights
app.include_router(upload.router)
app.include_router(forecast.router)
app.include_router(report.router)
app.include_router(insights.router)
```

- **Prophet**: `ValueError: Dataframe has less than 2 non-NaN rows.`
  - Check mapping produced at upload; ensure `Month/Income/Expense` exist and dates parse to daily frequency.
  - Remove blank rows after OCR; verify date format `%Y-%m-%d`.

- **Next.js client**: `ReferenceError: process is not defined`
  - Only reference `process.env.NEXT_PUBLIC_*` in client components.
  - Or compute a safe fallback:
    ```ts
const API_BASE = (process.env.NEXT_PUBLIC_API_URL?.replace(/\/+$/, ''))
  ?? (typeof window !== 'undefined' && window.location.port === '3000' ? 'http://127.0.0.1:8000' : '');
```

- **CORS**: If frontend can’t reach API
  - Add CORSMiddleware in `main.py` and set `ALLOWED_ORIGINS`.

- **PDF looks identical across uploads**
  - Make sure `modules/data_store.stored_df` is updated on each upload and fallback path reads the newest file.
  - Verify you call the report endpoints **after** a successful upload/forecast for that session.

---

## Roadmap

- Plaid/Direct bank connectors
- Multi-account consolidation & budgeting
- Fine-tuned categorization model per user
- What‑if simulator for goals & investments

---

## Contributing

Issues and PRs welcome! Please open an issue for large changes first.

---

## License

This project is licensed under the **MIT License**. See `LICENSE` for details.

---

## Acknowledgements

- Facebook/Meta Prophet
- Hugging Face (LayoutLM)
- FastAPI & Pydantic teams
- shadcn/ui & Recharts communities
