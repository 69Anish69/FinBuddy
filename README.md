# FinBuddy – Smart Invoice OCR & Auto‑Fill

FinBuddy is an AI-powered solution that extracts key information from invoices using a fine-tuned LayoutLM model and automatically fills forms via a clean web interface.

---

## 📁 Project Overview

This repository includes:

- `invoice_ocr/` – Python backend using LayoutLM for extracting fields from invoices.
- `Webpage/` – Frontend built with HTML, Bootstrap, and JavaScript to upload invoices and display extracted results.
- Flask server (in `Webpage/app.py`) connects both modules to provide seamless UI-to-Model integration.

---

## 🚀 Features

- ✅ Drag-and-drop invoice upload via web UI
- ✅ Extracts key fields: Vendor, Date, Amount
- ✅ Uses OCR (Tesseract) + LayoutLM for NER
- ✅ Real-time results displayed in browser
- ✅ Optional Payment flow simulation

---

## 🖥️ Demo Walkthrough

1. Open the app:  
  cd Webpage
  python app.py

Visit `http://127.0.0.1:5000` in your browser.

2. Upload an invoice PDF or image.

3. Extracted fields are displayed automatically:
- Vendor Name
- Invoice Date
- Total Amount

4. Click “Proceed to Payment” (mock flow).

---

## 📦 Installation

### 1. Clone the repo

git clone https://github.com/69Anish69/FinBuddy.git
cd FinBuddy/Webpage

### 2. Install dependencies
bash
Copy
Edit
pip install -r requirements.txt

FinBuddy/
│
├── invoice_ocr/         # LayoutLM model and inference logic
│   └── invoice_inference.py
│
├── Webpage/             # Frontend + Flask backend
│   ├── app.py
│   ├── index.html
│   ├── payment.html
│   ├── static/
│   └── templates/
│       └── payment.html
│
├── README.md
└── requirements.txt     # All dependencies


## Model Details
Model: LayoutLM Base

Fine-tuned on: Custom invoice NER dataset

Extracted Entities:
B-COMPANY
B-DATE
B-TOTAL
