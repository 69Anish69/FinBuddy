# FinBuddy Web Interface – Invoice OCR & Auto-Fill

This folder contains the web-based frontend and Flask backend for the **Invoice OCR & Auto-Fill** module of the FinBuddy project.

## 🌐 Overview

The web application allows users to:

- Upload invoice PDFs or images
- Automatically extract fields (Vendor, Date, Amount)
- View extracted data in a clean, user-friendly interface
- Navigate to a mock payment page with auto-filled data

## 📁 Folder Structure

```bash
Webpage/
├── app.py                   # Flask backend server
├── invoice_inference.py     # Field extraction logic (imports model)
├── templates/
│   ├── index.html           # Main UI (upload & form)
│   └── payment.html         # Payment page (uses auto-filled values)
├── static/
│   ├── style.css            # Custom styles
│   └── img2.png             # UI image asset
