# FinBuddy – Invoice OCR & Auto-Fill

FinBuddy is a smart finance assistant that automates invoice data extraction using AI-powered Optical Character Recognition (OCR). This module demonstrates automatic extraction of invoice fields such as vendor name, date, and amount from PDF and image files.

## 🔍 Use Case Overview

**Use Case 1: Invoice OCR & Auto-Fill**

- Extract key fields from invoices (Vendor, Date, Total)
- Auto-fill extracted data into web forms
- Supports PDFs and image formats (`.pdf`, `.png`, `.jpg`, `.jpeg`)
- Built with LayoutLM and Tesseract OCR

## 🧠 Technologies Used

- **Python**
- **Flask** (for backend API)
- **Tesseract** (OCR)
- **Transformers: LayoutLM** (token classification model)
- **pdf2image + PIL** (for handling invoice PDFs)
- **Bootstrap 5** (UI styling)
- **HTML/CSS + JavaScript** (frontend form and UI logic)
