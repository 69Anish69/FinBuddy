# FinBuddy – Webpage Frontend & Backend (Flask)

This folder contains the web interface for the FinBuddy project, including the Flask backend and the HTML/CSS/JS frontend.

## Features

- Upload invoice PDF or image
- Auto-extract fields using trained LayoutLM model
- Populate extracted values into a web form
- Redirect to payment page after extraction

## Structure

- `app.py`: Flask backend server
- `index.html`: Main UI for invoice upload
- `payment.html`: Dummy payment form
- `static/`: Contains CSS, JavaScript, and image assets
- `templates/`: Flask HTML templates (if used with `render_template`)

## How to Run

```bash
cd Webpage
pip install -r requirements.txt  # make sure dependencies are installed
python app.py
