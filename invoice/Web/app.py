"""
app.py – Flask backend for FinBuddy Invoice OCR
"""

from flask import (
    Flask, request, jsonify, send_from_directory, render_template
)
from werkzeug.utils import secure_filename
import os, sys, tempfile, mimetypes, pathlib

# ─────────────────────────────────────────────
#  Path setup - import invoice_inference
# ─────────────────────────────────────────────
BASE_DIR = pathlib.Path(__file__).resolve().parent          #  …/Webpage/
ROOT_DIR = BASE_DIR.parent                                  #  …/FinBuddy/
sys.path.append(str(ROOT_DIR))                              #  make root importable

from invoice_inference import extract_invoice_fields        #  ✅ now works everywhere

# ─────────────────────────────────────────────
#  Flask app  (templates folder = Webpage/templates)
# ─────────────────────────────────────────────
app = Flask(
    __name__,
    template_folder=str(BASE_DIR / "templates")             #  payment.html lives here
)

# ─────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────
ALLOWED_EXT = {".pdf", ".png", ".jpg", ".jpeg", ".tiff"}

def allowed_file(fname: str) -> bool:
    return pathlib.Path(fname).suffix.lower() in ALLOWED_EXT


def make_temp_copy(upload) -> str:
    """
    Save upload (FileStorage) to a temp file while preserving the original
    filename inside the path (so mock-rules can match it later).
    """
    tmp_dir   = tempfile.mkdtemp()                          # e.g. C:\\Temp\\tmpabc
    safe_name = secure_filename(upload.filename)            # sample_invoice.pdf
    temp_path = os.path.join(tmp_dir, safe_name)
    upload.save(temp_path)
    return temp_path


def map_layoutlm_to_ui(raw: dict) -> dict:
    """Convert raw LayoutLM labels to UI-friendly keys."""
    return {
        "vendor":  raw.get("B-COMPANY", ""),
        "date":    raw.get("B-DATE", "") or raw.get("B-INVOICE_DATE", ""),
        "amount":  raw.get("B-TOTAL", "") or raw.get("B-AMOUNT", ""),
        "raw":     raw,          # keep raw for debugging
    }

# ─────────────────────────────────────────────
#  Routes
# ─────────────────────────────────────────────
@app.route("/")
def index() -> str:
    """Serve homepage."""
    return send_from_directory(".", "index.html")

@app.route("/static/<path:fname>")
def static_files(fname):
    mime, _ = mimetypes.guess_type(fname)
    return send_from_directory("static", fname, mimetype=mime or "application/octet-stream")

@app.route("/upload", methods=["POST"])
def upload():
    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400
    if not allowed_file(file.filename):
        return jsonify({"error": "Unsupported file type"}), 400

    tmp_path = make_temp_copy(file)
    print(f"[INFO] Saved upload to {tmp_path}")

    try:
        raw_result = extract_invoice_fields(tmp_path)
        ui_payload = map_layoutlm_to_ui(raw_result)

        print("[DEBUG] Model output:", raw_result)
        print("[DEBUG] UI payload:", ui_payload)
        return jsonify(ui_payload)

    except Exception as exc:
        print("[ERROR]", exc)
        return jsonify({"error": str(exc)}), 500

    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass

@app.route("/payment")
def payment():
    vendor = request.args.get("vendor", "")
    amount = request.args.get("amount", "")
    return render_template("payment.html", vendor=vendor, amount=amount)

# ─────────────────────────────────────────────
#  Run
# ─────────────────────────────────────────────
if __name__ == "__main__":
    # visit http://127.0.0.1:5000/
    app.run(debug=True)
