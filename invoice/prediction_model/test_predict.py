import sys
import os
from pdf2image import convert_from_path
from invoice_ocr.prediction_model.inference import predict_invoice_fields

# Setup project path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "..", ".."))
sys.path.append(project_root)

# 📄 PDF path
pdf_path = os.path.join(project_root, "invoice_ocr", "invoice", "sample_invoice.pdf")

# 🖼️ Convert first page of PDF to image
images = convert_from_path(pdf_path, dpi=300)
if not images:
    raise ValueError("No pages found in PDF!")
image_path = os.path.join(project_root, "invoice_ocr", "invoice", "temp_page.png")
images[0].save(image_path)

# 🧠 Run prediction on the image
result = predict_invoice_fields(image_path)

print("Predicted Fields:")
print(result)
