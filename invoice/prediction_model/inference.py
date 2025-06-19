# inference.py  (keep the rest of your code unchanged)

from pathlib import Path
import torch
from PIL import Image
from transformers import LayoutLMTokenizer, LayoutLMForTokenClassification
import pytesseract
from pytesseract import Output

# ─────────────────────────────────────────
# 🔹 Locate model folder robustly
# ─────────────────────────────────────────
BASE_DIR  = Path(__file__).resolve().parent.parent   # ..../invoice_ocr
MODEL_DIR = BASE_DIR / "models" / "layoutlm_invoice" / "checkpoint-626"

tokenizer = LayoutLMTokenizer.from_pretrained("microsoft/layoutlm-base-uncased")
model     = LayoutLMForTokenClassification.from_pretrained(str(MODEL_DIR))
id2label  = model.config.id2label
model.eval()

# ─────────────────────────────────────────
# 🔹 Box Normalization (0–1000 scale)
# ─────────────────────────────────────────
def normalize_box(box, width, height):
    return [
        int(box[0] * 1000 / width),
        int(box[1] * 1000 / height),
        int(box[2] * 1000 / width),
        int(box[3] * 1000 / height),
    ]

# ─────────────────────────────────────────
# 🔹 Preprocess invoice image
# ─────────────────────────────────────────
def preprocess_image(image_path):
    image = Image.open(image_path).convert("RGB")
    width, height = image.size

    ocr_data = pytesseract.image_to_data(image, output_type=Output.DICT)

    words = []
    boxes = []

    for i in range(len(ocr_data["text"])):
        if int(ocr_data["conf"][i]) > 0:
            word = ocr_data["text"][i].strip()
            if word != "":
                words.append(word)
                bbox = [
                    ocr_data["left"][i],
                    ocr_data["top"][i],
                    ocr_data["left"][i] + ocr_data["width"][i],
                    ocr_data["top"][i] + ocr_data["height"][i],
                ]
                boxes.append(normalize_box(bbox, width, height))

    return words, boxes

# ─────────────────────────────────────────
# 🔹 Run LayoutLM Prediction
# ─────────────────────────────────────────
def predict_invoice_fields(image_path):
    words, boxes = preprocess_image(image_path)

    encoding = tokenizer(words, boxes=boxes, return_tensors="pt", truncation=True, padding="max_length", max_length=512)
    input_ids = encoding["input_ids"]
    attention_mask = encoding["attention_mask"]
    bbox = encoding["bbox"]

    with torch.no_grad():
        outputs = model(input_ids=input_ids, bbox=bbox, attention_mask=attention_mask)
        predictions = outputs.logits.argmax(dim=2)

    labels = [id2label[label.item()] for label in predictions[0]]
    result = {}

    for word, label in zip(words, labels):
        if label != "O":
            key = label.replace("B-", "")
            if key not in result:
                result[key] = word
            else:
                result[key] += " " + word

    return result
