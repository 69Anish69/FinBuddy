"""
invoice_model.py
----------------
Loads your fine-tuned LayoutLM model once and exposes
extract_invoice_fields(file_obj) → dict
"""

import io
from PIL import Image
import pytesseract
from pytesseract import Output
import torch
from transformers import LayoutLMTokenizer, LayoutLMForTokenClassification

# ───────────────────────────────────────────────
# 1. Load tokenizer & model ONE TIME at import
# ───────────────────────────────────────────────
# Change to your local model path or HF name
MODEL_NAME = "layoutlm-finetuned-sroie"  # e.g. "./layoutlm_sroie_model"

tokenizer = LayoutLMTokenizer.from_pretrained(MODEL_NAME)
model = LayoutLMForTokenClassification.from_pretrained(MODEL_NAME)

id2label = model.config.id2label

# ───────────────────────────────────────────────
# 2. Helper: normalize bbox to 0-1000 space
# ───────────────────────────────────────────────
def _normalize_box(box, width, height):
    x0, y0, x1, y1 = box
    return [
        int(x0 * 1000 / width),
        int(y0 * 1000 / height),
        int(x1 * 1000 / width),
        int(y1 * 1000 / height),
    ]

# ───────────────────────────────────────────────
# 3. Public function called by Flask
# ───────────────────────────────────────────────
def extract_invoice_fields(file_obj) -> dict:
    """
    Parameters
    ----------
    file_obj : werkzeug.datastructures.FileStorage
        The uploaded PDF/PNG/JPG stream from Flask (request.files['file'])

    Returns
    -------
    dict
        { "vendor": "...", "date": "...", "amount": "..." }
        Keys depend on your fine-tuned label set.
    """
    # If PDF, you’d convert first → PIL image list. For brevity we assume image.
    image_bytes = file_obj.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    width, height = image.size

    # OCR with Tesseract
    ocr_data = pytesseract.image_to_data(image, output_type=Output.DICT)

    words, boxes = [], []
    for i, text in enumerate(ocr_data["text"]):
        word = text.strip()
        if not word:
            continue
        x, y, w, h = (ocr_data["left"][i], ocr_data["top"][i],
                      ocr_data["width"][i], ocr_data["height"][i])
        words.append(word)
        boxes.append(_normalize_box([x, y, x + w, y + h], width, height))

    # Tokenize
    enc = tokenizer(words,
                    is_split_into_words=True,
                    return_tensors="pt",
                    truncation=True,
                    padding="max_length",
                    max_length=512)

    enc["bbox"] = torch.tensor(
        [boxes + [[0, 0, 0, 0]] * (512 - len(boxes))]
    )

    # Predict
    model.eval()
    with torch.no_grad():
        outputs = model(input_ids=enc["input_ids"],
                        attention_mask=enc["attention_mask"],
                        bbox=enc["bbox"])
    tokens  = tokenizer.convert_ids_to_tokens(enc["input_ids"][0])
    labels  = torch.argmax(outputs.logits, dim=2)[0].tolist()

    # Collect words by label
    collected = {}
    for token, lbl_idx, word in zip(tokens, labels, words):
        if token.startswith("##"):
            continue
        label = id2label[lbl_idx]
        if label != "O":          # Outside = ignore
            collected.setdefault(label, []).append(word)

    return {k.lower(): " ".join(v) for k, v in collected.items()}