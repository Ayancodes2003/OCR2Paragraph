# src/io_utils.py
import json

def load_ocr_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_paragraphs(out_path, paragraphs):
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(paragraphs, f, ensure_ascii=False, indent=2)
