import json
from typing import List, Dict, Any

def preprocess_json_to_lines(filepath: str) -> List[Dict[str, Any]]:
    """
    Loads OCR JSON data in the new format, extracts text blocks as lines,
    and returns a list of lines with their text and bounding boxes.

    Args:
        filepath (str): The path to the OCR JSON file.

    Returns:
        List[Dict[str, Any]]: A list of line objects, sorted by vertical position.
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error reading or parsing JSON file {filepath}: {e}")
        return []

    all_lines = []
    if 'pages' not in data or not isinstance(data['pages'], dict):
        print("Warning: JSON structure is missing 'pages' dictionary.")
        return []

    for page_num, page_data in data['pages'].items():
        if 'layout' not in page_data or not isinstance(page_data['layout'], list):
            continue

        # Heuristic: In some OCR outputs (like Google Vision), the first block
        # is the full text of the page. We can filter it out by looking for a
        # 'locale' key, which is present in the sample's full-text block.
        for block in page_data['layout']:
            if 'locale' in block:
                continue

            if 'description' in block and 'boundingPoly' in block:
                try:
                    vertices = block['boundingPoly']['vertices']
                    if not vertices or len(vertices) < 1:
                        continue
                    
                    xs = [v['x'] for v in vertices if 'x' in v]
                    ys = [v['y'] for v in vertices if 'y' in v]

                    if not xs or not ys:
                        continue

                    x0, y0 = min(xs), min(ys)
                    x1, y1 = max(xs), max(ys)
                    
                    # Treat the entire block as a single line, replacing internal
                    # newlines with spaces since we have one bbox for the whole block.
                    text = block['description'].strip().replace('\n', ' ')

                    if text:
                        all_lines.append({
                            "text": text,
                            "bbox": [x0, y0, x1, y1]
                        })

                except (KeyError, TypeError) as e:
                    print(f"Skipping a malformed block on page {page_num}: {e}")
                    continue
    
    # Sort lines by their top vertical coordinate to ensure correct reading order
    all_lines.sort(key=lambda line: line['bbox'][1])

    return all_lines 