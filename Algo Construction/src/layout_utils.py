

from statistics import median


LINE_OVERLAP_THRESHOLD = 0.5
LINE_MARGIN_RATIO = 0.7
X_OVERLAP_THRESHOLD = 0.3

def vertical_overlap(a, b):
    y0a, y1a = a[1], a[3]
    y0b, y1b = b[1], b[3]
    overlap = max(0, min(y1a, y1b) - max(y0a, y0b))
    return overlap / min((y1a - y0a), (y1b - y0b))

def horizontal_overlap(a, b):
    x0a, x1a = a[0], a[2]
    x0b, x1b = b[0], b[2]
    overlap = max(0, min(x1a, x1b) - max(x0a, x0b))
    return overlap / min((x1a - x0a), (x1b - x0b))

def group_words_into_lines(words):
    words = sorted(words, key=lambda w: w['bbox'][1])
    lines, current = [], []
    for w in words:
        if not current or vertical_overlap(current[-1]['bbox'], w['bbox']) >= LINE_OVERLAP_THRESHOLD:
            current.append(w)
        else:
            lines.append(current)
            current = [w]
    if current:
        lines.append(current)

    line_objs = []
    for line in lines:
        line.sort(key=lambda w: w['bbox'][0])
        text = ' '.join(w['text'] for w in line)
        xs = [b for w in line for b in (w['bbox'][0], w['bbox'][2])]
        ys = [b for w in line for b in (w['bbox'][1], w['bbox'][3])]
        line_objs.append({
            'text': text,
            'bbox': [min(xs), min(ys), max(xs), max(ys)],
            'words': line
        })
    return line_objs

def group_lines_into_paragraphs(lines):
    lines = sorted(lines, key=lambda l: l['bbox'][1])
    paragraphs, current = [], []

    for i, line in enumerate(lines):
        if i == 0:
            current.append(line)
            continue
        prev = lines[i - 1]
        gap = line['bbox'][1] - prev['bbox'][3]
        avg_height = ((line['bbox'][3] - line['bbox'][1]) + (prev['bbox'][3] - prev['bbox'][1])) / 2
        x_overlap = horizontal_overlap(line['bbox'], prev['bbox'])

        if gap <= LINE_MARGIN_RATIO * avg_height and x_overlap >= X_OVERLAP_THRESHOLD:
            current.append(line)
        else:
            paragraphs.append(current)
            current = [line]
    if current:
        paragraphs.append(current)

    para_objs = []
    for para in paragraphs:
        xs = [ln['bbox'][0] for ln in para] + [ln['bbox'][2] for ln in para]
        ys = [ln['bbox'][1] for ln in para] + [ln['bbox'][3] for ln in para]
        para_objs.append({
            'text': '\n'.join(ln['text'] for ln in para),
            'bbox': [min(xs), min(ys), max(xs), max(ys)],
            'lines': para
        })
    return para_objs
#formula
#page boundaries-fix the break

#translated text-api
#embed the translated text on the original pdf
#bg color->replace the box with translated text