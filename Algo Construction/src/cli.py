import os
import argparse
from collections import defaultdict

from io_utils import load_ocr_json, save_paragraphs
from layout_utils import (
    group_words_into_lines,
    group_lines_into_paragraphs
)

def merge_across_pages(paragraphs, indent_tol=20):
    """
    Merge paragraphs that actually continue from page N to N+1.
    - indent_tol: pixel tolerance for left-margin alignment
    """
    merged = []
    for para in paragraphs:
        if not merged:
            merged.append(para)
            continue

        prev = merged[-1]
        # if page boundary
        if para['page'] == prev['page'] + 1:
            # check left-indent alignment
            prev_left = prev['lines'][-1]['bbox'][0]
            curr_left = para['lines'][0]['bbox'][0]
            indent_ok = abs(curr_left - prev_left) <= indent_tol

            # check that prev para does NOT end in strong punctuation
            last_char = prev['text'].rstrip()[-1] if prev['text'].strip() else ''
            no_hard_stop = last_char not in '.!?:;'

            if indent_ok and no_hard_stop:
                # merge into prev
                # join with a space to avoid accidental run-together
                prev['text'] = prev['text'].rstrip() + ' ' + para['text'].lstrip()
                # union bbox
                xs = [*prev['bbox'][0:2], *prev['bbox'][2:4],
                      *para['bbox'][0:2], *para['bbox'][2:4]]
                ys = xs  # reuse same list for simplicity
                ys = [*prev['bbox'][1:3], *para['bbox'][1:3]]
                prev['bbox'] = [
                    min(xs[0::2]), min(ys[0::2]),
                    max(xs[1::2]), max(ys[1::2])
                ]
                # extend lines
                prev['lines'].extend(para['lines'])
                continue

        merged.append(para)
    return merged

def main():
    parser = argparse.ArgumentParser(
        description="Reconstruct paragraphs from OCR JSON (PDFMiner style)")
    parser.add_argument("-i", "--input",    default="data/odr_sample.json")
    parser.add_argument("-o", "--output",   default="outputs/paragraphs.json")
    parser.add_argument("-t", "--text-output", default="outputs/paragraphs.txt")
    args = parser.parse_args()

    data = load_ocr_json(args.input)

    # collect words page-wise
    pages = defaultdict(list)
    for page_str, page_data in data.get('pages', {}).items():
        page_num = int(page_str)
        for w in page_data.get('layout', [])[1:]:
            xs = [v['x'] for v in w['boundingPoly']['vertices']]
            ys = [v['y'] for v in w['boundingPoly']['vertices']]
            bbox = [min(xs), min(ys), max(xs), max(ys)]
            pages[page_num].append({'text': w['description'], 'bbox': bbox})

    # build paragraphs
    all_paras = []
    for pg, word_list in sorted(pages.items()):
        lines = group_words_into_lines(word_list)
        paras = group_lines_into_paragraphs(lines)
        for p in paras:
            p['page'] = pg
        all_paras.extend(paras)

    # merge those that continue across pages
    all_paras = merge_across_pages(all_paras, indent_tol=20)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    os.makedirs(os.path.dirname(args.text_output), exist_ok=True)

    save_paragraphs(args.output, all_paras)
    print(f"Saved {len(all_paras)} paragraphs to {args.output}")

    with open(args.text_output, 'w', encoding='utf-8') as f:
        for para in all_paras:
            f.write(f"--- Page {para['page']} ---\n")
            f.write(para['text'].strip() + "\n\n")
    print(f"Saved plain-text paragraphs to {args.text_output}")


if __name__ == "__main__":
    main()
