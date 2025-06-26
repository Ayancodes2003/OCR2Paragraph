import cv2
import numpy as np
import matplotlib.pyplot as plt
import random

def visualize_paragraphs(lines, edges, paragraphs, image_path=None, output_path='visualization.png'):
    """
    Visualizes detected lines, their connections, and paragraph groupings.
    Args:
        lines (list): List of dicts with 'bbox' and 'text'.
        edges (list): List of (i, j) tuples for connected lines.
        paragraphs (list): List of lists of line indices (one per paragraph).
        image_path (str): Path to the original image (optional).
        output_path (str): Where to save the visualization.
    """
    # Determine canvas size
    all_x = [b for l in lines for b in [l['bbox'][0], l['bbox'][2]]]
    all_y = [b for l in lines for b in [l['bbox'][1], l['bbox'][3]]]
    width = int(max(all_x) + 100)
    height = int(max(all_y) + 100)

    # Load image or create blank canvas
    if image_path:
        img = cv2.imread(image_path)
        if img is None:
            img = np.ones((height, width, 3), dtype=np.uint8) * 255
    else:
        img = np.ones((height, width, 3), dtype=np.uint8) * 255

    # Assign a color to each paragraph
    para_colors = [tuple([random.randint(0, 255) for _ in range(3)]) for _ in paragraphs]
    line_to_para = {}
    for idx, para in enumerate(paragraphs):
        for line_idx in para:
            line_to_para[line_idx] = idx

    # Draw lines (bounding boxes)
    for i, line in enumerate(lines):
        x0, y0, x1, y1 = map(int, line['bbox'])
        color = para_colors[line_to_para.get(i, 0)]
        cv2.rectangle(img, (x0, y0), (x1, y1), color, 2)
        # Optionally, draw line number
        cv2.putText(img, str(i), (x0, y0-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    # Draw edges (connections)
    for (i, j) in edges:
        x0, y0, x1, y1 = map(int, lines[i]['bbox'])
        x0c, y0c = (x0 + x1) // 2, (y0 + y1) // 2
        x2, y2, x3, y3 = map(int, lines[j]['bbox'])
        x1c, y1c = (x2 + x3) // 2, (y2 + y3) // 2
        color = (0, 0, 0)
        cv2.line(img, (x0c, y0c), (x1c, y1c), color, 1)

    # Save and show
    cv2.imwrite(output_path, img)
    plt.figure(figsize=(12, 12))
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.title('Paragraph Reconstruction Visualization')
    plt.show() 