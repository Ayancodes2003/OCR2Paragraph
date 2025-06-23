# 📘 Project: Paragraph Reconstruction from OCR using GCN

## 🎯 Goal
Given an OCR JSON file with word-level bounding boxes, reconstruct meaningful paragraphs.
This mimics human-style paragraph segmentation using a GCN-based model, as described in:
"Post-OCR Paragraph Recognition by GCN" (Wang et al., WACV 2022).

## 🧱 Architecture Overview
- Graph Neural Network (GCN)
- Nodes = text lines
- Edges = candidate relationships between adjacent lines
- Output = binary edge label (same paragraph or not)

## 📂 Dataset
- OCR JSON with: word text, boundingBox {x, y, width, height}
- Optional: paragraph labels for training (DocBank-style)
- Output should reconstruct grouped paragraph strings

## ✳️ Steps
1.  **`preprocess_json_to_lines(filepath)`**
    - Load OCR JSON
    - Group words into lines using vertical alignment (~±10px Y tolerance)
    - Sort words left to right within a line
    - Return list of lines with: `{"text": ..., "bbox": [...]}`

2.  **`build_graph_from_lines(lines)`**
    - Create node features:
        - Normalized [x0, y0, x1, y1]
        - Line length
        - Avg char width
        - Optional: sentence embeddings (from Sentence-BERT)
    - Connect each node to its k vertical neighbors (k=3)
    - Edge features:
        - Δy (vertical gap)
        - Δx (left indent shift)
        - overlap ratio
        - cosine similarity of embeddings

3.  **`define_gcn_model(input_dims)`**
    - Use PyTorch Geometric to define:
        - `GCNConv` or `SAGEConv` layers
        - Edge classifier head (MLP)
    - Input: node and edge features
    - Output: probability that an edge means "same paragraph"

4.  **`train_gcn_model(graph, edge_labels)`**
    - Binary classification loss: `BCEWithLogitsLoss`
    - Use connected line labels (1=same para, 0=not)
    - Evaluate with precision, recall, F1

5.  **`predict_paragraphs(graph)`**
    - Use predicted positive edges to merge connected components
    - Sort lines by Y and output merged paragraphs

6.  **`render_results(paragraphs, image)`**
    - Optional: visualize predicted paragraphs on top of original image using OpenCV

## 📦 Required Libraries
- `torch`, `torch_geometric`
- `sentence-transformers`
- `scikit-learn`
- `networkx`
- `numpy`, `opencv-python`

## 📂 Directory Structure
```
/data/ocr_sample.json
/models/gcn_model.py
/utils/ocr_parser.py
/utils/graph_builder.py
/utils/trainer.py
/main.py
```
