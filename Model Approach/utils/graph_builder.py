import numpy as np
import networkx as nx
from sklearn.preprocessing import normalize
from scipy.spatial.distance import cdist
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer
import torch

# Load a pre-trained sentence-transformer model.
# Using a multilingual model as the sample data contains non-English text.
# This will be loaded only once when the module is imported.
try:
    print("Loading pre-trained sentence-transformer model...")
    model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    print("Model loaded successfully.")
except Exception as e:
    print(f"Failed to load sentence-transformer model: {e}")
    model = None

def build_graph_from_lines(lines: List[Dict[str, Any]], k: int = 3) -> nx.Graph:
    """
    Builds a graph from lines, enriching it with sentence embeddings.

    Args:
        lines (List[Dict[str, Any]]): A list of line objects.
        k (int): The number of vertical neighbors to connect.

    Returns:
        nx.Graph: A NetworkX graph with rich node and edge features.
    """
    if not lines or model is None:
        if model is None:
            print("Cannot build graph because sentence-transformer model is not available.")
        return nx.Graph()

    # --- 1. Generate Sentence Embeddings ---
    all_texts = [line['text'] for line in lines]
    print("Generating sentence embeddings for all lines...")
    with torch.no_grad():
        embeddings = model.encode(all_texts, convert_to_tensor=True, show_progress_bar=True)
    print("Embeddings generated.")

    # --- 2. Create Node Features ---
    node_features = []
    page_width = max(line['bbox'][2] for line in lines) if lines else 1
    page_height = max(line['bbox'][3] for line in lines) if lines else 1

    for i, line in enumerate(lines):
        bbox = line['bbox']
        x0, y0, x1, y1 = bbox
        
        norm_bbox = [x0/page_width, y0/page_height, x1/page_width, y1/page_height]
        line_length = len(line['text'])
        avg_char_width = (x1 - x0) / line_length if line_length > 0 else 0
        
        # Combine geometric features with the sentence embedding
        geo_features = np.array([line_length, avg_char_width] + norm_bbox)
        full_features = np.concatenate([geo_features, embeddings[i].cpu().numpy()])
        
        node_features.append((i, {
            "features": full_features,
            "text": line['text'],
            "bbox": bbox,
            "embedding": embeddings[i]
        }))

    # --- 3. Create Graph and Add Nodes ---
    G = nx.Graph()
    for i, attrs in node_features:
        G.add_node(i, **attrs)

    if G.number_of_nodes() == 0:
        return G

    # --- 4. Connect Nodes and Create Edge Features ---
    node_positions = np.array([line['bbox'] for line in lines])
    y_centers = (node_positions[:, 1] + node_positions[:, 3]) / 2
    
    for i in range(len(lines)):
        current_y = y_centers[i]
        below_indices = [j for j, y in enumerate(y_centers) if y > current_y]
        
        if not below_indices:
            continue
            
        distances = cdist(np.array([[0, y_centers[i]]]), np.array([[0, y_centers[j]] for j in below_indices]))
        nearest_indices_local = np.argsort(distances[0])[:k]
        nearest_indices_global = [below_indices[j] for j in nearest_indices_local]

        for neighbor_idx in nearest_indices_global:
            if i == neighbor_idx: continue

            bbox_i = G.nodes[i]['bbox']
            bbox_j = G.nodes[neighbor_idx]['bbox']
            
            delta_y = bbox_j[1] - bbox_i[3]
            delta_x = bbox_j[0] - bbox_i[0]
            
            overlap_x = max(0, min(bbox_i[2], bbox_j[2]) - max(bbox_i[0], bbox_j[0]))
            width_i = bbox_i[2] - bbox_i[0]
            overlap_ratio = overlap_x / width_i if width_i > 0 else 0
            
            # Calculate cosine similarity between sentence embeddings
            emb_i = G.nodes[i]['embedding']
            emb_j = G.nodes[neighbor_idx]['embedding']
            cosine_sim = torch.nn.functional.cosine_similarity(emb_i.unsqueeze(0), emb_j.unsqueeze(0)).item()
            
            edge_features = [delta_y, delta_x, overlap_ratio, cosine_sim]
            
            G.add_edge(i, neighbor_idx, features=edge_features)
            
    # Clean up embeddings from node attributes to save memory
    for i in G.nodes():
        del G.nodes[i]['embedding']
            
    return G

def add_edge_labels(G: nx.Graph) -> nx.Graph:
    """
    Adds binary labels to edges based on refined heuristics to reduce over-segmentation.

    Heuristic: An edge is a paragraph break (0) only if there are strong signals,
    such as a large vertical gap or a significant indent on dissimilar text.
    Otherwise, it's the same paragraph (1).

    Args:
        G (nx.Graph): The input graph.

    Returns:
        nx.Graph: The graph with updated edge labels.
    """
    if G.number_of_nodes() == 0:
        return G

    line_heights = [node['bbox'][3] - node['bbox'][1] for _, node in G.nodes(data=True)]
    avg_line_height = np.mean(line_heights) if line_heights else 20

    # More conservative thresholds to prevent over-segmentation
    y_break_threshold = avg_line_height * 2.0  # Increased from 1.5
    indent_threshold = avg_line_height * 0.5
    similarity_threshold = 0.6  # Don't break if text is semantically similar

    for u, v, data in G.edges(data=True):
        if 'features' in data:
            delta_y, delta_x, _, cosine_sim = data['features']
            
            # Default to same-paragraph (1) and look for strong break signals (0)
            label = 1
            
            # Large vertical gap is a strong signal for a new paragraph
            if delta_y > y_break_threshold:
                label = 0
            
            # An indent is a signal ONLY IF the content is not similar
            elif delta_x > indent_threshold and cosine_sim < similarity_threshold:
                label = 0
            
            data['label'] = label
            
    return G 