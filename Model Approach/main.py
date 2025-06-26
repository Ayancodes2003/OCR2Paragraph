from utils.ocr_parser import preprocess_json_to_lines
from utils.graph_builder import build_graph_from_lines, add_edge_labels
from utils.trainer import Trainer
import numpy as np
import os

def main():
    """
    Main function to run the full paragraph reconstruction pipeline.
    It will train a model if one doesn't exist, otherwise it will load it.
    """
    MODEL_PATH = "trained_gcn.pth"
    OUTPUT_FILE = "reconstructed_paragraphs.txt"

    # --- Step 1 & 2: Pre-processing and Graph Construction ---
    # These steps are always needed to define the graph structure
    print("--- Step 1 & 2: Loading Data and Building Graph ---")
    lines = preprocess_json_to_lines('data/odr_sample.json')
    if not lines:
        print("No lines were detected. Exiting.")
        return
    
    graph = build_graph_from_lines(lines, k=3)
    if graph.number_of_nodes() == 0:
        print("Graph could not be constructed. Exiting.")
        return
    print(f"Graph created with {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges.")

    # --- Step 3: Heuristic Labeling (needed for training) ---
    graph = add_edge_labels(graph)
    
    # --- Step 4: Model Training or Loading ---
    trainer = Trainer(graph, model_path=MODEL_PATH, hidden_dim=128, lr=0.01, epochs=50)

    # Load the model if it exists, otherwise train it
    if not trainer.load_model():
        labels = [data['label'] for _, _, data in graph.edges(data=True)]
        print(f"\n--- Heuristic Labels for Training ---")
        print(f"Positive (same para): {np.sum(labels)} | Negative (new para): {len(labels) - np.sum(labels)}")
        trainer.train()

    # --- Step 5: Predict and Reconstruct Paragraphs ---
    predicted_paragraphs = trainer.predict_paragraphs()
    
    # --- Step 6: Display and Save Results ---
    print(f"\n--- Reconstructed Paragraphs ---")
    print(f"Saving {len(predicted_paragraphs)} paragraphs to {OUTPUT_FILE}...")

    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        for i, para in enumerate(predicted_paragraphs):
            f.write(f"--- Paragraph {i+1} ---\n")
            f.write(para)
            f.write("\n\n")

    print(f"Done. Full output saved to {OUTPUT_FILE}.")
    # Print a small sample to the console
    for i, para in enumerate(predicted_paragraphs[:5]):
        print(f"\nParagraph {i+1}:\n{para}")


if __name__ == "__main__":
    main() 