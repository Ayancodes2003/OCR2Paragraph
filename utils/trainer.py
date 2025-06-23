import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
import networkx as nx
from torch_geometric.data import Data
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support
from typing import List
import os

from models.gcn_model import EdgeGCN

def networkx_to_pyg_data(G: nx.Graph) -> Data:
    """Converts a NetworkX graph to a PyTorch Geometric Data object."""
    # Node features
    node_features = [data['features'] for _, data in G.nodes(data=True)]
    x = torch.tensor(node_features, dtype=torch.float)

    # Edge index and features
    edge_list = list(G.edges())
    edge_index = torch.tensor(list(zip(*edge_list)), dtype=torch.long)
    
    edge_features = [G.edges[edge]['features'] for edge in edge_list]
    edge_attr = torch.tensor(edge_features, dtype=torch.float)
    
    # Edge labels
    labels = [G.edges[edge]['label'] for edge in edge_list]
    y = torch.tensor(labels, dtype=torch.float).unsqueeze(1)
    
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)


class Trainer:
    """
    Handles the training, evaluation, and prediction pipeline for the EdgeGCN model.
    """
    def __init__(self, graph: nx.Graph, model_path: str, hidden_dim: int = 64, lr: float = 0.01, epochs: int = 50):
        """
        Args:
            graph (nx.Graph): The input graph with node/edge features and labels.
            model_path (str): Path to save/load the trained model.
            hidden_dim (int): The hidden dimension size for the GCN model.
            lr (float): Learning rate for the optimizer.
            epochs (int): Number of training epochs.
        """
        self.graph = graph
        self.epochs = epochs
        self.model_path = model_path
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

        self._prepare_data()

        # Define model
        self.model = EdgeGCN(
            node_input_dim=self.data.x.shape[1],
            edge_input_dim=self.data.edge_attr.shape[1],
            hidden_dim=hidden_dim
        ).to(self.device)

        # Define optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        
        # Define loss with weighting for class imbalance
        if (self.edge_labels == 1).sum() > 0:
            pos_weight_value = (self.edge_labels == 0).sum() / (self.edge_labels == 1).sum()
        else:
            pos_weight_value = 1.0 # Default if no positive samples
        self.criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight_value, device=self.device))

    def _prepare_data(self):
        """Converts the NetworkX graph to a PyTorch Geometric Data object."""
        self.data = Data()
        # Sort nodes to ensure consistent order
        nodes = sorted(self.graph.nodes())
        node_map = {node: i for i, node in enumerate(nodes)}

        # Extract node features
        x = np.array([self.graph.nodes[n]['features'] for n in nodes])
        self.data.x = torch.tensor(x, dtype=torch.float32).to(self.device)

        # Extract edge indices and features
        edge_indices, edge_attrs, edge_labels = [], [], []
        for u, v, attrs in self.graph.edges(data=True):
            edge_indices.append([node_map[u], node_map[v]])
            edge_attrs.append(attrs['features'])
            edge_labels.append(attrs['label'])
        
        self.data.edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous().to(self.device)
        self.data.edge_attr = torch.tensor(edge_attrs, dtype=torch.float32).to(self.device)
        self.edge_labels = torch.tensor(edge_labels, dtype=torch.float32).unsqueeze(1).to(self.device)
        self.data.y = self.edge_labels


    def train(self):
        """Runs the training loop and saves the model."""
        print("\n--- Starting Model Training ---")
        for epoch in range(self.epochs):
            self.model.train()
            self.optimizer.zero_grad()
            
            logits = self.model(self.data.x, self.data.edge_index, self.data.edge_attr)
            loss = self.criterion(logits, self.data.y)
            
            loss.backward()
            self.optimizer.step()
            
            if (epoch + 1) % 10 == 0:
                self.evaluate(epoch, loss)
        
        self.save_model()

    def save_model(self):
        """Saves the trained model's state dictionary."""
        print(f"\n--- Saving model to {self.model_path} ---")
        torch.save(self.model.state_dict(), self.model_path)
        print("Model saved successfully.")

    def load_model(self) -> bool:
        """Loads a pre-trained model state dictionary if it exists."""
        if os.path.exists(self.model_path):
            print(f"--- Loading pre-trained model from {self.model_path} ---")
            try:
                self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
                self.model.to(self.device)
                print("Model loaded successfully.")
                return True
            except Exception as e:
                print(f"Error loading model: {e}. Training from scratch.")
                return False
        return False

    def evaluate(self, epoch: int, loss: torch.Tensor):
        """Evaluates the model and prints performance metrics."""
        self.model.eval()
        with torch.no_grad():
            logits = self.model(self.data.x, self.data.edge_index, self.data.edge_attr)
            preds = (torch.sigmoid(logits) > 0.5).cpu().numpy()
            true_labels = self.data.y.cpu().numpy()
            
            precision, recall, f1, _ = precision_recall_fscore_support(
                true_labels, preds, average='binary', zero_division=0
            )
            
            print(f"Epoch {epoch+1:02d} | Loss: {loss.item():.4f} | "
                  f"Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f}")

    def predict_paragraphs(self) -> List[str]:
        """
        Uses the trained model to predict paragraphs.

        Returns:
            List[str]: A list of strings, where each string is a reconstructed paragraph.
        """
        print("\n--- Predicting Paragraphs ---")
        self.model.eval()
        with torch.no_grad():
            logits = self.model(self.data.x, self.data.edge_index, self.data.edge_attr)
            predictions = torch.sigmoid(logits) > 0.5

        # Create a new graph with only the edges predicted as "same-paragraph"
        predicted_graph = nx.Graph()
        edge_list = self.data.edge_index.t().cpu().numpy()

        for i, (u, v) in enumerate(edge_list):
            if predictions[i]:
                predicted_graph.add_edge(u, v)
        
        # Find connected components, which represent paragraphs
        components = list(nx.connected_components(predicted_graph))
        
        # Sort lines within each paragraph component by their original y-coordinate
        paragraphs = []
        original_nodes = sorted(self.graph.nodes())

        for component in components:
            component_nodes = [original_nodes[i] for i in component]
            # Sort nodes by vertical position (y0 of the bbox)
            component_nodes.sort(key=lambda n: self.graph.nodes[n]['bbox'][1])
            
            # Join the text of the lines to form a paragraph
            para_text = " ".join([self.graph.nodes[n]['text'] for n in component_nodes])
            paragraphs.append(para_text)
            
        # Sort the final paragraphs by the y-position of their first line
        paragraphs.sort(key=lambda p: [
            self.graph.nodes[n]['bbox'][1] 
            for n in original_nodes 
            if self.graph.nodes[n]['text'] in p
        ][0])

        return paragraphs 