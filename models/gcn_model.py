import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, GCNConv

class EdgeGCN(nn.Module):
    """
    A Graph Convolutional Network for edge classification.

    The model uses SAGEConv layers to learn node embeddings and then uses a
    multi-layer perceptron (MLP) to classify edges based on the embeddings
    of their incident nodes and the edge's own features.
    """
    def __init__(self, node_input_dim: int, edge_input_dim: int, hidden_dim: int, output_dim: int = 1):
        """
        Initializes the EdgeGCN model.

        Args:
            node_input_dim (int): The dimensionality of the input node features.
            edge_input_dim (int): The dimensionality of the input edge features.
            hidden_dim (int): The dimensionality of the hidden GCN layers.
            output_dim (int): The dimensionality of the output (1 for binary classification).
        """
        super(EdgeGCN, self).__init__()
        
        # GNN layers to process the graph structure and node features
        self.conv1 = SAGEConv(node_input_dim, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, hidden_dim)

        # MLP head to classify edges
        # The input to the MLP is the concatenation of:
        # - Source node embedding (hidden_dim)
        # - Target node embedding (hidden_dim)
        # - Edge features (edge_input_dim)
        self.edge_classifier = nn.Sequential(
            nn.Linear((hidden_dim * 2) + edge_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x, edge_index, edge_attr):
        """
        Defines the forward pass of the model.

        Args:
            x (torch.Tensor): Node features of shape [num_nodes, node_input_dim].
            edge_index (torch.Tensor): Graph connectivity in COO format with shape [2, num_edges].
            edge_attr (torch.Tensor): Edge features of shape [num_edges, edge_input_dim].

        Returns:
            torch.Tensor: The output logits for each edge, shape [num_edges, 1].
        """
        # 1. Obtain node embeddings using GNN layers
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        # x now contains the learned embeddings for each node (line)

        # 2. Prepare for edge classification
        source_nodes, target_nodes = edge_index
        
        # Get the embeddings of the source and target nodes for each edge
        source_embed = x[source_nodes]
        target_embed = x[target_nodes]
        
        # 3. Concatenate node embeddings and edge features
        combined_features = torch.cat([source_embed, target_embed, edge_attr], dim=1)
        
        # 4. Pass through the edge classifier to get logits
        edge_logits = self.edge_classifier(combined_features)
        
        return edge_logits 