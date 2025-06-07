"""
UTSP: Unsupervised Learning for Solving the Travelling Salesman Problem
Implementation based on arXiv:2303.10538v2

Requirements:
- torch
- torch_geometric
- numpy

To install requirements:
    pip install torch torch_geometric numpy

This script demonstrates:
- Generating random TSP instances
- Building a simple GNN (GraphConv) for edge heatmap prediction
- Implementing the surrogate loss function (Eq. 2 in the paper)
- A basic training loop

Note: This is a minimal, educational implementation. For full performance, use the SAG layer as in the paper.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv

# --- Generate random TSP instance ---
def generate_tsp_instance(n_cities, seed=None):
    if seed is not None:
        np.random.seed(seed)
    coords = np.random.rand(n_cities, 2)
    return coords

def compute_distance_matrix(coords):
    n = coords.shape[0]
    D = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=-1)
    return D

def build_adjacency_matrix(D, tau=0.1):
    W = np.exp(-D / tau)
    return W

# --- GNN Model ---
class SimpleGNN(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_layers=2):
        super().__init__()
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(in_dim, hidden_dim))
        for _ in range(n_layers-2):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
        self.convs.append(GCNConv(hidden_dim, hidden_dim))
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.edge_proj = nn.Linear(2*hidden_dim, 1)

    def forward(self, x, edge_index):
        for conv in self.convs:
            x = F.relu(conv(x, edge_index))
        x = self.out_proj(x)
        # Build edge features for all pairs
        n = x.size(0)
        x_i = x.unsqueeze(1).expand(-1, n, -1)
        x_j = x.unsqueeze(0).expand(n, -1, -1)
        edge_feat = torch.cat([x_i, x_j], dim=-1)
        edge_logits = self.edge_proj(edge_feat).squeeze(-1)
        return edge_logits

# --- Surrogate Loss (Eq. 2) ---
def surrogate_loss(T, H, D, lambda1=1.0, lambda2=1.0):
    n = T.size(0)
    # Row sum constraint (each row sums to 1)
    row_sum = torch.sum(T, dim=1)
    loss_row = ((row_sum - 1) ** 2).sum()
    # No self-loops (main diagonal of H)
    loss_diag = torch.diagonal(H).sum()
    # Minimize expected TSP length
    loss_dist = (torch.from_numpy(D).to(T.device) * H).sum()
    return lambda1 * loss_row + lambda2 * loss_diag + loss_dist

# --- T -> H transformation (Eq. 1) ---
def t_to_h(T):
    n = T.size(0)
    H = torch.zeros_like(T)
    for t in range(n-1):
        pt = T[:, t].unsqueeze(1)  # (n,1)
        pt1 = T[:, t+1].unsqueeze(0)  # (1,n)
        H += pt @ pt1
    # Last term: pn * p1^T
    pn = T[:, -1].unsqueeze(1)
    p1 = T[:, 0].unsqueeze(0)
    H += pn @ p1
    return H

# --- Training Loop ---
def train_utsp(n_cities=20, n_epochs=100, hidden_dim=64, lr=1e-3, tau=0.1, seed=42):
    coords = generate_tsp_instance(n_cities, seed)
    D = compute_distance_matrix(coords)
    W = build_adjacency_matrix(D, tau)
    # Build PyG Data object
    edge_index = torch.tensor(np.array(np.meshgrid(np.arange(n_cities), np.arange(n_cities))).reshape(2, -1), dtype=torch.long)
    x = torch.from_numpy(coords).float()
    data = Data(x=x, edge_index=edge_index)
    # Model
    model = SimpleGNN(in_dim=2, hidden_dim=hidden_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for epoch in range(n_epochs):
        model.train()
        optimizer.zero_grad()
        edge_logits = model(data.x, data.edge_index)
        # Reshape to (n, n)
        T = edge_logits.view(n_cities, n_cities)
        # Column-wise softmax
        T = F.softmax(T, dim=0)
        H = t_to_h(T)
        loss = surrogate_loss(T, H, D)
        loss.backward()
        optimizer.step()
        if (epoch+1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:3d} | Loss: {loss.item():.4f}")
    print("Training complete.")
    return model, coords, D

if __name__ == "__main__":
    # Example usage
    model, coords, D = train_utsp(n_cities=20, n_epochs=50)
    print("Done.")
