import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv

class GATGNN(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_layers=2, heads=4):
        super().__init__()
        self.gats = nn.ModuleList()
        self.gats.append(GATConv(in_dim, hidden_dim, heads=heads, concat=True))
        for _ in range(n_layers-2):
            self.gats.append(GATConv(hidden_dim*heads, hidden_dim, heads=heads, concat=True))
        self.gats.append(GATConv(hidden_dim*heads, hidden_dim, heads=1, concat=True))
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.edge_proj = nn.Linear(2*hidden_dim, 1)

    def forward(self, x, edge_index):
        for gat in self.gats:
            x = F.elu(gat(x, edge_index))
        x = self.out_proj(x)
        n = x.size(0)
        x_i = x.unsqueeze(1).expand(-1, n, -1)
        x_j = x.unsqueeze(0).expand(n, -1, -1)
        edge_feat = torch.cat([x_i, x_j], dim=-1)
        edge_logits = self.edge_proj(edge_feat).squeeze(-1)
        return edge_logits
