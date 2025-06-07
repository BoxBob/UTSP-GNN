import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv

class GINGNN(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_layers=2):
        super().__init__()
        self.gins = nn.ModuleList()
        nn1 = nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim))
        self.gins.append(GINConv(nn1))
        for _ in range(n_layers-1):
            nnk = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim))
            self.gins.append(GINConv(nnk))
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.edge_proj = nn.Linear(2*hidden_dim, 1)

    def forward(self, x, edge_index):
        for gin in self.gins:
            x = F.relu(gin(x, edge_index))
        x = self.out_proj(x)
        n = x.size(0)
        x_i = x.unsqueeze(1).expand(-1, n, -1)
        x_j = x.unsqueeze(0).expand(n, -1, -1)
        edge_feat = torch.cat([x_i, x_j], dim=-1)
        edge_logits = self.edge_proj(edge_feat).squeeze(-1)
        return edge_logits
