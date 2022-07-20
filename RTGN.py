from torch import nn
import torch
from torch_geometric.data import Data, Batch
import torch_geometric.nn as gnn

import numpy as np

from typing import List, Tuple, Dict

from graph_components import GAT

class RTGNGat(torch.nn.Module):
    def __init__(self, action_dim, hidden_dim, node_dim):
        super().__init__()
        self.gat = GAT(hidden_dim=hidden_dim, node_dim=node_dim)
        self.set2set = gnn.Set2Set(hidden_dim, processing_steps=6)
        self.mlp = nn.Sequential(
            nn.Linear(2*hidden_dim, hidden_dim), 
            nn.ReLU(), 
            nn.Linear(hidden_dim, hidden_dim), 
            nn.ReLU(), 
            nn.Linear(hidden_dim, 1)
            )

    def forward(self, obs):
        data, nonring, nrbidx, torsion_list_sizes = obs
        N = data.num_graphs

        out = self.gat(data)
        pool = self.set2set(out, data.batch)
        v = self.mlp(pool)

        return v