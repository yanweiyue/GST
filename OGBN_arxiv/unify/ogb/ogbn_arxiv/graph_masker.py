import torch
import torch.nn as nn



class DegreeAwareEdgeScoring(nn.Module):
    def __init__(self, embedding_dim,device=None):
        super(DegreeAwareEdgeScoring, self).__init__()
        self.edge_learner = nn.Sequential(
            nn.Linear(embedding_dim * 2, 128),  # +2
            nn.ReLU(),
            nn.Linear(128, 1)
        )
    

    def forward(self, edge_index, feat):
        row, col = edge_index
        # Create MLP input
        row_embs, col_embs = feat[row], feat[col]
        edge_learner_input = torch.cat([row_embs, col_embs], dim=1)
        # Compute edge weights using MLP
        edge_weight = self.edge_learner(edge_learner_input).squeeze(-1)

        # Make the edge score degree-aware by incorporating normalized adjacency values
        edge_score = edge_weight  # O(N)
        # Normalize the edge weights
        mean = edge_weight.mean()
        variance = edge_weight.var()
        edge_score = ((edge_score - mean) * torch.sqrt(0.0001 / variance)) + 1   #0.0001
        edge_score = edge_score.reshape(-1,1)
        return edge_score

	