import torch
import torch.nn as nn
from graph_masker import DegreeAwareEdgeScoring
from functools import partial

from dgl.nn import EdgeWeightNorm

"""
    GIN: Graph Isomorphism Networks
    HOW POWERFUL ARE GRAPH NEURAL NETWORKS? (Keyulu Xu, Weihua Hu, Jure Leskovec and Stefanie Jegelka, ICLR 2019)
    https://arxiv.org/pdf/1810.00826.pdf
"""

from layer_gin import GINLayer, ApplyNodeFunc, MLP
import pdb


class GINNet(nn.Module):
    
    def __init__(self, embedding_dim, graph, device, spar_wei, spar_adj, remain=1, coef=None, mode="prune"):
        super().__init__()

        self.mode = mode
        self.device = device
        self.n_layers = len(embedding_dim) - 1
        
        self.spar_wei = spar_wei
        self.spar_adj = spar_adj
        self.edge_mask_archive = []
        self.coef = coef
        self.remain = remain
        self.graph = graph
        self.num_nodes = graph.num_nodes()
        
        self.norm =  partial(EdgeWeightNorm(norm='right'), graph)
        in_dim = embedding_dim[0]
        hidden_dim = embedding_dim[1]
        n_classes = embedding_dim[-1]
        dropout = 0.5
        # self.n_layers = 2
        self.edge_num = graph.all_edges()[0].numel()
        n_mlp_layers = 1               # GIN
        learn_eps = False              # GIN
        neighbor_aggr_type = 'mean' # GIN
        graph_norm = False    
        batch_norm = False  
        residual = False  
        self.n_classes = n_classes
        
        # List of MLPs
        self.ginlayers = torch.nn.ModuleList()
        
        for layer in range(self.n_layers):
            if layer == 0:
                mlp = MLP(n_mlp_layers, in_dim, hidden_dim, hidden_dim) 
            elif layer == self.n_layers - 1:
                mlp = MLP(n_mlp_layers, hidden_dim, hidden_dim, n_classes)
            else:
                mlp = MLP(n_mlp_layers, hidden_dim, hidden_dim, hidden_dim)
            
            self.ginlayers.append(GINLayer(ApplyNodeFunc(mlp), neighbor_aggr_type,
                                           dropout, graph_norm, batch_norm, residual, 0, learn_eps))

        # Linear function for output of each layer
        # which maps the output of different layers into a prediction score

        self.linears_prediction = nn.Linear(hidden_dim, n_classes, bias=False)
        if spar_adj:
            # use edge_learner cal edge score
            self.edge_learner = DegreeAwareEdgeScoring(in_dim,self.device)
            # self.edge_score = nn.Parameter(torch.zeros(self.edge_num,1),requires_grad=True)
            self.edge_mask = nn.Parameter(torch.ones(self.edge_num,1),requires_grad=False)
        
        self.grads = {}
    
    def forward(self, g, h, snorm_n=0, snorm_e=0, **kwargs):

        g.edata['mask'] = torch.ones(size=(g.number_of_edges(),)).to(h.device)
        if self.spar_adj:
            self.edge_score = self.edge_learner(g,h)
            self.edge_score = self.edge_score*self.edge_mask
            g.edata['mask'] = self.edge_score
            
        hidden_rep = []

        for i in range(self.n_layers):
            h = self.ginlayers[i](g, h, snorm_n)
            hidden_rep.append(h)

        score_over_layer = (self.linears_prediction(hidden_rep[0]) + hidden_rep[1]) / 2

        return score_over_layer
