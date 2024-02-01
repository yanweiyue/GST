import torch
import torch.nn as nn
from graph_masker import DegreeAwareEdgeScoring

import dgl.function as fn
from layer_gcn import ApplyNodeFunc,GCNLayer


msg_mask = fn.src_mul_edge(src='h',edge='mask', out='m')
gcn_reduce = fn.sum(msg='m', out='h')

class net_gcn(nn.Module):
    def __init__(self, embedding_dim, graph, device, spar_wei, spar_adj, num_nodes, use_res, use_bn, feature_dim,remain=1,coef=None, mode="prune"):
        super().__init__()

        self.mode = mode
        self.n_layers = len(embedding_dim) - 1
        
        self.spar_wei = spar_wei
        self.spar_adj = spar_adj
        self.edge_mask_archive = []
        self.coef = coef
        
        self.use_bn = use_bn
        self.use_res = use_res
        
        self.step = 0
        
        in_dim = embedding_dim[0]
        hidden_dim = embedding_dim[1]
        n_classes = embedding_dim[-1]
        self.n_classes = n_classes
        self.graph = graph
        
        self.edge_num = graph.all_edges()[0].numel()
        self.num_nodes = num_nodes
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(0.5)

        self.device = device
        
        self.gcnlayers = torch.nn.ModuleList()
        self.remain = remain
        
        
        for layer in range(self.n_layers):
            if layer == 0:
                nodeFunc = ApplyNodeFunc(in_dim, hidden_dim)
                self.gcnlayers.append(GCNLayer(nodeFunc,self.dropout,device=self.device,  residual=self.use_res,activate=self.relu))
            elif layer == self.n_layers - 1:
                nodeFunc = ApplyNodeFunc(hidden_dim, n_classes)
                self.gcnlayers.append(GCNLayer(nodeFunc,self.dropout,device=self.device, residual=self.use_res))# non-activate
            else:
                nodeFunc = ApplyNodeFunc(hidden_dim, hidden_dim)
                self.gcnlayers.append(GCNLayer(nodeFunc,self.dropout, device=self.device,residual=self.use_res,activate=self.relu))

        if spar_adj:
            # use edge_learner cal edge score
            self.edge_learner = DegreeAwareEdgeScoring(in_dim,self.device)
            self.edge_mask = nn.Parameter(torch.ones(self.edge_num,1),requires_grad=False)
        
    def forward(self,g, h, **kwargs):
        g.edata['mask'] = torch.ones(size=(g.number_of_edges(),)).to(h.device)
        if self.spar_adj:
            self.edge_score = self.edge_learner(g,h)
            self.edge_score = self.edge_score*self.edge_mask
            g.edata['mask'] = self.edge_score
        for i in range(self.n_layers):
            h = self.gcnlayers[i](g, h)
        return h


   