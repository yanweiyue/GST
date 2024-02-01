import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn
import pdb
from dgl.nn.pytorch import GraphConv
from typing import Optional
import math
import torch as th
from torch.nn import init


from torch import Tensor, dropout
from torch.nn import Parameter

"""
    GIN: Graph Isomorphism Networks
    HOW POWERFUL ARE GRAPH NEURAL NETWORKS? (Keyulu Xu, Weihua Hu, Jure Leskovec and Stefanie Jegelka, ICLR 2019)
    https://arxiv.org/pdf/1810.00826.pdf
"""
msg_mask = fn.src_mul_edge('h', 'mask', 'm')
gcn_reduce = fn.sum(msg='m',out='h')

class GCNLayer(nn.Module):
    """
    [!] code adapted from dgl implementation of GINConv

    Parameters
    ----------
    apply_func : callable activation function/layer or None
        If not None, apply this function to the updated node feature,
        the :math:`f_\Theta` in the formula.
    out_dim :
        Rquired for batch norm layer; should match out_dim of apply_func if not None.
    dropout :
        Required for dropout of output features.
    graph_norm : 
        boolean flag for output features normalization w.r.t. graph sizes.
    batch_norm :
        boolean flag for batch_norm layer.
    residual :
        boolean flag for using residual connection.
    """
    def __init__(self, apply_func, dropout,device,residual=False,activate = None):
        super().__init__()
        self.apply_func = apply_func
        self.device =device
        self.residual = residual
        self.activate = activate
        self.dropout = dropout
        in_dim = apply_func.in_feats
        out_dim = apply_func.out_feats
        
        if in_dim != out_dim:
            self.residual = False
            
        self.bn_node_h = nn.BatchNorm1d(out_dim)


    def forward(self, g, h,**kwargs):
        g = g.local_var()
        # norm
        degs = g.out_degrees().to(h.device).float().clamp(min=1)
        norm = torch.pow(degs, -0.5)
        norm.masked_fill_(norm == float('inf'), 0.)
        shp = norm.shape + (1,) * (h.dim() - 1)
        norm = torch.reshape(norm, shp)
        h = h * norm
        
        h_in = h # for residual connection
        g.ndata['h'] = h

        ### pruning edges by cutting message passing process
        g.update_all(msg_mask, gcn_reduce)
        if self.apply_func is not None:
            g.apply_nodes(func=self.apply_func)
            h = g.ndata.pop('h')
        
        # norm
        degs = g.in_degrees().to(h.device).float().clamp(min=1)
        norm = torch.pow(degs, -0.5)
        norm.masked_fill_(norm == float('inf'), 0.)
        shp = norm.shape + (1,) * (h.dim() - 1)
        norm = torch.reshape(norm, shp)
        h = h * norm
        
        if self.dropout:
            h = self.dropout(h)
        
        if self.activate:
            h = self.activate(h) # non-linear activation
        
        if self.residual:
            h = h_in + h # residual connection
        
        return h
    

class ApplyNodeFunc(nn.Module):
    """
        This class is used in class GINNet
        Update the node feature hv with MLP
    """
    def __init__(self, in_feats, out_feats):
        super(ApplyNodeFunc, self).__init__()
        self.in_feats = in_feats
        self.out_feats = out_feats
        self.linear = nn.Linear(in_feats, out_feats)
 
    def forward(self, node):
        h = self.linear(node.data['h'])
        return {'h':h}
