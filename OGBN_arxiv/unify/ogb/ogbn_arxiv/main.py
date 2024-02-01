from random import shuffle
import torch
import torch.nn.functional as F
from args import parser_loader
from sklearn.metrics import f1_score
import dgl
import util
import copy
import warnings
from pruning import GSTScheduler

from torch_geometric.data import ClusterData, ClusterLoader, NeighborSampler
from torch_geometric.utils import subgraph
from ogb.nodeproppred import PygNodePropPredDataset,Evaluator
from torch_geometric.utils import to_undirected, add_self_loops
import net as net
from torch_geometric.data import DataLoader
import torch.nn as nn
import time
import pruning

warnings.filterwarnings('ignore')

def run_get_mask(args):
    rewind_weight=None
    device = args['device']
    
    dataset = PygNodePropPredDataset(name=args["dataset"])
    data = dataset[0]
    split_idx = dataset.get_idx_split()
    
    evaluator = Evaluator(args["dataset"])

    edge_index = data.edge_index.to(device)
    edge_index = to_undirected(edge_index, num_nodes=data.num_nodes)
    edge_index = add_self_loops(edge_index, num_nodes=data.num_nodes)[0]
    
    args["in_channels"] = data.x.size(-1)
    args["num_tasks"] = dataset.num_classes

    # Convert split indices to boolean masks and add them to `data`.
    for key, idx in split_idx.items():
        mask = torch.zeros(data.num_nodes, dtype=torch.bool)
        mask[idx] = True
        data[f'{key}_mask'] = mask
    
    cluster_data = ClusterData(data, num_parts=args['num_partitions'],
                               recursive=False, save_dir=dataset.processed_dir)
    loader = ClusterLoader(cluster_data, batch_size=args['batch_size'],
                           shuffle=False, num_workers=args['num_workers'])
    
    args['embedding_dim'] = [args['in_channels'],args['hidden_channels'],args['num_tasks']]
    # model
    net_gcn = net.net_gcn(embedding_dim=args['embedding_dim'],  device=device, spar_wei=args['spar_wei'], spar_adj=args['spar_adj'],
                                use_bn=True, use_res=True)
    net_gcn = net_gcn.to(device)
    net_gcn.apply(util.weight_init)
    
    
    # record the subgraph and edge_mask
    sub_graph_list = []
    sub_edgemask_list=[]
    pruner_list = []
    pivot_list = []
    output_list = []
    best_sub_edgemask_list = None
    
    node_num = data.num_nodes
    g_all = dgl.DGLGraph()
    g_all.add_nodes(node_num)
    g_all.add_edges(data.edge_index[0,:],data.edge_index[1,:])
    g_all = g_all.to(device)
    
    for d in loader:
        node_num = d.num_nodes
        g = dgl.DGLGraph()
        g.add_nodes(node_num)
        g.add_edges(d.edge_index[0,:],d.edge_index[1,:])
        g = g.to(device)
        sub_graph_list.append(g)
        sub_edgemask_list.append(nn.Parameter(torch.ones(g.edges()[0].shape[0],1,device=device),requires_grad=False))
        pruner_list.append(lambda: False)
        pivot_list.append(None)
        output_list.append(None)
    
    loss_func = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net_gcn.parameters(), lr=args['lr'], weight_decay=args['weight_decay'])
    
    acc_test = 0.0
    acc_val = 0.0
    best_val_acc = {'val_acc': 0, 'epoch': 0, 'test_acc': 0, "adj_spar": 0, "wei_spar": 0}
    
    rewind_weight = copy.deepcopy(net_gcn.state_dict())
    start_time = time.time()
    
    # pretrain
    print("========================================================")
    print("pretrain start")
    print("========================================================")   

    # record the pivot
    
    for epoch in range(args['total_epoch']+args['pretrain_epoch']+args['retrain_epoch']):
        # pretrain finish
        if epoch == args['pretrain_epoch']:
                print("========================================================")
                print("pretrain finish")
                print("========================================================")
                net_gcn.load_state_dict(pretrain_state_dict)
                pruner_list = [lambda: True for data in loader]
                best_val_acc = {'val_acc': 0, 'epoch': 0, 'test_acc': 0, "adj_spar": 0, "wei_spar": 0}
                if args["remain"] is not None:
                    T_end = args["total_epoch"]
                    pruner_list = [GSTScheduler(net_gcn, optimizer, remain=args["remain"], alpha=args["alpha"], delta=args["delta"], static_topo=args["static_topo"], T_end=T_end,  accumulation_n=args["accumulation_n"],ignore_linear_layers=not args["spar_wei"],
                                    pretrain=args['pretrain_epoch'],ignore_parameters=not args['spar_adj'],beta=args['beta'],warmup_steps=args["warmup_steps"],dataset=args["dataset"],backbone='cluster') for data in enumerate(loader)]
                    
        # retrain start
        if epoch == args["total_epoch"]+args['pretrain_epoch']:
            print("========================================================")
            print("retrain start")
            print("========================================================")
            with torch.no_grad():
                net_gcn.load_state_dict(rewind_weight)
                sub_edgemask_list = best_sub_edgemask_list
        """
        train 
        """ 
        net_gcn.train()
        for i, d in enumerate(loader):     
            d = d.to(device)
            # pivot loss
            if epoch >=args["pretrain_epoch"] and epoch < args["total_epoch"]+args['pretrain_epoch']:
                optimizer.zero_grad()
                output = net_gcn(sub_graph_list[i],d.x ,sub_edgemask_list[i], pretrain=(epoch < args['pretrain_epoch']))
                net_gcn.edge_score.retain_grad()
                pivot_loss = F.kl_div(output.softmax(dim=-1).log(), pivot_list[i].softmax(dim=-1),)
                pivot_loss.backward(retain_graph=True)

            # grow and cut step
            flag = False
            if epoch >= args['pretrain_epoch'] and epoch <args['pretrain_epoch'] + args['total_epoch']:
                if pruner_list[i](acc_val):  # grow and cut step
                    flag = True
                else: 
                    flag = False
            if flag:
                sub_edgemask_list[i] = pruner_list[i].backward_masks[0]  # update the mask(maybe update after the optimizer.step())
                flag = False
                
            # common loss
            optimizer.zero_grad()   # clear the grad
            output = net_gcn(sub_graph_list[i],d.x,sub_edgemask_list[i], pretrain=(epoch < args['pretrain_epoch'])) 
            net_gcn.edge_score.retain_grad()
            y = d.y.squeeze(1)
            loss = loss_func(output, y)
            loss.backward()
            optimizer.step()
            output_list[i] = output
        """
        test
        """

        with torch.no_grad():
            net_gcn.eval()
            if epoch<args["pretrain_epoch"]:
                remain = 1
            else:
                remain = args["remain"]
            edge_mask = get_edge_mask(net_gcn,data.edge_index.to(net_gcn.device),data.x.to(net_gcn.device),args['remain'])
            output = net_gcn(g_all,data.x.to(net_gcn.device),edge_mask, pretrain=(epoch < args['pretrain_epoch']), val_test=True)
            y_pred = output.argmax(dim=-1, keepdim=True)
            acc_train = evaluator.eval({
            'y_true': data.y[split_idx['train']],
            'y_pred': y_pred[split_idx['train']],
            })['acc']
            acc_val = evaluator.eval({
            'y_true': data.y[split_idx['valid']],
            'y_pred': y_pred[split_idx['valid']],
            })['acc']
            acc_test = evaluator.eval({
            'y_true': data.y[split_idx['test']],
            'y_pred': y_pred[split_idx['test']],
            })['acc']

            if epoch<args['pretrain_epoch']:
                wspar_here,aspar_here = (0,0)
            else:
                wspar_here,aspar_here = pruner_list[i].getS()
            
            meet = ((args['spar_adj'] and aspar_here >= ((1-args["remain"])*100-1)) or not args['spar_adj']) and \
                    ((args['spar_wei'] and wspar_here >= ((1-args["remain"])*100-1)) or not args['spar_wei']) and \
                        epoch >= args['pretrain_epoch']

            # record the pivot
            if acc_val > best_val_acc['val_acc'] and epoch < args['pretrain_epoch']:
                best_val_acc['test_acc'] = acc_test
                best_val_acc['val_acc'] = acc_val
                best_val_acc['epoch'] = epoch
                best_val_acc['wei_spar'] = wspar_here 
                best_val_acc['adj_spar'] = aspar_here
                pivot_list = output_list
                pretrain_state_dict = net_gcn.state_dict()
            
            # record the best_val_acc
            if acc_val > best_val_acc['val_acc'] and meet:
                best_val_acc['test_acc'] = acc_test
                best_val_acc['val_acc'] = acc_val
                best_val_acc['epoch'] = epoch
                best_val_acc['wei_spar'] = wspar_here 
                best_val_acc['adj_spar'] = aspar_here
                best_mask = copy.deepcopy(pruner_list[i].backward_masks)
                if epoch < args['pretrain_epoch'] + args['total_epoch']:
                    best_val_acc['time'] = time.time() - start_time
                

            print("Epoch:[{}] L:[{:.3f}] Train:[{:.2f}]  Val:[{:.2f}] Test:[{:.2f}] WS:[{:.2f}%] AS:[{:.2f}%] time:[{:.2f}]|"
                  .format(epoch, loss.item(),acc_train*100, acc_val * 100, acc_test * 100, wspar_here, aspar_here,time.time()-start_time), end=" ")
            if meet:
                print("Best Val:[{:.2f}] Test:[{:.2f}] AS:[{:.2f}%] WS:[{:.2f}%] at Epoch:[{}] time:[{}]"
                      .format(
                          best_val_acc['val_acc'] * 100,
                          best_val_acc['test_acc'] * 100,
                          best_val_acc['adj_spar'],
                          best_val_acc['wei_spar'],
                          best_val_acc['epoch'],
                          round(best_val_acc['time'],2)))
            else:
                print("")
                
    end_time = time.time()
    print("total time:",round(end_time-start_time,2))
            
    return best_mask


def get_edge_mask(net_gcn,edge_index,features,remain):
    edge_score = net_gcn.edge_learner(edge_index,features)
    n_total = edge_score.shape[0]  # the total num of the weights
    s = int((1-remain) * n_total)  # the sparisity num of the weights
    n_keep = n_total - s
    score_drop = torch.abs(edge_score)
    # create drop mask
    _, sorted_indices = torch.topk(score_drop.view(-1), k=n_total)
    new_values = torch.where(
                torch.arange(n_total, device=edge_score.device) < n_keep,
                torch.ones_like(sorted_indices),
                torch.zeros_like(sorted_indices))
    flat_mask = new_values.scatter(0, sorted_indices, new_values)  #new_values[sorted_indices[i]]=new_values[i]
            
    mask = torch.reshape(flat_mask,edge_score.shape)
    mask = mask.bool()    
    return mask

if __name__ == "__main__":
    args = parser_loader()
    print(args)
    util.fix_seed(args['seed'])
    best_mask = run_get_mask(args)
    
