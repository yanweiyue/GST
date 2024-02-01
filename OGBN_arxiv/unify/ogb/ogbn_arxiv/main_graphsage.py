import torch
import torch.nn.functional as F
from args import parser_loader
import torch_geometric.transforms as T

import util
import copy
import warnings
from pruning import GSTScheduler

from torch_sparse import SparseTensor
from ogb.nodeproppred import PygNodePropPredDataset,Evaluator
from torch_geometric.utils import to_undirected, add_self_loops
from net import SAGE

import time


warnings.filterwarnings('ignore')

@torch.no_grad()
def test(model, x, edge_index, y_true, split_idx, evaluator):
    model.eval()
    out = model(x, edge_index)
    y_pred = out.argmax(dim=-1, keepdim=True)

    train_acc = evaluator.eval({
        'y_true': y_true[split_idx['train']],
        'y_pred': y_pred[split_idx['train']],
    })['acc']
    valid_acc = evaluator.eval({
        'y_true': y_true[split_idx['valid']],
        'y_pred': y_pred[split_idx['valid']],
    })['acc']
    test_acc = evaluator.eval({
        'y_true': y_true[split_idx['test']],
        'y_pred': y_pred[split_idx['test']],
    })['acc']

    return train_acc, valid_acc, test_acc


def run_get_mask(args):
    rewind_weight=None
    device = args['device']
    
    dataset = PygNodePropPredDataset(name=args["dataset"],transform=T.ToSparseTensor())
    data = dataset[0]
    split_idx = dataset.get_idx_split()
    evaluator = Evaluator(args["dataset"])

    x = data.x.to(device)
    y_true = data.y.to(device)
    train_idx = split_idx['train'].to(device)
    data.adj_t = data.adj_t.to_symmetric()
    row, col, _ = data.adj_t.coo()
    values_ori = torch.ones_like(row).float()
    edge_index = torch.stack((row, col))
    data.adj_t = SparseTensor.from_edge_index(edge_index, values_ori, 
                        sparse_sizes=(data.num_nodes, data.num_nodes)) # .to(device)
    edge_index = edge_index.to(device)
    # print(edge_index.shape)
    args['n_classes'] = data.y.max().item() + 1
    args['n_nodes'] = data.num_nodes
    args['n_edges'] = data.adj_t.nnz()
    args["in_channels"] = data.x.size(-1)
    args["num_tasks"] = dataset.num_classes
    
    out_channels = args['n_classes']
    args["embedding_dim"] = 512
    net_gcn = SAGE(data.x.shape[1], args['embedding_dim'], out_channels, args['num_layers'],args['dropout'],row.shape[0],edge_index, args).to(device)
    optimizer = torch.optim.Adam(net_gcn.parameters(), lr=args['lr'], weight_decay=args['weight_decay'])
   
    acc_test = 0.0
    acc_val = 0.0
    best_val_acc = {'val_acc': 0, 'epoch': 0, 'test_acc': 0, "adj_spar": 0, "wei_spar": 0}
    best_mask = None
    mask_ls = []

    rewind_weight = copy.deepcopy(net_gcn.state_dict())
    pruner = lambda:False
    last_acc_val = None
    start_time = time.time()
    
    # pretrain
    print("========================================================")
    print("pretrain start")
    print("========================================================")   
    
    # record the pivot
    pivot = None
    
    for epoch in range(args['total_epoch']+args['pretrain_epoch']+args['retrain_epoch']):
         # pretrain finished
        if epoch == args['pretrain_epoch']:
            print("========================================================")
            print("pretrain finish")
            print("========================================================")
            net_gcn.load_state_dict(pretrain_state_dict)
            pruner = lambda: True
            best_val_acc = {'val_acc': 0, 'epoch': 0, 'test_acc': 0, "adj_spar": 0, "wei_spar": 0}
            best_mask = None
            if args["remain"] is not None:
                T_end = args["total_epoch"]
                pruner = GSTScheduler(net_gcn, optimizer, remain=args["remain"], alpha=args["alpha"], delta=args["delta"], static_topo=args["static_topo"], T_end=T_end,  accumulation_n=args["accumulation_n"],ignore_linear_layers=not args["spar_wei"],
                                   pretrain=args['pretrain_epoch'],ignore_parameters=not args['spar_adj'],beta=args['beta'],warmup_steps=args["warmup_steps"],topo_k=args["topo_k"],sema_k=args["sema_k"],dataset=args["dataset"],backbone='sage')
                
        # retrain start
        if epoch == args["total_epoch"]+args['pretrain_epoch']:
            print("========================================================")
            print("retrain start")
            print("========================================================")
            with torch.no_grad():
                net_gcn.load_state_dict(rewind_weight)
                net_gcn.edge_mask.data = best_mask[0]
        
        net_gcn.train()
        
       # pivot loss
        if epoch >=args["pretrain_epoch"] and epoch < args["total_epoch"]+args['pretrain_epoch']:
            optimizer.zero_grad()
            output = net_gcn(x, edge_index)[train_idx]
            net_gcn.edge_score.retain_grad()
            pivot_loss = F.kl_div(output.softmax(dim=-1).log(), pivot.softmax(dim=-1),)
            pivot_loss.backward()

        # add the r1 regularization, grow and cut step
        flag = False
        if epoch >= args['pretrain_epoch'] and epoch <args['pretrain_epoch'] + args['total_epoch']:
            if pruner(acc_val):  # grow and cut step
                flag = True
            else: 
                flag = False
        if flag:
            net_gcn.edge_mask.data = pruner.backward_masks[0]  # update the mask(maybe update after the optimizer.step())
            flag = False
        
        # common loss
        optimizer.zero_grad()   # clear the grad
        output = net_gcn(x, edge_index)[train_idx] 
        net_gcn.edge_score.retain_grad()
        loss = F.nll_loss(torch.log_softmax(output, dim=-1), y_true.squeeze(1)[train_idx])
        loss.backward()
        optimizer.step()
       
        result = test(net_gcn, x, edge_index, y_true, split_idx, evaluator)
        acc_train, acc_val, acc_test = result
        
        
        with torch.no_grad():
            if epoch<args['pretrain_epoch']:
                wspar_here,aspar_here = (0,0)
            else:
                wspar_here,aspar_here = pruner.getS()
            
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
                pivot = output.detach()
                pretrain_state_dict = net_gcn.state_dict()
            
            # record the best_val_acc
            if acc_val > best_val_acc['val_acc'] and meet:
                best_val_acc['test_acc'] = acc_test
                best_val_acc['val_acc'] = acc_val
                best_val_acc['epoch'] = epoch
                best_val_acc['wei_spar'] = wspar_here 
                best_val_acc['adj_spar'] = aspar_here
                best_mask = copy.deepcopy(pruner.backward_masks)
                if epoch < args['pretrain_epoch'] + args['total_epoch']:
                    best_val_acc['time'] = time.time() - start_time
                

            print("Epoch:[{}] L:[{:.3f}] Train:[{:.2f}] Val:[{:.2f}] Test:[{:.2f}] WS:[{:.2f}%] AS:[{:.2f}%] time:[{:.2f}]|"
                  .format(epoch, loss.item(), acc_train * 100, acc_val * 100, acc_test * 100, wspar_here, aspar_here,time.time()-start_time), end=" ")
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
    return best_mask


if __name__ == "__main__":
    args = parser_loader()
    print(args)
    util.fix_seed(args['seed'])
    best_mask = run_get_mask(args)
    
    