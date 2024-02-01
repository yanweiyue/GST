import torch
import torch.nn as nn

import net as net
from args import parser_loader
import utils
from sklearn.metrics import f1_score
import torch.nn.functional as F
import copy

import warnings
from pruning import GSTScheduler

import dgl
import time

warnings.filterwarnings('ignore')


def run_get_mask(args):
    rewind_weight=None
    device = args['device']
    
    adj, features, labels, idx_train, idx_val, idx_test, degree, learning_type = \
        utils.load_citation(args['dataset'], task_type=args['task_type'])  # adj: csr_matrix
    if args['dataset'] in ['cora','pubmed','citeseer']:
        adj = utils.load_adj_raw(args['dataset'])

    node_num = features.size()[0]
    class_num = labels.numpy().max() + 1
    g = dgl.DGLGraph()
    g.add_nodes(node_num)
    if args['dataset'] in ['cora','pubmed','citeseer']:
        adj = adj.tocoo()
        g.add_edges(adj.row, adj.col)
    else:
        g.add_edges(adj[0,:],adj[1,:])
    
    g = g.to(device)
    features = features.cuda()
    labels = labels.cuda()
    loss_func = nn.CrossEntropyLoss()

    net_gcn = net.net_gcn(embedding_dim=args['embedding_dim'], graph=g, device=device, 
                                spar_wei=args['spar_wei'], spar_adj=args['spar_adj'], num_nodes=features.shape[0],
                                use_bn=args['use_bn'], use_res=args['use_res'],coef=args['coef'],feature_dim = features.shape[1],
                                remain=args["remain"])
    
    net_gcn = net_gcn.to(device)
    
    net_gcn.apply(utils.weight_init)
    
    optimizer = torch.optim.Adam(net_gcn.parameters(), lr=args['lr'], weight_decay=args['weight_decay'])
    
    acc_test = 0.0
    acc_val = 0.0
    best_val_acc = {'val_acc': 0, 'epoch': 0, 'test_acc': 0, "adj_spar": 0, "wei_spar": 0}
    best_mask = None
    mask_ls = []

    rewind_weight = copy.deepcopy(net_gcn.state_dict())
    pruner = lambda:False
    last_acc_val = None
    start_time = time.time()  # start time
    
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
                                   pretrain=args['pretrain_epoch'],ignore_parameters=not args['spar_adj'],beta=args['beta'],warmup_steps=args["warmup_steps"],topo_k=args["topo_k"],sema_k=args["sema_k"],dataset=args["dataset"],backbone='gcn')
                
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
            output = net_gcn(g, features, pretrain=(epoch < args['pretrain_epoch']))
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
            net_gcn.edge_mask.data = pruner.backward_masks[0]  # update the mask
            flag = False
        
        # common loss
        optimizer.zero_grad()   # clear the grad
        output = net_gcn(g, features, pretrain=(epoch < args['pretrain_epoch'])) 
        # score.retain_grad()
        net_gcn.edge_score.retain_grad()
        loss = loss_func(output[idx_train], labels[idx_train])
        add_loss = 0
        add_loss += utils.log_variance_loss(net_gcn.edge_score)*0.1 # Chameleon 0.02
        loss += add_loss
        loss.backward()
        optimizer.step()
        
        with torch.no_grad():
            net_gcn.eval()
            output = net_gcn(g,features, val_test=True, pretrain=(epoch < args['pretrain_epoch']))
            acc_val = f1_score(labels[idx_val].cpu().numpy( # type: ignore
            ), output[idx_val].cpu().numpy().argmax(axis=1), average='micro')
            acc_test = f1_score(labels[idx_test].cpu().numpy(
            ), output[idx_test].cpu().numpy().argmax(axis=1), average='micro')
            acc_train = f1_score(labels[idx_train].cpu().numpy( # type: ignore
            ), output[idx_train].cpu().numpy().argmax(axis=1), average='micro')
            
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
                

            print("Epoch:[{}] L:[{:.3f}] AddL:[{:.3f}] Train:[{:.2f}] Val:[{:.2f}] Test:[{:.2f}] WS:[{:.2f}%] AS:[{:.2f}%] time:[{:.2f}]|"
                  .format(epoch, loss.item(), float(add_loss), acc_train * 100, acc_val * 100, acc_test * 100, wspar_here, aspar_here,time.time()-start_time), end=" ")
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



if __name__ == "__main__":
    args = parser_loader()
    print(args)
    utils.fix_seed(args['seed'])
    best_mask = run_get_mask(args)
    
