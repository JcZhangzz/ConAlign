import os
import time
import numpy as np
import random

import torch
import torch.nn as nn

from model import *

import arguments

import utils.load_dataset_new as load_dataset_custom
import utils.load_kuairand as load_kuairand
import utils.data_loader
import utils.metrics
from utils.early_stop import EarlyStopping, Stop_args

def setup_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available(): 
        torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def para(args): 
    if args.dataset == 'yahooR3': 
        args.training_args = {
            'batch_size': 1024,
            'epochs': 500,
            'patience': 60,
            'block_batch': [6000, 500]
        }
        args.base_model_args = {
            'emb_dim': 10,
            'learning_rate': 0.001,
            'weight_decay': 0
        }
        args.teacher_model_args = {
            'emb_dim': 10,
            'learning_rate': 0.1,
            'weight_decay': 12
        }
    elif args.dataset == 'coat':
        args.training_args = {
            'batch_size': 128,
            'epochs': 500,
            'patience': 60,
            'block_batch': [64, 64]
        }
        args.base_model_args = {
            'emb_dim': 10,
            'learning_rate': 0.001,
            'weight_decay': 0.1
        }
        args.teacher_model_args = {
            'emb_dim': 10,
            'learning_rate': 0.1,
            'weight_decay': 10
        }
    elif 'KuaiRand' in args.dataset:
        args.training_args = {
            'batch_size': 1024,
            'epochs': 600,
            'patience': 80,
            'block_batch': [500, 10000]
        }
        args.base_model_args = {
            'emb_dim': 10,
            'learning_rate': 0.01,
            'weight_decay': 0.001
        }
        args.teacher_model_args = {
            'emb_dim': 10,
            'learning_rate': 0.01,
            'weight_decay': 100
        }
                

def train_and_eval(train_data, biased_val_data, unif_train_data, unif_val_data, 
                   test_data, biased_test_data, args, device='cuda'):
    train_loader = utils.data_loader.Block(
        train_data,
        u_batch_size=args.training_args['block_batch'][0],
        i_batch_size=args.training_args['block_batch'][1],
        device=device
    )
    unif_loader = utils.data_loader.Block(
        unif_train_data,
        u_batch_size=args.training_args['block_batch'][0],
        i_batch_size=args.training_args['block_batch'][1],
        device=device
    )
    biased_val_loader = utils.data_loader.DataLoader(
        utils.data_loader.Interactions(biased_val_data),
        batch_size=args.training_args['batch_size']
    )
    unbiased_val_loader = utils.data_loader.DataLoader(
        utils.data_loader.Interactions(unif_val_data),
        batch_size=args.training_args['batch_size']
    )
    test_loader = utils.data_loader.DataLoader(
        utils.data_loader.Interactions(test_data),
        batch_size=args.training_args['batch_size']
    )
    biased_test_loader = utils.data_loader.DataLoader(
        utils.data_loader.Interactions(biased_test_data),
        batch_size=args.training_args['batch_size']
    )

    n_user, n_item = train_data.shape

    model = MF(n_user, n_item * 2, dim=args.base_model_args['emb_dim'], dropout=0).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.base_model_args['learning_rate'], weight_decay=0)
    
    criterion = nn.BCELoss(reduction='sum')

    def run_test(model, loader, name):
        preds, trues, users_list, items_list = [], [], [], []
        
        ut_dict, pt_dict = {}, {}
        
        with torch.no_grad():
            for u, i, r in loader:
                pred = model(u.to(device), i.to(device))
                preds.append(pred)
                trues.append(r.to(device).float())
                users_list.append(u.to(device))
                items_list.append(i.to(device))
                
                for idx, user_id in enumerate(u):
                    uid = user_id.item()
                    rating = r[idx].item()
                    score = pred[idx].item()
                    
                    if uid not in ut_dict:
                        ut_dict[uid] = []
                        pt_dict[uid] = []
                    ut_dict[uid].append(rating)
                    pt_dict[uid].append(score)
        
        preds = torch.cat(preds)
        trues = torch.cat(trues)
        users_all = torch.cat(users_list)
        items_all = torch.cat(items_list)
        
        print(f"\n{'='*60}")
        print(f"[{name} Test Results]")
        print(f"{'='*60}")
        
        metric_list = ['BCE', 'AUC', 'UAUC', 'UNDCG', 'Recall_Precision_NDCG@']
        
        res = utils.metrics.evaluate(
            preds, trues, metric_list,
            users=users_all, items=items_all,
            UAUC=(ut_dict, pt_dict),
            UNDCG=(ut_dict, pt_dict)
        )
        
        for k, v in res.items():
            print(f"  {k}: {v:.4f}")
        
        return res
    
    stopping_args = Stop_args(patience=args.training_args['patience'], max_epochs=args.training_args['epochs'])
    early_stopping = EarlyStopping(model, **stopping_args)

    train_start = time.time()
    epoch_times = []

    for epo in range(early_stopping.max_epochs):
        epoch_start = time.time()
        training_loss = 0
        for u_batch_idx, users in enumerate(train_loader.User_loader): 
            for i_batch_idx, items in enumerate(train_loader.Item_loader): 
                model.train()
                users_train, items_train, y_train = train_loader.get_batch(users, items)
                users_unif, items_unif, y_unif = unif_loader.get_batch(users, items)
                items_unif = items_unif + n_item

                users_combine = torch.cat((users_train, users_unif))
                items_combine = torch.cat((items_train, items_unif))
                y_combine = torch.cat((y_train, y_unif))

                y_hat = model(users_combine, items_combine)

                student_items_embedding = model.item_latent.weight[items_train]
                teacher_items_embedding = model.item_latent.weight[items_train + n_item].detach()
                reg = torch.sum(torch.abs(student_items_embedding - teacher_items_embedding))

                loss = criterion(y_hat, y_combine.float()) + args.base_model_args['weight_decay'] * model.l2_norm(users_combine, items_combine) \
                        + args.teacher_model_args['weight_decay'] * reg

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                training_loss += loss.item()
        
        model.eval()
        with torch.no_grad():
            val_preds = []
            val_trues = []
            for u, i, r in unbiased_val_loader:
                pred = model(u.to(device), i.to(device))
                val_preds.append(pred)
                val_trues.append(r.to(device).float())
            
            val_preds = torch.cat(val_preds)
            val_trues = torch.cat(val_trues)
            val_results = utils.metrics.evaluate(val_preds, val_trues, ['BCE', 'AUC'])

        epoch_time = time.time() - epoch_start
        epoch_times.append(epoch_time)
        print(f'Epoch: {epo:3d} / {args.training_args["epochs"]}, Training Loss: {training_loss:.4f}, Val BCE: {val_results["BCE"]:.4f}, Val AUC: {val_results["AUC"]:.4f} | EpochTime: {epoch_time:.1f}s')

        if early_stopping.check([val_results['AUC']], epo):
            break

    train_total = time.time() - train_start
    avg_epoch = sum(epoch_times) / max(len(epoch_times), 1)
    print(f'Loading {early_stopping.best_epoch}th epoch')
    print(f'[Timing] Total Train: {train_total:.1f}s ({train_total/60:.2f}min) | Epochs: {len(epoch_times)} | Avg/epoch: {avg_epoch:.2f}s')
    model.load_state_dict(early_stopping.best_state)

    run_test(model, test_loader, "Unbiased Test Data (Random)")
    run_test(model, biased_test_loader, "Biased Test Data (Standard)")


if __name__ == "__main__": 
    args = arguments.parse_args()
    para(args)
    setup_seed(args.seed)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    print(f"\nLoading dataset: {args.dataset}")
    if args.dataset == 'KuaiRand':
        data_dict = load_kuairand.load_kuairand(device=device, verbose=True, random_seed=args.seed)
        train_data = data_dict['train_biased']
        biased_val_data = data_dict['val_biased']
        unif_train_data = data_dict['train_unbiased']
        unif_val_data = data_dict['val_unbiased']
        test_data = data_dict['test_unbiased']
        biased_test_data = data_dict['test_biased']
    else:
        train_data, biased_val_data, unif_train_data, unif_val_data, test_data, biased_test_data = \
            load_dataset_custom.load_dataset(data_name=args.dataset, unif_ratio=0.5, seed=args.seed, device=device)
        
        print(f"  Biased Train: {train_data._nnz()} samples")
        print(f"  Biased Val:   {biased_val_data._nnz()} samples")
        print(f"  Unbiased Train: {unif_train_data._nnz()} samples")
        print(f"  Unbiased Val:   {unif_val_data._nnz()} samples")
        print(f"  Biased Test:  {biased_test_data._nnz()} samples")
        print(f"  Unbiased Test: {test_data._nnz()} samples")
    
    train_and_eval(
        train_data, biased_val_data, unif_train_data, unif_val_data, 
        test_data, biased_test_data, args, device=device
    )
