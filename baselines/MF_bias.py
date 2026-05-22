import os
import sys
import time
import numpy as np
import random
import torch
import torch.nn as nn
from model import MF
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
        args.model_args = {
            'emb_dim': 10,
            'learning_rate': 0.001,
            'weight_decay': 0.0001
        }
    elif args.dataset == 'coat':
        args.training_args = {
            'batch_size': 128,
            'epochs': 500,
            'patience': 60,
            'block_batch': [64, 64]
        }
        args.model_args = {
            'emb_dim': 10,
            'learning_rate': 0.01,
            'weight_decay': 0.001
        }
    elif 'KuaiRand' in args.dataset:
        args.training_args = {
            'batch_size': 1024,
            'epochs': 1000,
            'patience': 60,
            'block_batch': [500, 10000]
        }
        args.model_args = {
            'emb_dim': 10,
            'learning_rate': 0.1,
            'weight_decay': 0.01
        }
    else: 
        print(f'Invalid dataset: {args.dataset}')
        os._exit(1)


def train_and_eval(train_data, biased_val_data, unif_train_data, unif_val_data, 
                   test_data, biased_test_data, args, device='cuda'):
    n_user, n_item = train_data.shape
    
    train_loader = utils.data_loader.Block(
        train_data, 
        u_batch_size=args.training_args['block_batch'][0], 
        i_batch_size=args.training_args['block_batch'][1], 
        device=device
    )
    val_loader = utils.data_loader.DataLoader(
        utils.data_loader.Interactions(unif_val_data),
        batch_size=args.training_args['batch_size'],
        shuffle=False, 
        num_workers=0
    )
    test_loader = utils.data_loader.DataLoader(
        utils.data_loader.Interactions(test_data),
        batch_size=args.training_args['batch_size'],
        shuffle=False, 
        num_workers=0
    )
    biased_test_loader = utils.data_loader.DataLoader(
        utils.data_loader.Interactions(biased_test_data),
        batch_size=args.training_args['batch_size'],
        shuffle=False, 
        num_workers=0
    )
    
    model = MF(n_user, n_item, dim=args.model_args['emb_dim'], dropout=0).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.model_args['learning_rate'], weight_decay=0)
    
    criterion = nn.BCELoss(reduction='sum')
    
    early_stopping = EarlyStopping(
        model,
        stop_varnames=[utils.early_stop.StopVariable.AUC],
        patience=args.training_args['patience'],
        max_epochs=args.training_args['epochs']
    )
    
    train_start = time.time()
    epoch_times = []

    for epo in range(args.training_args['epochs']):
        epoch_start = time.time()
        model.train()
        total_l2_loss = 0.0
        n_batches = 0
        
        for u_batch_idx, u_batch in enumerate(train_loader.User_loader):
            for i_batch_idx, i_batch in enumerate(train_loader.Item_loader):
                users_train, items_train, y_train = train_loader.get_batch(u_batch, i_batch, device=device)
                
                optimizer.zero_grad()
                pred = model(users_train, items_train)
                
                loss = criterion(pred, y_train.float()) + \
                       args.model_args['weight_decay'] * model.l2_norm(users_train, items_train)
                
                total_l2_loss += model.l2_norm(users_train, items_train).item()
                loss.backward()
                optimizer.step()
                n_batches += 1
        
        model.eval()
        with torch.no_grad():
            train_preds = []
            train_trues = []
            for u_batch_idx, u_batch in enumerate(train_loader.User_loader):
                for i_batch_idx, i_batch in enumerate(train_loader.Item_loader):
                    users_train, items_train, y_train = train_loader.get_batch(u_batch, i_batch, device=device)
                    train_preds.append(model(users_train, items_train))
                    train_trues.append(y_train.float())
            train_preds = torch.cat(train_preds)
            train_trues = torch.cat(train_trues)
            train_res = utils.metrics.evaluate(train_preds, train_trues, ['BCE', 'AUC'])
            
            val_preds = []
            val_trues = []
            for u, i, r in val_loader:
                val_preds.append(model(u.to(device), i.to(device)))
                val_trues.append(r.to(device).float())
            val_preds = torch.cat(val_preds)
            val_trues = torch.cat(val_trues)
            val_res = utils.metrics.evaluate(val_preds, val_trues, ['BCE', 'AUC'])
        
        avg_l2_loss = total_l2_loss / max(n_batches, 1)
        reg_term = args.model_args['weight_decay'] * avg_l2_loss

        epoch_time = time.time() - epoch_start
        epoch_times.append(epoch_time)

        print(f"Epoch {epo:3d} | Train BCE: {train_res['BCE']:.4f} AUC: {train_res['AUC']:.4f} | "
              f"Val BCE: {val_res['BCE']:.4f} AUC: {val_res['AUC']:.4f} | "
              f"L2: {avg_l2_loss:.2f} Reg: {reg_term:.4f} | EpochTime: {epoch_time:.1f}s")
        
        if early_stopping.check([val_res['AUC']], epo):
            print(f"Early stopping at epoch {epo}")
            break
    
    train_total = time.time() - train_start
    avg_epoch = sum(epoch_times) / max(len(epoch_times), 1)
    model.load_state_dict(early_stopping.best_state)
    print(f"\nBest model loaded (Best Epoch: {early_stopping.best_epoch})")
    print(f"[Timing] Total Train: {train_total:.1f}s ({train_total/60:.2f}min) | Epochs: {len(epoch_times)} | Avg/epoch: {avg_epoch:.2f}s")
    
    def run_test(loader, name):
        preds, trues, us, is_ = [], [], [], []
        
        ut_dict, pt_dict = {}, {}
        
        with torch.no_grad():
            for u, i, r in loader:
                pred = model(u.to(device), i.to(device))
                preds.append(pred)
                trues.append(r.to(device).float())
                us.append(u.to(device))
                is_.append(i.to(device))
                
                for idx, user_id in enumerate(u):
                    uid = user_id.item()
                    rating = r[idx].item()
                    score = pred[idx].item()
                    
                    if uid not in ut_dict:
                        ut_dict[uid] = []
                        pt_dict[uid] = []
                    ut_dict[uid].append(rating)
                    pt_dict[uid].append(score)
        
        preds_cat = torch.cat(preds)
        trues_cat = torch.cat(trues)
        users_cat = torch.cat(us)
        items_cat = torch.cat(is_)
        
        metric_list = ['BCE', 'AUC', 'UAUC', 'UNDCG', 'Recall_Precision_NDCG@']
        
        res = utils.metrics.evaluate(
            preds_cat, trues_cat, metric_list,
            users=users_cat, items=items_cat,
            UAUC=(ut_dict, pt_dict),
            UNDCG=(ut_dict, pt_dict)
        )
        
        print(f"\n{'='*60}")
        print(f"[{name} Test Results]")
        print(f"{'='*60}")
        for k, v in res.items():
            print(f"  {k}: {v:.4f}")
        return res
    
    run_test(test_loader, "Unbiased (Random)")
    run_test(biased_test_loader, "Biased (Standard)")


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
            load_dataset_custom.load_dataset(
                data_name=args.dataset, 
                unif_ratio=0.5, 
                seed=args.seed, 
                device=device
            )
        
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
