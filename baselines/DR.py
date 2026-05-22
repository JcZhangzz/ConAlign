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
            'learning_rate': 0.0001,
            'weight_decay': 1
        }
        args.imputation_model_args = {
            'emb_dim': 10,
            'learning_rate': 1,
            'weight_decay': 0.1
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
        args.imputation_model_args = {
            'emb_dim': 10,
            'learning_rate': 1,
            'weight_decay': 0.1
        }
    elif 'KuaiRand' in args.dataset:
        args.training_args = {
            'batch_size': 1024,
            'epochs': 500,
            'patience': 60,
            'block_batch': [500, 10000]
        }
        args.base_model_args = {
            'emb_dim': 10,
            'learning_rate': 0.01,
            'weight_decay': 0.01
        }
        args.imputation_model_args = {
            'emb_dim': 10,
            'learning_rate': 0.01,
            'weight_decay': 0.1
        }


def train_and_eval(train_data, biased_val_data, unif_train_data, unif_val_data,
                   test_data, biased_test_data, args, device='cuda'):
    train_loader = utils.data_loader.Block(
        train_data,
        u_batch_size=args.training_args['block_batch'][0],
        i_batch_size=args.training_args['block_batch'][1],
        device=device
    )
    biased_val_loader = utils.data_loader.DataLoader(
        utils.data_loader.Interactions(biased_val_data),
        batch_size=args.training_args['batch_size'],
        shuffle=False, num_workers=0
    )
    unbiased_val_loader = utils.data_loader.DataLoader(
        utils.data_loader.Interactions(unif_val_data),
        batch_size=args.training_args['batch_size'],
        shuffle=False, num_workers=0
    )
    test_loader = utils.data_loader.DataLoader(
        utils.data_loader.Interactions(test_data),
        batch_size=args.training_args['batch_size'],
        shuffle=False, num_workers=0
    )
    biased_test_loader = utils.data_loader.DataLoader(
        utils.data_loader.Interactions(biased_test_data),
        batch_size=args.training_args['batch_size'],
        shuffle=False, num_workers=0
    )

    def naive_bayes_propensity(train, unif):
        P_Oeq1 = train._nnz() / (train.size()[0] * train.size()[1])
        y_unique = torch.unique(train._values())
        P_y_givenO = torch.zeros(y_unique.shape).to(device)
        P_y = torch.zeros(y_unique.shape).to(device)
        for i in range(len(y_unique)):
            P_y_givenO[i] = (torch.sum(train._values() == y_unique[i]).float()
                             / train._values().shape[0])
            P_y[i] = (torch.sum(unif._values() == y_unique[i]).float()
                      / unif._values().shape[0])
        Propensity = P_y_givenO / (P_y + 1e-8) * P_Oeq1
        return y_unique, Propensity

    y_unique, Propensity = naive_bayes_propensity(train_data, unif_train_data)
    InvP = torch.reciprocal(Propensity + 1e-8)

    n_user, n_item = train_data.shape
    base_model = MF(n_user, n_item, dim=args.base_model_args['emb_dim'], dropout=0).to(device)
    base_optimizer = torch.optim.SGD(
        base_model.parameters(),
        lr=args.base_model_args['learning_rate'], weight_decay=0
    )

    imputation_model = MF(n_user, n_item, dim=args.imputation_model_args['emb_dim'], dropout=0).to(device)
    imputation_optimizer = torch.optim.SGD(
        imputation_model.parameters(),
        lr=args.imputation_model_args['learning_rate'], weight_decay=0
    )

    none_criterion = nn.BCELoss(reduction='none')
    sum_criterion = nn.BCELoss(reduction='sum')
    none_mse = nn.MSELoss(reduction='none')

    stopping_args = Stop_args(
        patience=args.training_args['patience'],
        max_epochs=args.training_args['epochs']
    )
    early_stopping = EarlyStopping(base_model, **stopping_args)

    print("=" * 60)
    print(f"DR Training | Dataset: {args.dataset} | Users: {n_user}, Items: {n_item}")
    print("=" * 60)

    train_start = time.time()
    epoch_times = []

    for epo in range(early_stopping.max_epochs):
        epoch_start = time.time()
        base_model.train()
        imputation_model.train()

        for u_batch_idx, users in enumerate(train_loader.User_loader):
            for i_batch_idx, items in enumerate(train_loader.Item_loader):
                users_train, items_train, y_train = train_loader.get_batch(users, items, device=device)
                y_train = y_train.float()

                weight = torch.ones(y_train.shape).to(device)
                for i in range(len(y_unique)):
                    weight[y_train == y_unique[i]] = InvP[i]

                imputation_model.train()
                e_hat = imputation_model(users_train, items_train)
                with torch.no_grad():
                    e = y_train - base_model(users_train, items_train)
                cost_e = none_mse(e_hat, e)
                loss_imp = (torch.sum(weight * cost_e)
                            + args.imputation_model_args['weight_decay']
                            * imputation_model.l2_norm(users_train, items_train))
                imputation_optimizer.zero_grad()
                loss_imp.backward()
                imputation_optimizer.step()

                base_model.train()

                all_pair = torch.cartesian_prod(users, items)
                users_all, items_all = all_pair[:, 0], all_pair[:, 1]

                y_hat_all = base_model(users_all, items_all)
                y_hat_all_detach = y_hat_all.detach()
                g_all = imputation_model(users_all, items_all).detach()
                target_all = torch.clamp(g_all + y_hat_all_detach, 0.0, 1.0)
                loss_all = sum_criterion(y_hat_all, target_all)

                y_hat_obs = base_model(users_train, items_train)
                y_hat_obs_detach = y_hat_obs.detach()
                g_obs = imputation_model(users_train, items_train).detach()
                target_obs = torch.clamp(g_obs + y_hat_obs_detach, 0.0, 1.0)
                e_obs = none_criterion(y_hat_obs, y_train)
                e_hat_obs = none_criterion(y_hat_obs, target_obs)
                cost_obs = e_obs - e_hat_obs
                loss_base = (loss_all + torch.sum(weight * cost_obs)
                             + args.base_model_args['weight_decay']
                             * base_model.l2_norm(users_all, items_all))
                base_optimizer.zero_grad()
                loss_base.backward()
                base_optimizer.step()

        base_model.eval()
        with torch.no_grad():
            val_preds, val_trues = [], []
            for u, i, r in unbiased_val_loader:
                preds = base_model(u.to(device), i.to(device))
                val_preds.append(preds)
                val_trues.append(r.to(device).float())
            val_preds = torch.cat(val_preds)
            val_trues = torch.cat(val_trues)

        val_res = utils.metrics.evaluate(val_preds, val_trues, ['MSE', 'AUC'])
        epoch_time = time.time() - epoch_start
        epoch_times.append(epoch_time)
        print(f'Epoch {epo:3d} | Val MSE: {val_res["MSE"]:.4f} AUC: {val_res["AUC"]:.4f} | EpochTime: {epoch_time:.1f}s')

        if early_stopping.check([val_res['AUC']], epo):
            print(f'Early stopping at epoch {epo}')
            break

    train_total = time.time() - train_start
    avg_epoch = sum(epoch_times) / max(len(epoch_times), 1)
    print(f'\nLoading best model (epoch {early_stopping.best_epoch})')
    print(f'[Timing] Total Train: {train_total:.1f}s ({train_total/60:.2f}min) | Epochs: {len(epoch_times)} | Avg/epoch: {avg_epoch:.2f}s')
    base_model.load_state_dict(early_stopping.best_state)

    def run_test(loader, name):
        preds_list, trues_list, users_list, items_list = [], [], [], []
        ut_dict, pt_dict = {}, {}
        base_model.eval()
        with torch.no_grad():
            for u, i, r in loader:
                u, i, r = u.to(device), i.to(device), r.to(device).float()
                pred = base_model(u, i)
                preds_list.append(pred)
                trues_list.append(r)
                users_list.append(u)
                items_list.append(i)
                for idx in range(len(u)):
                    uid = u[idx].item()
                    if uid not in ut_dict:
                        ut_dict[uid] = []
                        pt_dict[uid] = []
                    ut_dict[uid].append(r[idx].item())
                    pt_dict[uid].append(pred[idx].item())

        preds = torch.cat(preds_list)
        trues = torch.cat(trues_list)
        users_all = torch.cat(users_list)
        items_all = torch.cat(items_list)

        metric_list = ['BCE', 'AUC', 'UAUC', 'UNDCG', 'Recall_Precision_NDCG@']
        res = utils.metrics.evaluate(
            preds, trues, metric_list,
            users=users_all, items=items_all,
            UAUC=(ut_dict, pt_dict),
            UNDCG=(ut_dict, pt_dict)
        )
        print(f"\n{'='*60}")
        print(f"[{name}]")
        print(f"{'='*60}")
        for k, v in res.items():
            print(f"  {k}: {v:.4f}")
        return res

    run_test(test_loader, "Unbiased Test Data (Random)")
    run_test(biased_test_loader, "Biased Test Data (Standard)")


if __name__ == "__main__":
    args = arguments.parse_args()
    para(args)
    setup_seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"Dataset: {args.dataset}")

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
                data_name=args.dataset, unif_ratio=0.5, seed=args.seed, device=device
            )
        print(f"  Biased Train:   {train_data._nnz()} samples")
        print(f"  Biased Val:     {biased_val_data._nnz()} samples")
        print(f"  Unbiased Train: {unif_train_data._nnz()} samples")
        print(f"  Unbiased Val:   {unif_val_data._nnz()} samples")
        print(f"  Unbiased Test:  {test_data._nnz()} samples")
        print(f"  Biased Test:    {biased_test_data._nnz()} samples")

    train_and_eval(
        train_data, biased_val_data, unif_train_data, unif_val_data,
        test_data, biased_test_data, args, device=device
    )
