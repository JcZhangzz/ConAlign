import os
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
    if torch.cuda.is_available(): torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def para(args): 
    if args.dataset == 'yahooR3': 
        args.training_args = {'batch_size': 1024, 'epochs': 1000, 'patience': 60, 'block_batch': [6000, 500]}
        args.base_model_args = {'emb_dim': 10, 'learning_rate': 0.0001, 'imputaion_lambda': 0.1, 'weight_decay': 1}
        args.weight1_model_args = {'learning_rate': 0.01, 'weight_decay': 0.01}
        args.weight2_model_args = {'learning_rate': 1e-3, 'weight_decay': 1e-2}
        args.imputation_model_args = {'learning_rate': 1e-1, 'weight_decay': 1e-4}
    elif args.dataset == 'coat':
        args.training_args = {'batch_size': 128, 'epochs': 1000, 'patience': 60, 'block_batch': [64, 64]}
        args.base_model_args = {'emb_dim': 10, 'learning_rate': 0.0001, 'imputaion_lambda': 0.05, 'weight_decay': 1}
        args.weight1_model_args = {'learning_rate': 1e-4, 'weight_decay':1e-4}
        args.weight2_model_args = {'learning_rate': 1e-4, 'weight_decay': 1e-3}
        args.imputation_model_args = {'learning_rate': 1e-3, 'weight_decay': 1e-4}
    elif 'KuaiRand' in args.dataset:
        args.training_args = {'batch_size': 2048, 'epochs': 600, 'patience': 80, 'block_batch': [2000, 200]}
        args.base_model_args = {'emb_dim': 32, 'learning_rate': 0.0001, 'imputaion_lambda': 0.05, 'weight_decay': 1e-3}
        args.weight1_model_args = {'learning_rate': 0.01, 'weight_decay': 1e-3}
        args.weight2_model_args = {'learning_rate': 1e-3, 'weight_decay': 1e-3}
        args.imputation_model_args = {'learning_rate': 1e-2, 'weight_decay': 1e-4}

def train_and_eval(train_data, unif_train_data, val_data, test_data, biased_test_data, args, device='cuda'):
    users_unif = unif_train_data._indices()[0]
    items_unif = unif_train_data._indices()[1]
    y_unif = unif_train_data._values().float()
    
    train_loader = utils.data_loader.Block(train_data, u_batch_size=args.training_args['block_batch'][0], 
                                          i_batch_size=args.training_args['block_batch'][1], device=device)
    val_loader = utils.data_loader.DataLoader(utils.data_loader.Interactions(val_data), batch_size=args.training_args['batch_size'])
    test_loader = utils.data_loader.DataLoader(utils.data_loader.Interactions(test_data), batch_size=args.training_args['batch_size'])
    biased_test_loader = utils.data_loader.DataLoader(utils.data_loader.Interactions(biased_test_data), batch_size=args.training_args['batch_size'])
    
    n_user, n_item = train_data.shape
    base_model = MetaMF(n_user, n_item, dim=args.base_model_args['emb_dim']).to(device)
    weight1_model = ThreeLinear(n_user, n_item, 2).to(device)
    weight2_model = ThreeLinear(n_user, n_item, 2).to(device)
    imputation_model = OneLinear(3).to(device)
    
    save_path = f"best_model_{args.dataset}.pth"
    is_test_only = getattr(args, 'test_only', False)

    if not is_test_only:
        print("Starting Training Loop...")
        base_optimizer = torch.optim.SGD(base_model.params(), lr=args.base_model_args['learning_rate'])
        weight1_optimizer = torch.optim.Adam(
            weight1_model.parameters(), 
            lr=args.weight1_model_args['learning_rate'],
            weight_decay=args.weight1_model_args['weight_decay']
        )
        weight2_optimizer = torch.optim.Adam(
            weight2_model.parameters(), 
            lr=args.weight2_model_args['learning_rate'],
            weight_decay=args.weight2_model_args['weight_decay']
        )
        imputation_optimizer = torch.optim.Adam(
            imputation_model.parameters(), 
            lr=args.imputation_model_args['learning_rate'],
            weight_decay=args.imputation_model_args['weight_decay']
        )
        
        sum_criterion = nn.BCELoss(reduction='sum')
        none_criterion = nn.BCELoss(reduction='none')
        stopping_args = Stop_args(patience=args.training_args['patience'], max_epochs=args.training_args['epochs'])
        early_stopping = EarlyStopping(base_model, **stopping_args)

        for epo in range(args.training_args['epochs']):
            base_model.train()
            total_l2_loss = 0.0
            train_preds = []
            train_trues = []
            
            for u_batch_idx, u_batch in enumerate(train_loader.User_loader): 
                for i_batch_idx, i_batch in enumerate(train_loader.Item_loader): 
                    users_train, items_train, y_train = train_loader.get_batch(u_batch, i_batch, device=device)
                    y_train = y_train.float()
                    
                    all_pair = torch.cartesian_prod(u_batch, i_batch)
                    u_all, i_all = all_pair[:,0], all_pair[:,1]
                    
                    local_status_map = torch.full((len(u_batch), len(i_batch)), -1.0, device=device)
                    for ut, it, yt in zip(users_train, items_train, y_train):
                        u_idx = (u_batch == ut).nonzero(as_tuple=True)[0]
                        i_idx = (i_batch == it).nonzero(as_tuple=True)[0]
                        local_status_map[u_idx, i_idx] = yt

                    flat_status = local_status_map.flatten().long()
                    impu_indices = (flat_status + 1).clamp(0, 2)
                    obs_status = (flat_status != -1).long()

                    one_step_model = MetaMF(n_user, n_item, dim=args.base_model_args['emb_dim']).to(device)
                    one_step_model.load_state_dict(base_model.state_dict())
                    
                    weight1_model.train()
                    weight2_model.train()
                    imputation_model.train()
                    
                    w1 = torch.exp(weight1_model(users_train, items_train, (y_train == 1) * 1) / 5)
                    w2 = torch.exp(weight2_model(u_all, i_all, obs_status) / 5)
                    impu_f = torch.sigmoid(imputation_model(impu_indices))

                    l_obs = torch.sum(none_criterion(one_step_model(users_train, items_train), y_train) * w1)
                    l_all = torch.sum(none_criterion(one_step_model(u_all, i_all), impu_f) * w2)
                    l_total = l_obs + args.base_model_args['imputaion_lambda'] * l_all + args.base_model_args['weight_decay'] * one_step_model.l2_norm(u_all, i_all)
                    
                    grads = torch.autograd.grad(l_total, one_step_model.params(), create_graph=True)
                    one_step_model.update_params(args.base_model_args['learning_rate'], source_params=grads)

                    loss_meta = sum_criterion(one_step_model(users_unif, items_unif), y_unif)
                    weight1_optimizer.zero_grad(); weight2_optimizer.zero_grad(); imputation_optimizer.zero_grad()
                    loss_meta.backward()
                    if epo >= 20: weight1_optimizer.step(); weight2_optimizer.step()
                    imputation_optimizer.step()

                    base_optimizer.zero_grad()
                    
                    weight1_model.train()
                    weight2_model.train()
                    imputation_model.train()
                    
                    final_w1 = torch.exp(weight1_model(users_train, items_train, (y_train == 1) * 1) / 5)
                    final_w2 = torch.exp(weight2_model(u_all, i_all, obs_status) / 5)
                    final_impu = torch.sigmoid(imputation_model(impu_indices))
                    
                    loss = torch.sum(none_criterion(base_model(users_train, items_train), y_train) * final_w1) + \
                           args.base_model_args['imputaion_lambda'] * torch.sum(none_criterion(base_model(u_all, i_all), final_impu) * final_w2) + \
                           args.base_model_args['weight_decay'] * base_model.l2_norm(u_all, i_all)
                    loss.backward()
                    base_optimizer.step()
                    
                    total_l2_loss += base_model.l2_norm(u_all, i_all).item()
                    train_preds.append(base_model(users_train, items_train).detach())
                    train_trues.append(y_train.detach())
            
            base_model.eval()
            with torch.no_grad():
                train_preds = torch.cat(train_preds)
                train_trues = torch.cat(train_trues)
                train_res = utils.metrics.evaluate(train_preds, train_trues, ['BCE', 'AUC'])
            
            with torch.no_grad():
                v_p = torch.cat([base_model(u.to(device), i.to(device)) for u, i, r in val_loader])
                v_t = torch.cat([r.to(device).float() for u, i, r in val_loader])
                val_res = utils.metrics.evaluate(v_p, v_t, ['BCE', 'AUC'])
            
            total_batches = len(train_loader.User_loader) * len(train_loader.Item_loader)
            avg_l2_loss = total_l2_loss / total_batches
            reg_term = args.base_model_args['weight_decay'] * avg_l2_loss
            
            print(f"Epoch {epo:3d} | Train BCE: {train_res['BCE']:.4f} AUC: {train_res['AUC']:.4f} | "
                  f"Val BCE: {val_res['BCE']:.4f} AUC: {val_res['AUC']:.4f} | "
                  f"L2: {avg_l2_loss:.2f} Reg: {reg_term:.4f}")
            
            if early_stopping.check([val_res['AUC']], epo): break
        
        base_model.load_state_dict(early_stopping.best_state)
        torch.save(base_model.state_dict(), save_path)

    else:
        base_model.load_state_dict(torch.load(save_path, map_location=device))

    base_model.eval()
    def run_test(loader, name):
        preds, trues, us, is_ = [], [], [], []
        
        ut_dict, pt_dict = {}, {}
        
        with torch.no_grad():
            for u, i, r in loader:
                pred = base_model(u.to(device), i.to(device))
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
        
        items_num = torch.max(items_cat).item() + 1
        
        base_metrics = ['BCE', 'AUC', 'UAUC', 'UNDCG']
        
        if items_num <= 100000:
            metric_list = base_metrics + ['Recall_Precision_NDCG@']
        else:
            metric_list = base_metrics
        
        res = utils.metrics.evaluate(preds_cat, trues_cat, metric_list,
                                     users=users_cat, items=items_cat,
                                     UAUC=(ut_dict, pt_dict),
                                     UNDCG=(ut_dict, pt_dict))
        print(f"\n[{name} Test Results]:")
        for k, v in res.items():
            print(f"  {k}: {v:.4f}")

    run_test(test_loader, "Unbiased (Random)")
    run_test(biased_test_loader, "Biased (Standard)")

if __name__ == "__main__": 
    args = arguments.parse_args()
    para(args)
    setup_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if args.dataset == 'KuaiRand':
        data_dict = load_kuairand.load_kuairand(device=device, verbose=True, random_seed=args.seed)
        train_data = data_dict['train_biased']
        biased_val_data = data_dict['val_biased']
        unif_train_data = data_dict['train_unbiased']
        unif_val_data = data_dict['val_unbiased']
        test_unbiased = data_dict['test_unbiased']
        test_biased = data_dict['test_biased']
    else:
        train, biased_val, unif_train, unif_val, test_unbiased, test_biased = load_dataset_custom.load_dataset(data_name=args.dataset, seed=args.seed, device=device, unif_ratio=0.05)
    
    train_and_eval(train, unif_train, unif_val, test_unbiased, test_biased, args, device=device)