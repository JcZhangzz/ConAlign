import numpy as np
from scipy.sparse import lil_matrix
import torch
import pandas as pd

import utils.data_loader
import cppimport.import_hook


def calc(n,m,ttuser,ttitem,pre,ttrating,atk=5):
    user=ttuser.cpu().detach().numpy()
    item=ttitem.cpu().detach().numpy()
    pre=pre.cpu().detach().numpy()
    rating=ttrating.cpu().numpy()
    
    posid=np.where(rating==1)
    posuser=user[posid]
    positem=item[posid]
    
    user_labels = {}
    for i in range(len(user)):
        u = user[i]
        if u not in user_labels:
            user_labels[u] = set()
        user_labels[u].add(rating[i])
    
    valid_users = set()
    for u, labels in user_labels.items():
        if (1 in labels) and (0 in labels):
            valid_users.add(u)
    
    user_preds = {}
    user_items = {}
    for i in range(len(user)):
        u, it, p = user[i], item[i], pre[i]
        if u not in valid_users:
            continue
        if u not in user_preds:
            user_preds[u] = []
            user_items[u] = []
        user_preds[u].append(p)
        user_items[u].append(it)
    
    logsum = 1 / np.log2(np.arange(m + 2)[2:])
    logsum = np.cumsum(logsum)
    
    precision, recall, ndcg = 0, 0, 0
    n_interacted_user = 0
    
    keys = set()
    user_pos_count = {}
    for i in range(len(posuser)):
        u, it = posuser[i], positem[i]
        if u not in valid_users:
            continue
        keys.add((u, it))
        user_pos_count[u] = user_pos_count.get(u, 0) + 1
    
    for u in user_preds:
        if user_pos_count.get(u, 0) == 0:
            continue
        
        n_interacted_user += 1
        preds = np.array(user_preds[u])
        items = np.array(user_items[u])
        n_pos = user_pos_count[u]
        
        sorted_idx = np.argsort(preds)[::-1]
        topk_items = items[sorted_idx[:atk]]
        
        hit = sum(1 for it in topk_items if (u, it) in keys)
        
        precision += hit / atk
        recall += hit / n_pos
        
        dcg = 0
        for i, it in enumerate(topk_items):
            if (u, it) in keys:
                dcg += 1.0 / np.log2(i + 2)
        
        idcg = logsum[min(n_pos, atk) - 1] if n_pos > 0 else 0
        
        if idcg > 0:
            ndcg += dcg / idcg
    
    if n_interacted_user == 0:
        return [0.0, 0.0, 0.0]
    
    return [precision / n_interacted_user, 
            recall / n_interacted_user, 
            ndcg / n_interacted_user]


def auc(vector_predict, vector_true, device = 'cpu'): 
    pos_indexes = torch.where(vector_true == 1)[0].to(device)
    n_pos = len(pos_indexes)
    n_total = len(vector_predict)
    n_neg = n_total - n_pos
    
    if n_pos == 0 or n_neg == 0:
        return 0.5
    
    pos_whe = (vector_true == 1).to(device)
    sort_indexes = torch.argsort(vector_predict).to(device)
    rank = torch.zeros((n_total)).to(device)
    rank[sort_indexes] = torch.FloatTensor(list(range(n_total))).to(device)
    rank = rank * pos_whe
    auc = (torch.sum(rank) - n_pos * (n_pos - 1) / 2) / (n_pos * n_neg)
    return auc.item()

def uauc(UAUC, device='cpu'):
    (ut_dict, pt_dict) = UAUC
    uauc_sum = 0.0
    valid_users = 0
    
    for user_id in ut_dict:
        user_trues = ut_dict[user_id]
        user_preds = pt_dict[user_id]
        
        if (1 in user_trues) and (0 in user_trues):
            user_auc = auc(torch.tensor(user_preds), torch.tensor(user_trues), device=device)
            uauc_sum += user_auc
            valid_users += 1
    
    if valid_users == 0:
        return 0.0
    return uauc_sum / valid_users

def undcg(UNDCG, atk=5):
    (ut_dict, pt_dict) = UNDCG
    ndcg_sum = 0.0
    valid_users = 0
    
    for user_id in ut_dict:
        user_trues = np.array(ut_dict[user_id])
        user_preds = np.array(pt_dict[user_id])
        
        if 1 not in user_trues:
            continue
        
        sorted_indices = np.argsort(user_preds)[::-1]
        sorted_trues = user_trues[sorted_indices]
        
        dcg = 0.0
        for i in range(min(atk, len(sorted_trues))):
            if sorted_trues[i] == 1:
                dcg += 1.0 / np.log2(i + 2)
        
        n_pos = np.sum(user_trues == 1)
        idcg = 0.0
        for i in range(min(atk, n_pos)):
            idcg += 1.0 / np.log2(i + 2)
        
        if idcg > 0:
            ndcg_sum += dcg / idcg
            valid_users += 1
    
    if valid_users == 0:
        return 0.0
    return ndcg_sum / valid_users

def mse(vector_predict, vector_true): 
    mse = torch.mean((vector_predict - vector_true)**2)
    return mse.item()

def bce(vector_predict, vector_true):
    eps = 1e-6
    vector_predict = vector_predict.float()
    vector_true = vector_true.float()
    vector_predict = torch.clamp(vector_predict, eps, 1 - eps)
    bce = -torch.mean(vector_true * torch.log(vector_predict) + (1 - vector_true) * torch.log(1 - vector_predict))
    if torch.isinf(bce) or torch.isnan(bce):
        return float('nan')
    return bce.item()

def evaluate(vector_Predict, vector_Test, metric_names, users = None, items = None, UAUC=None, UNDCG=None):
    global_metrics = {
        "AUC": auc,
        "MSE": mse,
        "BCE": bce,
        'Recall_Precision_NDCG@': 5}

    results = {}
    for name in metric_names:
        if name not in ['Recall_Precision_NDCG@', 'UAUC', 'UNDCG']:
            results[name] = global_metrics[name](vector_predict=vector_Predict,
                                                      vector_true=vector_Test)

    if 'Recall_Precision_NDCG@' in metric_names: 
        users_num = torch.max(users).item() + 1
        items_num = torch.max(items).item() + 1
        Recall_Precision_NDCG = calc(users_num, items_num, users, items, vector_Predict, vector_Test, atk=global_metrics['Recall_Precision_NDCG@'])
        results['Precision'] =  Recall_Precision_NDCG[0]
        results['Recall'] =  Recall_Precision_NDCG[1]
        results['NDCG'] =  Recall_Precision_NDCG[2]
    
    if 'UAUC' in metric_names and UAUC is not None:
        results['UAUC'] = uauc(UAUC)
    
    if 'UNDCG' in metric_names and UNDCG is not None:
        results['UNDCG'] = undcg(UNDCG, atk=global_metrics.get('Recall_Precision_NDCG@', 5))
        
    return results
