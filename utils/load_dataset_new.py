import os
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch


def seed_randomly_split(df, ratio, split_seed, shape):
    np.random.seed(split_seed)
    indices = np.random.permutation(len(df))
    idx1 = int(ratio[0] * len(df))
    idx2 = int((ratio[0] + ratio[1]) * len(df))
    
    def to_matrix(data_df):
        return sp.csr_matrix((data_df['rating'], (data_df['uid'], data_df['iid'])), shape=shape, dtype='float32')

    return to_matrix(df.iloc[indices[:idx1]]), to_matrix(df.iloc[indices[idx1:idx2]]), to_matrix(df.iloc[indices[idx2:]])

def load_dataset(data_name='yahooR3', type='explicit', unif_ratio=0.05, seed=0, threshold=4, device='cuda'):
    path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'datasets', data_name)
    
    user_df = pd.read_csv(os.path.join(path, 'user.txt'), sep=',', header=None, names=['uid', 'iid', 'rating'])
    random_df = pd.read_csv(os.path.join(path, 'random.txt'), sep=',', header=None, names=['uid', 'iid', 'rating'])

    print("Compressing IDs...")
    all_uids = pd.unique(pd.concat([user_df['uid'], random_df['uid']]))
    all_iids = pd.unique(pd.concat([user_df['iid'], random_df['iid']]))
    
    uid_map = {old: new for new, old in enumerate(all_uids)}
    iid_map = {old: new for new, old in enumerate(all_iids)}
    
    user_df['uid'] = user_df['uid'].map(uid_map)
    user_df['iid'] = user_df['iid'].map(iid_map)
    random_df['uid'] = random_df['uid'].map(uid_map)
    random_df['iid'] = random_df['iid'].map(iid_map)

    if 'KuaiRand' in data_name:
        user_df['rating'] = user_df['rating'].apply(lambda x: 1.0 if x > 0 else 0.0)
        random_df['rating'] = random_df['rating'].apply(lambda x: 1.0 if x > 0 else 0.0)
    else:
        user_df['rating'] = np.where(user_df['rating'] >= threshold, 1.0, 0.0)
        random_df['rating'] = np.where(random_df['rating'] >= threshold, 1.0, 0.0)
    
    print(f"Label range: [{user_df['rating'].min():.0f}, {user_df['rating'].max():.0f}] (0=negative, 1=positive)")

    shape = (len(all_uids), len(all_iids))
    print(f"Dataset: {data_name} | Users: {shape[0]}, Items: {shape[1]}")

    biased_ratio = (0.7, 0.1, 0.2)
    biased_train_mat, biased_val_mat, biased_test_mat = seed_randomly_split(user_df, biased_ratio, seed, shape)
    
    u_ratio = (unif_ratio, 0.1, 1 - unif_ratio - 0.1)
    unif_train_mat, unif_val_mat, unif_test_mat = seed_randomly_split(random_df, u_ratio, seed, shape)

    train = sparse_mx_to_torch_sparse_tensor(biased_train_mat).to(device)
    biased_val = sparse_mx_to_torch_sparse_tensor(biased_val_mat).to(device)
    unif_train = sparse_mx_to_torch_sparse_tensor(unif_train_mat).to(device)
    unif_val = sparse_mx_to_torch_sparse_tensor(unif_val_mat).to(device)
    test_unbiased = sparse_mx_to_torch_sparse_tensor(unif_test_mat).to(device)
    test_biased = sparse_mx_to_torch_sparse_tensor(biased_test_mat).to(device)

    return train, biased_val, unif_train, unif_val, test_unbiased, test_biased

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    sparse_mx = sparse_mx.tocoo()
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data).float()
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse_coo_tensor(indices, values, shape)

def split_per_user(df, ratios, seed):
    if len(ratios) == 2:
        train_list, test_list = [], []
        for uid, group in df.groupby('uid'):
            shuffled = group.sample(frac=1, random_state=seed).reset_index(drop=True)
            n = len(shuffled)
            idx = int(n * ratios[0])
            if idx == 0 and n > 0: idx = 1
            train_list.append(shuffled.iloc[:idx])
            test_list.append(shuffled.iloc[idx:])
        return pd.concat(train_list), pd.concat(test_list)
    elif len(ratios) == 3:
        train_list, val_list, test_list = [], [] , []
        for uid, group in df.groupby('uid'):
            shuffled = group.sample(frac=1, random_state=seed).reset_index(drop=True)
            n = len(shuffled)
            idx1 = int(n * ratios[0])
            idx2 = int(n * (ratios[0] + ratios[1]))
            if idx1 == 0 and n > 0: idx1 = 1
            if idx2 <= idx1 and n > idx1: idx2 = idx1 + 1
            train_list.append(shuffled.iloc[:idx1])
            val_list.append(shuffled.iloc[idx1:idx2])
            test_list.append(shuffled.iloc[idx2:])
        return pd.concat(train_list), pd.concat(val_list), pd.concat(test_list)
    else:
        raise ValueError(f"Unsupported number of ratios: {len(ratios)}")