# -*- coding: utf-8 -*-
"""
KuaiRand-Pure Data Loader - Extended Time Range Version
Modeling is_click
Data split:
- Biased training: 2022-04-08 to 2022-05-05 (first 28 days, merged from two files)
- Validation: 2022-05-06 (1 day)
- Test: 2022-05-07 to 2022-05-08 (last 2 days)

Note: Although named "1k", this loader uses Pure dataset (larger scale).
"""
import os
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from datetime import datetime


def load_kuairand(data_dir=None, device='cuda', verbose=True, biased_item_sample_ratio=1, random_seed=None):
    """
    Load KuaiRand-Pure dataset with extended time-based split
    
    Merges two biased data files:
    - log_standard_4_08_to_4_21_pure.csv (Apr 8-21)
    - log_standard_4_22_to_5_08_pure.csv (Apr 22 - May 8)
    
    Args:
        data_dir: Data directory path
        device: torch device
        verbose: Print detailed information
        biased_item_sample_ratio: Sampling ratio for biased data items (default 1.0)
        random_seed: Random seed for biased item sampling
    
    Returns:
        Dictionary containing:
        - train_biased, val_biased, test_biased: Biased data splits
        - train_unbiased, val_unbiased, test_unbiased: Unbiased data splits
        - n_users, n_items: Number of users and items
    """
    # Set path
    if data_dir is None:
        data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 
                               'datasets', 'KuaiRand-Pure', 'data')
    
    # Define file paths
    biased_file_early = os.path.join(data_dir, 'log_standard_4_08_to_4_21_pure.csv')
    biased_file_late = os.path.join(data_dir, 'log_standard_4_22_to_5_08_pure.csv')
    unbiased_file = os.path.join(data_dir, 'log_random_4_22_to_5_08_pure.csv')
    
    if verbose:
        print("="*70)
        print("Loading KuaiRand-Pure Dataset (Extended Time Split)")
        print("="*70)
        print(f"Biased file 1 (Apr 8-21): {biased_file_early}")
        print(f"Biased file 2 (Apr 22 - May 8): {biased_file_late}")
        print(f"Unbiased file: {unbiased_file}")
    
    # ========== 1. Load Data ==========
    if verbose:
        print("\n[1/6] Loading raw data files...")
    
    # Load both biased files
    biased_df_early = pd.read_csv(biased_file_early)
    biased_df_late = pd.read_csv(biased_file_late)
    unbiased_df = pd.read_csv(unbiased_file)
    
    if verbose:
        print(f"  Biased early (Apr 8-21): {len(biased_df_early):,} samples")
        print(f"  Biased late (Apr 22 - May 8): {len(biased_df_late):,} samples")
        print(f"  Unbiased: {len(unbiased_df):,} samples")
    
    # ========== 2. Merge Biased Data ==========
    if verbose:
        print("\n[2/6] Merging biased data files...")
    
    # Concatenate two biased files
    biased_df = pd.concat([biased_df_early, biased_df_late], ignore_index=True)
    
    if verbose:
        print(f"  Total biased samples after merge: {len(biased_df):,}")
        print(f"  Date range: {biased_df['date'].min()} to {biased_df['date'].max()}")
    
    # ========== 3. Data Preprocessing ==========
    if verbose:
        print("\n[3/6] Preprocessing data...")
    
    # Keep only needed columns
    biased_df = biased_df[['user_id', 'video_id', 'date', 'is_click']].copy()
    unbiased_df = unbiased_df[['user_id', 'video_id', 'date', 'is_click']].copy()
    
    # Remove missing values
    biased_df = biased_df.dropna()
    unbiased_df = unbiased_df.dropna()
    
    if verbose:
        print(f"  After dropna - Biased: {len(biased_df):,}, Unbiased: {len(unbiased_df):,}")
    
    # Optional: sample biased items by ratio (e.g. 1/100)
    if biased_item_sample_ratio < 1.0:
        if biased_item_sample_ratio <= 0:
            raise ValueError(f"biased_item_sample_ratio must be in (0, 1], got {biased_item_sample_ratio}")
        if biased_item_sample_ratio > 1.0:
            raise ValueError(f"biased_item_sample_ratio must be in (0, 1], got {biased_item_sample_ratio}")

        biased_unique_items = biased_df['video_id'].unique()
        sample_item_num = max(1, int(len(biased_unique_items) * biased_item_sample_ratio))
        rng = np.random.default_rng(random_seed)
        sampled_items = rng.choice(biased_unique_items, size=sample_item_num, replace=False)
        sampled_item_set = set(sampled_items.tolist())
        biased_df = biased_df[biased_df['video_id'].isin(sampled_item_set)].copy()

        if verbose:
            print(f"  Biased item sampling enabled: ratio={biased_item_sample_ratio}, seed={random_seed}")
            print(f"  Biased unique items: {len(biased_unique_items):,} -> {sample_item_num:,}")
            print(f"  Biased samples after item sampling: {len(biased_df):,}")
    
    # is_click distribution
    if verbose:
        print("\n  is_click distribution:")
        print(f"  Biased: {biased_df['is_click'].value_counts().to_dict()}")
        print(f"  Unbiased: {unbiased_df['is_click'].value_counts().to_dict()}")
    
    # ========== 4. ID Compression Mapping ==========
    if verbose:
        print("\n[4/6] Compressing IDs...")
    
    # Merge all user and item IDs
    all_users = pd.unique(pd.concat([biased_df['user_id'], unbiased_df['user_id']]))
    all_items = pd.unique(pd.concat([biased_df['video_id'], unbiased_df['video_id']]))
    
    # Create mapping
    user_map = {old: new for new, old in enumerate(all_users)}
    item_map = {old: new for new, old in enumerate(all_items)}
    
    n_users = len(all_users)
    n_items = len(all_items)
    
    # Apply mapping
    biased_df['user_id'] = biased_df['user_id'].map(user_map)
    biased_df['video_id'] = biased_df['video_id'].map(item_map)
    unbiased_df['user_id'] = unbiased_df['user_id'].map(user_map)
    unbiased_df['video_id'] = unbiased_df['video_id'].map(item_map)
    
    if verbose:
        print(f"  Users: {n_users:,}, Items: {n_items:,}")
        print(f"  Matrix shape: ({n_users}, {n_items})")
    
    # ========== 5. Split by Date ==========
    if verbose:
        print("\n[5/6] Splitting by date...")
    
    # Define date split
    # Training: Apr 8 - May 5 (first 28 days)
    # Validation: May 6 (1 day)
    # Test: May 7-8 (last 2 days)
    train_dates = [20220408 + i for i in range(28)]  # 20220408 to 20220505
    val_date = 20220506
    test_dates = [20220507, 20220508]
    
    if verbose:
        print(f"  Train dates: 2022-04-08 to 2022-05-05 (28 days)")
        print(f"  Val date: 2022-05-06 (1 day)")
        print(f"  Test dates: 2022-05-07 to 2022-05-08 (2 days)")
    
    # Biased data split
    biased_train = biased_df[biased_df['date'].isin(train_dates)].copy()
    biased_val = biased_df[biased_df['date'] == val_date].copy()
    biased_test = biased_df[biased_df['date'].isin(test_dates)].copy()
    
    # Unbiased data split
    unbiased_train = unbiased_df[unbiased_df['date'].isin(train_dates)].copy()
    unbiased_val = unbiased_df[unbiased_df['date'] == val_date].copy()
    unbiased_test = unbiased_df[unbiased_df['date'].isin(test_dates)].copy()
    
    # Deduplicate user-item pairs in each split by mean(is_click), then binarize to 0/1
    biased_train = deduplicate_and_binarize_interactions(biased_train)
    biased_val = deduplicate_and_binarize_interactions(biased_val)
    biased_test = deduplicate_and_binarize_interactions(biased_test)
    unbiased_train = deduplicate_and_binarize_interactions(unbiased_train)
    unbiased_val = deduplicate_and_binarize_interactions(unbiased_val)
    unbiased_test = deduplicate_and_binarize_interactions(unbiased_test)
    
    if verbose:
        print("\n  Biased data split:")
        print(f"    Train: {len(biased_train):,} samples")
        print(f"    Val: {len(biased_val):,} samples")
        print(f"    Test: {len(biased_test):,} samples")
        
        print("\n  Unbiased data split:")
        print(f"    Train: {len(unbiased_train):,} samples")
        print(f"    Val: {len(unbiased_val):,} samples")
        print(f"    Test: {len(unbiased_test):,} samples")
    
    # ========== 6. Convert to Sparse Tensor ==========
    if verbose:
        print("\n[6/6] Converting to sparse tensors...")
    
    shape = (n_users, n_items)
    
    # Convert to sparse matrix
    train_biased_mat = df_to_sparse_matrix(biased_train, shape)
    val_biased_mat = df_to_sparse_matrix(biased_val, shape)
    test_biased_mat = df_to_sparse_matrix(biased_test, shape)
    
    train_unbiased_mat = df_to_sparse_matrix(unbiased_train, shape)
    val_unbiased_mat = df_to_sparse_matrix(unbiased_val, shape)
    test_unbiased_mat = df_to_sparse_matrix(unbiased_test, shape)
    
    # Convert to PyTorch sparse tensor
    train_biased = sparse_mx_to_torch_sparse_tensor(train_biased_mat).to(device)
    val_biased = sparse_mx_to_torch_sparse_tensor(val_biased_mat).to(device)
    test_biased = sparse_mx_to_torch_sparse_tensor(test_biased_mat).to(device)
    
    train_unbiased = sparse_mx_to_torch_sparse_tensor(train_unbiased_mat).to(device)
    val_unbiased = sparse_mx_to_torch_sparse_tensor(val_unbiased_mat).to(device)
    test_unbiased = sparse_mx_to_torch_sparse_tensor(test_unbiased_mat).to(device)
    
    if verbose:
        print("\n" + "="*70)
        print("Dataset loaded successfully!")
        print("="*70)
        print(f"Final shapes:")
        print(f"  train_biased: {train_biased.shape}, nnz: {train_biased._nnz():,}")
        print(f"  val_biased: {val_biased.shape}, nnz: {val_biased._nnz():,}")
        print(f"  test_biased: {test_biased.shape}, nnz: {test_biased._nnz():,}")
        print(f"  train_unbiased: {train_unbiased.shape}, nnz: {train_unbiased._nnz():,}")
        print(f"  val_unbiased: {val_unbiased.shape}, nnz: {val_unbiased._nnz():,}")
        print(f"  test_unbiased: {test_unbiased.shape}, nnz: {test_unbiased._nnz():,}")
        print("="*70 + "\n")
    
    return {
        'train_biased': train_biased,
        'val_biased': val_biased,
        'test_biased': test_biased,
        'train_unbiased': train_unbiased,
        'val_unbiased': val_unbiased,
        'test_unbiased': test_unbiased,
        'n_users': n_users,
        'n_items': n_items,
    }


def deduplicate_and_binarize_interactions(df, threshold=0.5):
    """
    Aggregate duplicated user-item pairs by mean(is_click), then binarize to {0,1}.
    
    **How duplicates are handled:**
    
    1. **Same user-item pair, multiple timestamps:**
       - Example: User 123 viewed Video 456 three times, with is_click = [0, 1, 0]
       - Aggregation: mean([0, 1, 0]) = 0.333
       - Binarization: 0.333 < 0.5 → final label = 0
    
    2. **Binarization threshold:**
       - mean(is_click) >= threshold → 1 (positive)
       - mean(is_click) < threshold → 0 (negative)
    
    3. **Rationale:**
       - Most duplicates are non-clicks (user repeatedly viewing but not clicking)
       - Threshold 0.5 means: "clicked at least half the time → positive"
       - Prevents a single click among many non-clicks from dominating
    
    Args:
        df: DataFrame with columns user_id, video_id, is_click
        threshold: >= threshold -> 1, else 0 (default 0.5)
    
    Returns:
        Deduplicated DataFrame with binary is_click values
    """
    if df.empty:
        return df[['user_id', 'video_id', 'is_click']].copy()
    
    # Count duplicates before deduplication
    n_before = len(df)
    n_unique_pairs = df.groupby(['user_id', 'video_id']).size()
    n_duplicates = (n_unique_pairs > 1).sum()
    
    # Aggregate by mean and binarize
    dedup = (
        df.groupby(['user_id', 'video_id'], as_index=False)['is_click']
          .mean()
    )
    dedup['is_click'] = (dedup['is_click'] >= threshold).astype(np.float32)
    
    # Print duplicate statistics if significant
    n_after = len(dedup)
    if n_duplicates > 0 and n_before > 10000:  # Only print for larger datasets
        print(f"    Deduplication: {n_before:,} -> {n_after:,} "
              f"(removed {n_before - n_after:,} duplicates, {n_duplicates:,} pairs had multiple interactions)")
    
    return dedup


def df_to_sparse_matrix(df, shape):
    """
    Convert DataFrame to sparse matrix
    
    Args:
        df: DataFrame with user_id, video_id, is_click
        shape: Matrix shape (n_users, n_items)
    
    Returns:
        scipy.sparse.csr_matrix
    """
    return sp.csr_matrix(
        (df['is_click'].values, (df['user_id'].values, df['video_id'].values)),
        shape=shape,
        dtype='float32'
    )


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """
    Convert scipy sparse matrix to PyTorch sparse tensor
    
    Args:
        sparse_mx: scipy sparse matrix
    
    Returns:
        torch.sparse_coo_tensor
    """
    sparse_mx = sparse_mx.tocoo()
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64)
    )
    values = torch.from_numpy(sparse_mx.data).float()
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse_coo_tensor(indices, values, shape)


def load_kuairand_for_mf(data_dir=None, device='cuda', verbose=True, biased_item_sample_ratio=1.0, random_seed=None):
    """
    Load KuaiRand-Pure dataset for matrix factorization
    Returns format compatible with load_dataset_new.load_dataset
    
    Args:
        data_dir: Data directory path
        device: torch device
        verbose: Print detailed information
    
    Returns:
        train, biased_val, unif_train, unif_val, test_unbiased, test_biased
    """
    data = load_kuairand(
        data_dir=data_dir,
        device=device,
        verbose=verbose,
        biased_item_sample_ratio=biased_item_sample_ratio,
        random_seed=random_seed
    )
    
    return (
        data['train_biased'],
        data['val_biased'],
        data['train_unbiased'],
        data['val_unbiased'],
        data['test_unbiased'],
        data['test_biased']
    )


# ========== Data Exploration Tools ==========
def explore_kuairand_data(data_dir=None):
    """
    Explore KuaiRand-Pure dataset statistics
    
    Args:
        data_dir: Data directory path
    """
    if data_dir is None:
        data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 
                               'datasets', 'KuaiRand-Pure', 'data')
    
    biased_file_early = os.path.join(data_dir, 'log_standard_4_08_to_4_21_pure.csv')
    biased_file_late = os.path.join(data_dir, 'log_standard_4_22_to_5_08_pure.csv')
    unbiased_file = os.path.join(data_dir, 'log_random_4_22_to_5_08_pure.csv')
    
    print("\n" + "="*70)
    print("KuaiRand-Pure Dataset Exploration")
    print("="*70)
    
    # Load data
    biased_df_early = pd.read_csv(biased_file_early)
    biased_df_late = pd.read_csv(biased_file_late)
    unbiased_df = pd.read_csv(unbiased_file)
    
    biased_df = pd.concat([biased_df_early, biased_df_late], ignore_index=True)
    
    print(f"\n[Basic Statistics]")
    print(f"Biased file 1 (Apr 8-21): {biased_file_early}")
    print(f"  Total samples: {len(biased_df_early):,}")
    
    print(f"\nBiased file 2 (Apr 22 - May 8): {biased_file_late}")
    print(f"  Total samples: {len(biased_df_late):,}")
    
    print(f"\nTotal biased samples: {len(biased_df):,}")
    
    print(f"\nUnbiased file: {unbiased_file}")
    print(f"  Total samples: {len(unbiased_df):,}")
    
    # User/item statistics
    print(f"\n[User/Item Statistics]")
    print(f"Biased (merged):")
    print(f"  Unique users: {biased_df['user_id'].nunique():,}")
    print(f"  Unique items: {biased_df['video_id'].nunique():,}")
    print(f"  Avg interactions per user: {len(biased_df) / biased_df['user_id'].nunique():.2f}")
    
    print(f"\nUnbiased:")
    print(f"  Unique users: {unbiased_df['user_id'].nunique():,}")
    print(f"  Unique items: {unbiased_df['video_id'].nunique():,}")
    print(f"  Avg interactions per user: {len(unbiased_df) / unbiased_df['user_id'].nunique():.2f}")
    
    # Date statistics
    print(f"\n[Date Distribution - Biased (merged)]")
    date_dist_biased = biased_df.groupby('date').size().sort_index()
    for date, count in date_dist_biased.items():
        print(f"  {date}: {count:,}")
    
    print(f"\n[Date Distribution - Unbiased]")
    date_dist_unbiased = unbiased_df.groupby('date').size().sort_index()
    for date, count in date_dist_unbiased.items():
        print(f"  {date}: {count:,}")
    
    # is_click distribution
    print(f"\n[is_click Distribution]")
    print(f"Biased (merged):")
    click_dist_biased = biased_df['is_click'].value_counts()
    total_biased = len(biased_df)
    for val, count in click_dist_biased.items():
        print(f"  is_click={val}: {count:,} ({count/total_biased*100:.2f}%)")
    
    print(f"\nUnbiased:")
    click_dist_unbiased = unbiased_df['is_click'].value_counts()
    total_unbiased = len(unbiased_df)
    for val, count in click_dist_unbiased.items():
        print(f"  is_click={val}: {count:,} ({count/total_unbiased*100:.2f}%)")
    
    print("="*70 + "\n")


# ========== Test Code ==========
if __name__ == '__main__':
    # Explore data
    explore_kuairand_data()
    
    # Test loading
    print("\nTesting load_kuairand()...")
    data = load_kuairand(device='cpu')
    
    print("\nTesting load_kuairand_for_mf()...")
    train, biased_val, unif_train, unif_val, test_unbiased, test_biased = \
        load_kuairand_for_mf(device='cpu')
    
    print("\nAll tests passed!")
