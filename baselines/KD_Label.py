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
            'batch_size': 1024,  # 验证/测试数据加载batch_size
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
            'weight_decay': 0
        }
    elif args.dataset == 'coat':
        args.training_args = {
            'batch_size': 128,  # 验证/测试数据加载batch_size
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
            'weight_decay': 0.01
        }
    elif 'KuaiRand' in args.dataset:
        args.training_args = {
            'batch_size': 2048,  # 验证/测试数据加载batch_size
            'epochs': 600,
            'patience': 80,
            'block_batch': [500, 10000]
        }
        args.base_model_args = {
            'emb_dim': 10,
            'learning_rate': 0.001,
            'weight_decay': 0.001
        }
        args.teacher_model_args = {
            'emb_dim': 10,
            'learning_rate': 0.01,
            'weight_decay': 0.001
        }
                

def train_and_eval(train_data, biased_val_data, unif_train_data, unif_val_data, 
                   test_data, biased_test_data, args, device='cuda'):
    """训练和评估函数 - 对齐mymodel.py"""
    
    # build data_loader
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

    # data shape
    n_user, n_item = train_data.shape

    # base model and its optimizer
    base_model = MF(n_user, n_item, dim=args.base_model_args['emb_dim'], dropout=0).to(device)
    base_optimizer = torch.optim.SGD(base_model.parameters(), lr=args.base_model_args['learning_rate'], weight_decay=0)

    # teacher model and its optimizer
    teacher_model = MF(n_user, n_item, dim=args.teacher_model_args['emb_dim'], dropout=0).to(device)
    teacher_optimizer = torch.optim.SGD(teacher_model.parameters(), lr=args.teacher_model_args['learning_rate'], weight_decay=0)
    
    # loss_criterion - 使用BCELoss
    criterion = nn.BCELoss(reduction='sum')

    # ===== 测试函数 =====
    def run_test(model, loader, name):
        """运行测试并打印结果"""
        preds, trues, users_list, items_list = [], [], [], []
        
        # 用于计算UAUC和UNDCG的字典
        ut_dict, pt_dict = {}, {}
        
        with torch.no_grad():
            for u, i, r in loader:
                pred = model(u.to(device), i.to(device))
                preds.append(pred)
                trues.append(r.to(device).float())
                users_list.append(u.to(device))
                items_list.append(i.to(device))
                
                # 收集每个用户的预测和真实标签（用于UAUC和UNDCG）
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
        
        # 所有数据集都计算全部指标
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
    
    # begin base model training
    stopping_args = Stop_args(patience=args.training_args['patience'], max_epochs=args.training_args['epochs'])
    early_stopping = EarlyStopping(base_model, **stopping_args)

    base_train_start = time.time()
    base_epoch_times = []

    for epo in range(early_stopping.max_epochs):
        epoch_start = time.time()
        training_loss = 0
        for u_batch_idx, users in enumerate(train_loader.User_loader): 
            for i_batch_idx, items in enumerate(train_loader.Item_loader): 
                # loss of training set
                base_model.train()
                users_train, items_train, y_train = train_loader.get_batch(users, items)
                y_hat = base_model(users_train, items_train)
                loss = criterion(y_hat, y_train.float()) + args.base_model_args['weight_decay'] * base_model.l2_norm(users_train, items_train)

                base_optimizer.zero_grad()
                loss.backward()
                base_optimizer.step()

                training_loss += loss.item()
        
        # 验证（使用无偏验证集）
        base_model.eval()
        with torch.no_grad():
            val_preds = []
            val_trues = []
            for u, i, r in unbiased_val_loader:
                pred = base_model(u.to(device), i.to(device))
                val_preds.append(pred)
                val_trues.append(r.to(device).float())
            
            val_preds = torch.cat(val_preds)
            val_trues = torch.cat(val_trues)
            val_results = utils.metrics.evaluate(val_preds, val_trues, ['BCE', 'AUC'])

        epoch_time = time.time() - epoch_start
        base_epoch_times.append(epoch_time)
        print(f'[Base] Epoch: {epo:3d} / {args.training_args["epochs"]}, Training Loss: {training_loss:.4f}, Val BCE: {val_results["BCE"]:.4f}, Val AUC: {val_results["AUC"]:.4f} | EpochTime: {epoch_time:.1f}s')

        if early_stopping.check([val_results['AUC']], epo):
            break

    base_train_total = time.time() - base_train_start
    base_avg_epoch = sum(base_epoch_times) / max(len(base_epoch_times), 1)
    # Load best base model
    print(f'Loading {early_stopping.best_epoch}th epoch')
    print(f'[Base Timing] Total: {base_train_total:.1f}s ({base_train_total/60:.2f}min) | Epochs: {len(base_epoch_times)} | Avg/epoch: {base_avg_epoch:.2f}s')
    base_model.load_state_dict(early_stopping.best_state)

    # ===== 测试base模型 =====
    run_test(base_model, test_loader, "Unbiased Test Data (Random) - Base Model")
    run_test(base_model, biased_test_loader, "Biased Test Data (Standard) - Base Model")

    # begin teacher model training
    stopping_args = Stop_args(patience=args.training_args['patience'], max_epochs=args.training_args['epochs'])
    early_stopping = EarlyStopping(teacher_model, **stopping_args)

    teacher_train_start = time.time()
    teacher_epoch_times = []

    for epo in range(early_stopping.max_epochs):
        epoch_start = time.time()
        training_loss = 0
        for u_batch_idx, users in enumerate(unif_loader.User_loader): 
            for i_batch_idx, items in enumerate(unif_loader.Item_loader): 
                # loss of training set
                teacher_model.train()
                users_train, items_train, y_train = unif_loader.get_batch(users, items)
                y_hat = teacher_model(users_train, items_train)
                loss = criterion(y_hat, y_train.float()) + args.teacher_model_args['weight_decay'] * teacher_model.l2_norm(users_train, items_train)

                teacher_optimizer.zero_grad()
                loss.backward()
                teacher_optimizer.step()

                training_loss += loss.item()
        
        # 验证（使用无偏验证集）
        teacher_model.eval()
        with torch.no_grad():
            val_preds = []
            val_trues = []
            for u, i, r in unbiased_val_loader:
                pred = teacher_model(u.to(device), i.to(device))
                val_preds.append(pred)
                val_trues.append(r.to(device).float())
            
            val_preds = torch.cat(val_preds)
            val_trues = torch.cat(val_trues)
            val_results = utils.metrics.evaluate(val_preds, val_trues, ['BCE', 'AUC'])

        epoch_time = time.time() - epoch_start
        teacher_epoch_times.append(epoch_time)
        print(f'[Teacher] Epoch: {epo:3d} / {args.training_args["epochs"]}, Training Loss: {training_loss:.4f}, Val BCE: {val_results["BCE"]:.4f}, Val AUC: {val_results["AUC"]:.4f} | EpochTime: {epoch_time:.1f}s')

        if early_stopping.check([val_results['AUC']], epo):
            break

    teacher_train_total = time.time() - teacher_train_start
    teacher_avg_epoch = sum(teacher_epoch_times) / max(len(teacher_epoch_times), 1)
    # Load best teacher model
    print(f'Loading {early_stopping.best_epoch}th epoch')
    print(f'[Teacher Timing] Total: {teacher_train_total:.1f}s ({teacher_train_total/60:.2f}min) | Epochs: {len(teacher_epoch_times)} | Avg/epoch: {teacher_avg_epoch:.2f}s')
    teacher_model.load_state_dict(early_stopping.best_state)

    # ===== 测试teacher模型 =====
    run_test(teacher_model, test_loader, "Unbiased Test Data (Random) - Teacher Model")
    run_test(teacher_model, biased_test_loader, "Biased Test Data (Standard) - Teacher Model")

    # base model re-tune with teacher's soft labels
    stopping_args = Stop_args(patience=args.training_args['patience'], max_epochs=args.training_args['epochs'])
    early_stopping = EarlyStopping(base_model, **stopping_args)
    
    # 软标签权重系数（控制原始标签和软标签的比例）
    soft_label_weight = 0.5

    retune_train_start = time.time()
    retune_epoch_times = []

    for epo in range(early_stopping.max_epochs):
        epoch_start = time.time()
        training_loss = 0
        for u_batch_idx, users in enumerate(train_loader.User_loader): 
            for i_batch_idx, items in enumerate(train_loader.Item_loader): 
                # loss of training set
                base_model.train()
                users_train, items_train, y_train = train_loader.get_batch(users, items)
                y_hat = base_model(users_train, items_train)
                y_res = teacher_model(users_train, items_train) 
                y_res = (y_res - torch.min(y_res)) / (torch.max(y_res) - torch.min(y_res))
                # 使用加权平均融合原始标签和软标签，确保标签在[0,1]范围内
                y_train = (1 - soft_label_weight) * y_train + soft_label_weight * y_res
                loss = criterion(y_hat, y_train.float()) + args.base_model_args['weight_decay'] * base_model.l2_norm(users_train, items_train)

                base_optimizer.zero_grad()
                loss.backward()
                base_optimizer.step()

                training_loss += loss.item()
        
        # 验证（使用无偏验证集）
        base_model.eval()
        with torch.no_grad():
            val_preds = []
            val_trues = []
            for u, i, r in unbiased_val_loader:
                pred = base_model(u.to(device), i.to(device))
                val_preds.append(pred)
                val_trues.append(r.to(device).float())
            
            val_preds = torch.cat(val_preds)
            val_trues = torch.cat(val_trues)
            val_results = utils.metrics.evaluate(val_preds, val_trues, ['BCE', 'AUC'])

        epoch_time = time.time() - epoch_start
        retune_epoch_times.append(epoch_time)
        print(f'[Retune] Epoch: {epo:3d} / {args.training_args["epochs"]}, Training Loss: {training_loss:.4f}, Val BCE: {val_results["BCE"]:.4f}, Val AUC: {val_results["AUC"]:.4f} | EpochTime: {epoch_time:.1f}s')

        if early_stopping.check([val_results['AUC']], epo):
            break

    retune_train_total = time.time() - retune_train_start
    retune_avg_epoch = sum(retune_epoch_times) / max(len(retune_epoch_times), 1)
    pipeline_total = base_train_total + teacher_train_total + retune_train_total
    # Load best re-tuned base model
    print(f'Loading {early_stopping.best_epoch}th epoch')
    print(f'[Retune Timing] Total: {retune_train_total:.1f}s ({retune_train_total/60:.2f}min) | Epochs: {len(retune_epoch_times)} | Avg/epoch: {retune_avg_epoch:.2f}s')
    base_model.load_state_dict(early_stopping.best_state)

    # 汇总训练时间
    print('\n' + '='*60)
    print('Training Speed Summary (KD_Label)')
    print('='*60)
    print(f"  {'Stage':<25} {'Total(s)':>10} {'Total(min)':>12} {'Epochs':>8} {'Avg/epoch(s)':>14}")
    print(f"  {'-'*25} {'-'*10} {'-'*12} {'-'*8} {'-'*14}")
    print(f"  {'Base Model':<25} {base_train_total:>10.1f} {base_train_total/60:>12.2f} {len(base_epoch_times):>8} {base_avg_epoch:>14.2f}")
    print(f"  {'Teacher Model':<25} {teacher_train_total:>10.1f} {teacher_train_total/60:>12.2f} {len(teacher_epoch_times):>8} {teacher_avg_epoch:>14.2f}")
    print(f"  {'Retune (Base+SoftLabel)':<25} {retune_train_total:>10.1f} {retune_train_total/60:>12.2f} {len(retune_epoch_times):>8} {retune_avg_epoch:>14.2f}")
    print(f"  {'-'*25} {'-'*10} {'-'*12} {'-'*8} {'-'*14}")
    print(f"  {'Total Pipeline':<25} {pipeline_total:>10.1f} {pipeline_total/60:>12.2f} {len(base_epoch_times)+len(teacher_epoch_times)+len(retune_epoch_times):>8} {'':>14}")
    print('='*60)

    # ===== 测试re-tuned base模型 =====
    run_test(base_model, test_loader, "Unbiased Test Data (Random) - Re-tuned Base Model")
    run_test(base_model, biased_test_loader, "Biased Test Data (Standard) - Re-tuned Base Model")


if __name__ == "__main__": 
    args = arguments.parse_args()
    para(args)
    setup_seed(args.seed)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 加载数据 - 对齐mymodel.py
    print(f"\nLoading dataset: {args.dataset}")
    if args.dataset == 'KuaiRand':
        # 使用新的KuaiRand数据加载器（基于时间划分）
        data_dict = load_kuairand.load_kuairand(device=device, verbose=True, random_seed=args.seed)
        train_data = data_dict['train_biased']
        biased_val_data = data_dict['val_biased']
        unif_train_data = data_dict['train_unbiased']
        unif_val_data = data_dict['val_unbiased']
        test_data = data_dict['test_unbiased']
        biased_test_data = data_dict['test_biased']
    else:
        # 其他数据集使用原有加载器（全局随机划分）
        train_data, biased_val_data, unif_train_data, unif_val_data, test_data, biased_test_data = \
            load_dataset_custom.load_dataset(data_name=args.dataset, unif_ratio=0.5, seed=args.seed, device=device)
        
        print(f"  Biased Train: {train_data._nnz()} samples")
        print(f"  Biased Val:   {biased_val_data._nnz()} samples")
        print(f"  Unbiased Train: {unif_train_data._nnz()} samples")
        print(f"  Unbiased Val:   {unif_val_data._nnz()} samples")
        print(f"  Biased Test:  {biased_test_data._nnz()} samples")
        print(f"  Unbiased Test: {test_data._nnz()} samples")
    
    # 训练和评估
    train_and_eval(
        train_data, biased_val_data, unif_train_data, unif_val_data, 
        test_data, biased_test_data, args, device=device
    )
