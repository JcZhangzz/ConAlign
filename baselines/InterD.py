"""
InterD: Integrated Debiasing Model
结合MF和AutoDebias的集成去偏模型

参考论文: Interpolating Training and Debiasing for Recommender Systems
"""

import os
import time
import numpy as np
import random
import torch
import torch.nn as nn
from model import MF, MetaMF, ThreeLinear, OneLinear
import arguments
import utils.load_dataset_new as load_dataset_custom
import utils.load_kuairand as load_kuairand
import utils.data_loader
import utils.metrics
import utils.early_stop
from utils.early_stop import EarlyStopping, Stop_args

def setup_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available(): 
        torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def para(args): 
    if args.dataset == 'yahooR3': 
        args.training_args = {'batch_size': 1024, 'epochs': 500, 'patience': 60, 'block_batch': [6000, 500]}
        args.InterD_model_args = {"emb_dim": 10, "learning_rate": 0.01, "weight_decay": 1}
        args.MF_model_args = {"emb_dim": 10, "learning_rate": 1e-3, "weight_decay": 0.1, 'patience': 80}
        args.Auto_model_args = {"emb_dim": 10, "learning_rate": 0.0001, "weight_decay": 10, 'imputaion_lambda': 0.05, 'epoch': 500}
        args.weight1_model_args = {"learning_rate": 0.001, "weight_decay": 0.01}
        args.weight2_model_args = {"learning_rate": 0.0001, "weight_decay": 0.01}
        args.imputation_model_args = {"learning_rate": 0.01, "weight_decay": 0.0001}
        args.gama = 0.17
        args.gama2 = 1.25
        args.beta = 0.01
    elif args.dataset == 'coat':
        args.training_args = {'batch_size': 128, 'epochs': 500, 'patience': 60, 'block_batch': [64, 64]}
        args.InterD_model_args = {"emb_dim": 10, "learning_rate": 0.01, "weight_decay": 0}
        args.MF_model_args = {"emb_dim": 10, "learning_rate": 0.001, "weight_decay": 0, 'patience': 80}
        args.Auto_model_args = {"emb_dim": 10, "learning_rate": 1e-2, "weight_decay": 0, 'imputaion_lambda': 0.01, 'epoch': 500}
        args.weight1_model_args = {"learning_rate": 1e-4, "weight_decay": 1e-6}
        args.weight2_model_args = {"learning_rate": 1e-4, "weight_decay": 0}
        args.imputation_model_args = {"learning_rate": 0.001, "weight_decay": 0.01}
        args.gama = 0.05
        args.gama2 = 1
        args.beta = 0.05
    elif 'KuaiRand' in args.dataset:
        args.training_args = {'batch_size': 1024, 'epochs': 500, 'patience': 40, 'block_batch': [500, 10000]}
        args.InterD_model_args = {"emb_dim": 10, "learning_rate": 0.01, "weight_decay": 1e-3}
        args.MF_model_args = {"emb_dim": 10, "learning_rate": 0.01, "weight_decay": 1e-1, 'patience': 60}
        args.Auto_model_args = {"emb_dim": 10, "learning_rate": 0.001, "weight_decay": 1e-3, 'imputaion_lambda': 0.01, 'epoch': 500}
        args.weight1_model_args = {"learning_rate": 0.001, "weight_decay": 1e-3}
        args.weight2_model_args = {"learning_rate": 1e-3, "weight_decay": 1e-3}
        args.imputation_model_args = {"learning_rate": 1e-3, "weight_decay": 1e-4}
        args.gama = 0.05
        args.gama2 = 1
        args.beta = 0.05


# ============================================================================
#                          通用测试函数
# ============================================================================

def run_test(model, loader, name, model_name, device):
    """
    通用测试函数
    
    Args:
        model: 模型或模型元组（AutoDebias的情况）
        loader: 数据加载器
        name: 测试集名称（如 "Unbiased (Random)"）
        model_name: 模型名称（如 "MF", "AutoDebias", "InterD"）
        device: 设备
    """
    preds, trues, us, is_ = [], [], [], []
    
    # 用于计算UAUC和UNDCG的字典
    ut_dict, pt_dict = {}, {}
    
    with torch.no_grad():
        for u, i, r in loader:
            # 根据模型类型选择正确的预测方式
            if model_name == "AutoDebias":
                # AutoDebias返回的是base_model (CF_model)
                pred = model[0](u.to(device), i.to(device))
            else:
                pred = model(u.to(device), i.to(device))
            
            preds.append(pred)
            trues.append(r.to(device).float())
            us.append(u.to(device))
            is_.append(i.to(device))
            
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
    
    preds_cat = torch.cat(preds)
    trues_cat = torch.cat(trues)
    users_cat = torch.cat(us)
    items_cat = torch.cat(is_)
    
    # 所有数据集都计算全部指标（已优化内存使用）
    metric_list = ['BCE', 'AUC', 'UAUC', 'UNDCG', 'Recall_Precision_NDCG@']
    
    res = utils.metrics.evaluate(preds_cat, trues_cat, metric_list,
                                 users=users_cat, items=items_cat,
                                 UAUC=(ut_dict, pt_dict),
                                 UNDCG=(ut_dict, pt_dict))
    print(f"\n[{model_name} - {name} Test Results]:")
    for k, v in res.items():
        print(f"  {k}: {v:.4f}")
    return res


# ============================================================================
#                          阶段1: 训练MF模型
# ============================================================================

def train_MF_model(bias_train, unif_train, bias_val, unif_val, bias_test, unif_test, n_user, n_item, args, device):
    """训练基础的MF模型（使用有偏+无偏训练数据）"""
    print('='*50)
    print('Training MF Model (Baseline)')
    print('='*50)
    mf_train_start = time.time()
    
    # 合并有偏和无偏训练数据
    combined_train = bias_train + unif_train
    
    # 创建数据加载器
    train_loader = utils.data_loader.Block(
        combined_train, 
        u_batch_size=args.training_args['block_batch'][0],
        i_batch_size=args.training_args['block_batch'][1],
        device=device
    )
    val_loader = utils.data_loader.DataLoader(
        utils.data_loader.Interactions(unif_val),
        batch_size=args.training_args['batch_size']
    )
    
    # 创建模型
    mf_model = MF(n_user, n_item, dim=args.MF_model_args['emb_dim']).to(device)
    optimizer = torch.optim.SGD(mf_model.parameters(), lr=args.MF_model_args['learning_rate'])
    
    # 早停
    early_stopping = EarlyStopping(
        mf_model, 
        stop_varnames=[utils.early_stop.StopVariable.AUC],
        patience=args.MF_model_args['patience'], 
        max_epochs=args.training_args['epochs']*10
    )
    
    criterion = nn.BCELoss(reduction='sum')
    
    for epo in range(args.training_args['epochs']*10):
        # 训练
        mf_model.train()
        total_l2_loss = 0.0
        total_batches = 0
        for u_batch_idx, u_batch in enumerate(train_loader.User_loader):
            for i_batch_idx, i_batch in enumerate(train_loader.Item_loader):
                users_train, items_train, y_train = train_loader.get_batch(u_batch, i_batch, device=device)
                
                optimizer.zero_grad()
                pred = mf_model(users_train, items_train)
                loss = criterion(pred, y_train.float()) + args.MF_model_args['weight_decay'] * mf_model.l2_norm(users_train, items_train)
                total_l2_loss += mf_model.l2_norm(users_train, items_train).item()
                loss.backward()
                optimizer.step()
                total_batches += 1
        
        # 验证
        mf_model.eval()
        with torch.no_grad():
            val_preds = []
            val_trues = []
            for u, i, r in val_loader:
                val_preds.append(mf_model(u.to(device), i.to(device)))
                val_trues.append(r.to(device).float())
            val_res = utils.metrics.evaluate(torch.cat(val_preds), torch.cat(val_trues), ['BCE', 'AUC'])
        
        avg_l2 = total_l2_loss / max(total_batches, 1)
        print(f"[MF] Epoch {epo:3d} | Val BCE: {val_res['BCE']:.4f} AUC: {val_res['AUC']:.4f} | L2: {avg_l2:.2f}")
        
        if epo >= 20 and early_stopping.check([val_res['AUC']], epo):
            break
    
    mf_model.load_state_dict(early_stopping.best_state)
    mf_train_time = time.time() - mf_train_start
    print(f"MF Model Loaded (Best Epoch: {early_stopping.best_epoch}) | Train Time: {mf_train_time:.1f}s")
    
    # 测试MF模型
    print('\n' + '='*70)
    print('Testing MF Model (Baseline)')
    print('='*70)
    
    # 创建测试数据加载器
    test_loader = utils.data_loader.DataLoader(
        utils.data_loader.Interactions(unif_test),
        batch_size=args.training_args['batch_size']
    )
    biased_test_loader = utils.data_loader.DataLoader(
        utils.data_loader.Interactions(bias_test),
        batch_size=args.training_args['batch_size']
    )
    
    mf_model.eval()
    mf_test_start = time.time()
    run_test(mf_model, test_loader, "Unbiased (Random)", "MF", device)
    run_test(mf_model, biased_test_loader, "Biased (Standard)", "MF", device)
    mf_test_time = time.time() - mf_test_start
    mf_total_time = mf_train_time + mf_test_time
    print(f"\n[MF Timing] Train: {mf_train_time:.1f}s | Test: {mf_test_time:.1f}s | Total: {mf_total_time:.1f}s")

    return mf_model, mf_total_time


def train_AutoDebias_model(bias_train, unif_train, bias_val, unif_val, bias_test, unif_test, n_user, n_item, args, device):
    """训练AutoDebias模型"""
    print('='*50)
    print('Training AutoDebias Model')
    print('='*50)
    auto_train_start = time.time()
    
    # 准备数据
    users_unif = unif_train._indices()[0]
    items_unif = unif_train._indices()[1]
    y_unif = unif_train._values().float()
    
    # 不使用dense矩阵，避免将缺失与label=0混淆
    
    # 创建数据加载器
    train_loader = utils.data_loader.Block(
        bias_train,
        u_batch_size=args.training_args['block_batch'][0],
        i_batch_size=args.training_args['block_batch'][1],
        device=device
    )
    val_loader = utils.data_loader.DataLoader(
        utils.data_loader.Interactions(unif_val),
        batch_size=args.training_args['batch_size']
    )
    
    # 创建模型
    base_model = MetaMF(n_user, n_item, dim=args.Auto_model_args['emb_dim']).to(device)
    weight1_model = ThreeLinear(n_user, n_item, 2).to(device)
    weight2_model = ThreeLinear(n_user, n_item, 2).to(device)
    imputation_model = OneLinear(3).to(device)
    
    # 优化器
    base_optimizer = torch.optim.SGD(base_model.params(), lr=args.Auto_model_args['learning_rate'])
    weight1_optimizer = torch.optim.Adam(weight1_model.parameters(), lr=args.weight1_model_args['learning_rate'], weight_decay=args.weight1_model_args['weight_decay'])
    weight2_optimizer = torch.optim.Adam(weight2_model.parameters(), lr=args.weight2_model_args['learning_rate'], weight_decay=args.weight2_model_args['weight_decay'])
    imputation_optimizer = torch.optim.Adam(imputation_model.parameters(), lr=args.imputation_model_args['learning_rate'], weight_decay=args.imputation_model_args['weight_decay'])
    
    # 损失函数
    sum_criterion = nn.BCELoss(reduction='sum')
    none_criterion = nn.BCELoss(reduction='none')
    
    # 早停
    early_stopping = EarlyStopping(
        base_model,
        stop_varnames=[utils.early_stop.StopVariable.AUC],
        patience=60,
        max_epochs=args.Auto_model_args['epoch']
    )
    
    for epo in range(args.Auto_model_args['epoch']):
        # 训练
        base_model.train()
        total_l2_loss = 0.0
        total_batches = 0
        
        for u_batch_idx, u_batch in enumerate(train_loader.User_loader):
            for i_batch_idx, i_batch in enumerate(train_loader.Item_loader):
                users_train, items_train, y_train = train_loader.get_batch(u_batch, i_batch, device=device)
                y_train = y_train.float()
                
                # 笛卡尔积
                all_pair = torch.cartesian_prod(u_batch, i_batch)
                u_all, i_all = all_pair[:, 0], all_pair[:, 1]
                
                # 计算权重和填充值
                weight1_model.train()
                weight2_model.train()
                imputation_model.train()
                
                # 构建二维状态图：缺失=-1，观测负样本=0，观测正样本=1
                local_status_map = torch.full((len(u_batch), len(i_batch)), -1.0, device=device)
                for ut, it, yt in zip(users_train, items_train, y_train):
                    u_idx = (u_batch == ut).nonzero(as_tuple=True)[0]
                    i_idx = (i_batch == it).nonzero(as_tuple=True)[0]
                    local_status_map[u_idx, i_idx] = yt
                flat_status_all = local_status_map.flatten().long()
                obs_status_all = (flat_status_all != -1).long()
                impu_indices_all = (flat_status_all + 1).clamp(0, 2)

                w1 = torch.exp(weight1_model(users_train, items_train, (y_train == 1) * 1) / 5)
                w2 = torch.exp(weight2_model(u_all, i_all, obs_status_all) / 5)
                impu_f = torch.sigmoid(imputation_model(impu_indices_all))
                
                # 元学习虚拟步
                one_step_model = MetaMF(n_user, n_item, dim=args.Auto_model_args['emb_dim']).to(device)
                one_step_model.load_state_dict(base_model.state_dict())
                
                l_obs = torch.sum(none_criterion(one_step_model(users_train, items_train), y_train) * w1)
                l_all = torch.sum(none_criterion(one_step_model(u_all, i_all), impu_f) * w2)
                l_total = l_obs + args.Auto_model_args['imputaion_lambda'] * l_all + args.Auto_model_args['weight_decay'] * one_step_model.l2_norm(u_all, i_all)
                
                grads = torch.autograd.grad(l_total, one_step_model.params(), create_graph=True)
                one_step_model.update_params(args.Auto_model_args['learning_rate'], source_params=grads)
                
                # 元优化
                loss_meta = sum_criterion(one_step_model(users_unif, items_unif), y_unif)
                weight1_optimizer.zero_grad()
                weight2_optimizer.zero_grad()
                imputation_optimizer.zero_grad()
                loss_meta.backward()
                if epo >= 20:
                    weight1_optimizer.step()
                    weight2_optimizer.step()
                imputation_optimizer.step()
                
                # 真实更新
                base_optimizer.zero_grad()
                
                weight1_model.train()
                weight2_model.train()
                imputation_model.train()
                
                final_w1 = torch.exp(weight1_model(users_train, items_train, (y_train == 1) * 1) / 5)
                final_w2 = torch.exp(weight2_model(u_all, i_all, obs_status_all) / 5)
                final_impu = torch.sigmoid(imputation_model(impu_indices_all))
                
                loss = (torch.sum(none_criterion(base_model(users_train, items_train), y_train) * final_w1) +
                        args.Auto_model_args['imputaion_lambda'] * torch.sum(none_criterion(base_model(u_all, i_all), final_impu) * final_w2) +
                        args.Auto_model_args['weight_decay'] * base_model.l2_norm(u_all, i_all))
                total_l2_loss += base_model.l2_norm(u_all, i_all).item()
                loss.backward()
                base_optimizer.step()
                total_batches += 1
        
        # 验证
        base_model.eval()
        with torch.no_grad():
            val_preds = []
            val_trues = []
            for u, i, r in val_loader:
                val_preds.append(base_model(u.to(device), i.to(device)))
                val_trues.append(r.to(device).float())
            val_res = utils.metrics.evaluate(torch.cat(val_preds), torch.cat(val_trues), ['BCE', 'AUC'])
        
        avg_l2 = total_l2_loss / max(total_batches, 1)
        print(f"[AutoDebias] Epoch {epo:3d} | Val BCE: {val_res['BCE']:.4f} AUC: {val_res['AUC']:.4f} | L2: {avg_l2:.2f}")
        
        if epo >= 50 and early_stopping.check([val_res['AUC']], epo):
            break
    
    base_model.load_state_dict(early_stopping.best_state)
    auto_train_time = time.time() - auto_train_start
    print(f"AutoDebias Model Loaded (Best Epoch: {early_stopping.best_epoch}) | Train Time: {auto_train_time:.1f}s")
    
    # 测试AutoDebias模型
    print('\n' + '='*70)
    print('Testing AutoDebias Model (Debiased)')
    print('='*70)
    
    # 创建测试数据加载器
    test_loader = utils.data_loader.DataLoader(
        utils.data_loader.Interactions(unif_test),
        batch_size=args.training_args['batch_size']
    )
    biased_test_loader = utils.data_loader.DataLoader(
        utils.data_loader.Interactions(bias_test),
        batch_size=args.training_args['batch_size']
    )
    
    base_model.eval()
    auto_test_start = time.time()
    AutoDebias_models = (base_model, weight1_model, weight2_model, imputation_model)
    run_test(AutoDebias_models, test_loader, "Unbiased (Random)", "AutoDebias", device)
    run_test(AutoDebias_models, biased_test_loader, "Biased (Standard)", "AutoDebias", device)
    auto_test_time = time.time() - auto_test_start
    auto_total_time = auto_train_time + auto_test_time
    print(f"\n[AutoDebias Timing] Train: {auto_train_time:.1f}s | Test: {auto_test_time:.1f}s | Total: {auto_total_time:.1f}s")

    return base_model, weight1_model, weight2_model, imputation_model, auto_total_time


def train_InterD_model(bias_train, unif_train, bias_val, unif_val, bias_test, unif_test, 
                       MF_model, AutoDebias_models, n_user, n_item, args, device):
    """训练InterD集成模型"""
    print('='*50)
    print('Training InterD Model')
    print('='*50)
    interd_train_start = time.time()
    
    # 解包AutoDebias模型
    CF_model = AutoDebias_models[0]
    weight1_model = AutoDebias_models[1]
    weight2_model = AutoDebias_models[2]
    imputation_model = AutoDebias_models[3]
    F_model = MF_model
    
    # 合并训练数据
    combined_train = bias_train + unif_train
    
    # 不使用dense矩阵，避免将缺失与label=0混淆
    
    # 创建数据加载器
    train_loader = utils.data_loader.Block(
        combined_train,
        u_batch_size=args.training_args['block_batch'][0],
        i_batch_size=args.training_args['block_batch'][1],
        device=device
    )
    val_loader = utils.data_loader.DataLoader(
        utils.data_loader.Interactions(unif_val),
        batch_size=args.training_args['batch_size']
    )
    test_loader = utils.data_loader.DataLoader(
        utils.data_loader.Interactions(unif_test),
        batch_size=args.training_args['batch_size']
    )
    biased_test_loader = utils.data_loader.DataLoader(
        utils.data_loader.Interactions(bias_test),
        batch_size=args.training_args['batch_size']
    )
    
    # 创建InterD模型
    CFF_model = MF(n_user, n_item, dim=args.InterD_model_args['emb_dim']).to(device)
    # 从CF模型初始化
    CFF_model.load_state_dict(CF_model.state_dict())
    
    optimizer = torch.optim.SGD(CFF_model.parameters(), lr=args.InterD_model_args['learning_rate'])
    criterion = nn.BCELoss(reduction='sum')
    
    # 早停
    early_stopping = EarlyStopping(
        CFF_model,
        stop_varnames=[utils.early_stop.StopVariable.AUC],
        patience=args.training_args['patience'],
        max_epochs=args.training_args['epochs']
    )
    
    for epo in range(args.training_args['epochs']):
        # 训练
        CFF_model.train()
        total_l2_loss = 0.0
        total_batches = 0
        
        for u_batch_idx, u_batch in enumerate(train_loader.User_loader):
            for i_batch_idx, i_batch in enumerate(train_loader.Item_loader):
                users_train, items_train, y_train = train_loader.get_batch(u_batch, i_batch, device=device)
                y_train = y_train.float()
                
                # 构建二维状态图：缺失=-1，观测负样本=0，观测正样本=1
                local_status_map = torch.full((len(u_batch), len(i_batch)), -1.0, device=device)
                for ut, it, yt in zip(users_train, items_train, y_train):
                    u_idx = (u_batch == ut).nonzero(as_tuple=True)[0]
                    i_idx = (i_batch == it).nonzero(as_tuple=True)[0]
                    local_status_map[u_idx, i_idx] = yt

                # 观测样本状态与填充值索引
                obs_status_train = torch.ones_like(y_train, dtype=torch.long)
                impu_indices_train = (y_train.long() + 1).clamp(0, 2)

                # 获取两个模型的预测
                with torch.no_grad():
                    CF_pred = CF_model(users_train, items_train)
                    F_pred = F_model(users_train, items_train)
                
                # 计算weight1（观测数据权重）
                weight1 = torch.exp(weight1_model(users_train, items_train, (y_train == 1) * 1) / 5)
                
                # 计算weight2（填充权重）
                weight2 = torch.exp(weight2_model(users_train, items_train, obs_status_train) / 5)
                
                # 计算填充值
                impu_train = torch.sigmoid(imputation_model(impu_indices_train))
                
                # 计算Auto_loss（包含观测损失和填充损失）
                Auto_loss = nn.BCELoss(reduction='none')(CF_pred, y_train)
                cost_impu = nn.BCELoss(reduction='none')(CF_pred, impu_train)
                CF_loss = Auto_loss * weight1 + cost_impu * weight2
                
                # 计算F_loss
                F_loss = nn.BCELoss(reduction='none')(F_pred, y_train)
                
                # ========== 缺失数据的填充训练（Imputation train）==========
                # 获取笛卡尔积（所有可能的用户-物品对）
                all_pair = torch.cartesian_prod(u_batch, i_batch)
                users_all, items_all = all_pair[:, 0], all_pair[:, 1]
                flat_status_all = local_status_map.flatten().long()
                
                # 获取所有对的预测
                with torch.no_grad():
                    CF_pred_A = CF_model(users_all, items_all)
                    F_pred_A = F_model(users_all, items_all)
                
                # 当前模型对所有对的预测
                y_hat_obsA = CFF_model(users_all, items_all)
                
                # 计算损失
                Loss_FA = nn.BCELoss(reduction='none')(y_hat_obsA, F_pred_A)
                
                # 计算填充相关的权重和值
                obs_status_all = (flat_status_all != -1).long()
                impu_indices_all = (flat_status_all + 1).clamp(0, 2)
                weight2A = torch.exp(weight2_model(users_all, items_all, obs_status_all) / 5)
                impu_trainA = torch.sigmoid(imputation_model(impu_indices_all))
                Loss_CFA = nn.BCELoss(reduction='none')(impu_trainA, y_hat_obsA) * weight2A
                
                # 计算因果融合权重（针对缺失数据）
                W_CFA = torch.pow(Loss_FA, args.gama2) / (torch.pow(Loss_CFA, args.gama2) + torch.pow(Loss_FA, args.gama2))
                W_FA = torch.pow(Loss_CFA, args.gama2) / (torch.pow(Loss_CFA, args.gama2) + torch.pow(Loss_FA, args.gama2))
                y_causal_trainA = W_CFA * CF_pred_A + W_FA * F_pred_A
                
                # 计算缺失数据的损失（只对缺失位置计算）
                y_hat_obs_A = CFF_model(users_all, items_all)
                loss_A_per_sample = nn.BCELoss(reduction='none')(y_hat_obs_A, y_causal_trainA.detach())
                imp_mask = (flat_status_all == -1).float()  # 缺失位置的mask
                loss_A = torch.sum(loss_A_per_sample * imp_mask)
                
                # ========== 观测数据的因果融合 ==========
                # 因果融合权重（针对观测数据）
                W_CF = torch.pow(F_loss, args.gama) / (torch.pow(CF_loss, args.gama) + torch.pow(F_loss, args.gama))
                W_F = torch.pow(CF_loss, args.gama) / (torch.pow(CF_loss, args.gama) + torch.pow(F_loss, args.gama))
                y_causal_train = W_CF * CF_pred + W_F * F_pred
                
                # 训练InterD
                optimizer.zero_grad()
                pred = CFF_model(users_train, items_train)
                cost_obs = criterion(pred, y_causal_train.detach())
                
                # 总损失 = 观测数据损失 + beta*缺失数据损失 + L2正则
                loss = cost_obs + args.beta * loss_A + args.InterD_model_args['weight_decay'] * CFF_model.l2_norm(users_all, items_all)
                total_l2_loss += CFF_model.l2_norm(users_all, items_all).item()
                loss.backward()
                optimizer.step()
                total_batches += 1
        
        # 验证
        CFF_model.eval()
        with torch.no_grad():
            val_preds = []
            val_trues = []
            for u, i, r in val_loader:
                val_preds.append(CFF_model(u.to(device), i.to(device)))
                val_trues.append(r.to(device).float())
            val_res = utils.metrics.evaluate(torch.cat(val_preds), torch.cat(val_trues), ['BCE', 'AUC'])
        
        avg_l2 = total_l2_loss / max(total_batches, 1)
        print(f"[InterD] Epoch {epo:3d} | Val BCE: {val_res['BCE']:.4f} AUC: {val_res['AUC']:.4f} | L2: {avg_l2:.2f}")
        
        if early_stopping.check([val_res['AUC']], epo):
            break
    
    CFF_model.load_state_dict(early_stopping.best_state)
    interd_train_time = time.time() - interd_train_start
    print(f"InterD Model Loaded (Best Epoch: {early_stopping.best_epoch}) | Train Time: {interd_train_time:.1f}s")
    
    # 测试InterD模型
    print('\n' + '='*70)
    print('Testing InterD Model (Integrated)')
    print('='*70)
    
    CFF_model.eval()
    interd_test_start = time.time()
    run_test(CFF_model, test_loader, "Unbiased (Random)", "InterD", device)
    run_test(CFF_model, biased_test_loader, "Biased (Standard)", "InterD", device)
    interd_test_time = time.time() - interd_test_start
    interd_total_time = interd_train_time + interd_test_time
    print(f"\n[InterD Timing] Train: {interd_train_time:.1f}s | Test: {interd_test_time:.1f}s | Total: {interd_total_time:.1f}s")

    return interd_total_time


def train_and_eval(train_data, biased_val_data, unif_train_data, unif_val_data, 
                   test_data, biased_test_data, args, device='cuda'):
    """InterD完整训练流程"""
    n_user, n_item = train_data.shape
    pipeline_start = time.time()

    # 阶段1: 训练MF模型
    MF_model, mf_time = train_MF_model(
        train_data, unif_train_data, biased_val_data, unif_val_data,
        biased_test_data, test_data,
        n_user, n_item, args, device
    )
    
    # 阶段2: 训练AutoDebias模型
    *AutoDebias_models_list, auto_time = train_AutoDebias_model(
        train_data, unif_train_data, biased_val_data, unif_val_data,
        biased_test_data, test_data,
        n_user, n_item, args, device
    )
    AutoDebias_models = tuple(AutoDebias_models_list)

    # 阶段3: 训练InterD模型
    interd_time = train_InterD_model(
        train_data, unif_train_data, biased_val_data, unif_val_data,
        biased_test_data, test_data,
        MF_model, AutoDebias_models, n_user, n_item, args, device
    )

    pipeline_total = time.time() - pipeline_start
    print('\n' + '='*70)
    print('Training Speed Summary')
    print('='*70)
    print(f"  {'Stage':<20} {'Time (s)':>10} {'Time (min)':>12} {'% of Total':>12}")
    print(f"  {'-'*20} {'-'*10} {'-'*12} {'-'*12}")
    print(f"  {'MF':<20} {mf_time:>10.1f} {mf_time/60:>12.2f} {mf_time/pipeline_total*100:>11.1f}%")
    print(f"  {'AutoDebias':<20} {auto_time:>10.1f} {auto_time/60:>12.2f} {auto_time/pipeline_total*100:>11.1f}%")
    print(f"  {'InterD':<20} {interd_time:>10.1f} {interd_time/60:>12.2f} {interd_time/pipeline_total*100:>11.1f}%")
    print(f"  {'-'*20} {'-'*10} {'-'*12} {'-'*12}")
    print(f"  {'Total Pipeline':<20} {pipeline_total:>10.1f} {pipeline_total/60:>12.2f} {'100.0%':>12}")
    print('='*70)


if __name__ == "__main__":
    args = arguments.parse_args()
    para(args)
    setup_seed(args.seed)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 加载数据（支持KuaiRand数据集）
    print(f"\nLoading dataset: {args.dataset}")
    if args.dataset == 'KuaiRand':
        data_dict = load_kuairand.load_kuairand(device=device, verbose=True, random_seed=args.seed)
        train = data_dict['train_biased']
        biased_val = data_dict['val_biased']
        unif_train = data_dict['train_unbiased']
        unif_val = data_dict['val_unbiased']
        test = data_dict['test_unbiased']
        biased_test = data_dict['test_biased']
    else:
        train, biased_val, unif_train, unif_val, test, biased_test = \
            load_dataset_custom.load_dataset(data_name=args.dataset, seed=args.seed, device=device, unif_ratio=0.5)
    
    print(f"  Biased Train: {train._nnz()} samples")
    print(f"  Biased Val:   {biased_val._nnz()} samples")
    print(f"  Unbiased Train: {unif_train._nnz()} samples")
    print(f"  Unbiased Val:   {unif_val._nnz()} samples")
    print(f"  Biased Test:  {biased_test._nnz()} samples")
    print(f"  Unbiased Test: {test._nnz()} samples")
    
    # 训练和评估
    train_and_eval(
        train, biased_val, unif_train, unif_val,
        test, biased_test, args, device=device
    )
