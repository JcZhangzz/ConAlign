# coding: utf-8
import itertools
import math
import os
import sys
import random
import time
import numpy as np
import torch
import torch.nn as nn


_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(_THIS_DIR))
sys.path.insert(0, _PROJECT_ROOT)
sys.path.insert(0, _THIS_DIR)

import utils.load_dataset_new as load_dataset_custom
import utils.data_loader
import utils.metrics
from utils.early_stop import EarlyStopping, Stop_args

import arguments
from model import InvPrefImplicit_changed, KD_Debias_student


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
            'batch_size_KD': 1024,
            'T_epochs': 300,
            'S_epochs': 500,
            'cluster_interval': 5,
            'evaluate_interval': 1,
            'patience': 80,
        }
        args.model_args = {
            'env_num': 2,
            'factor_num': 40,
            'reg_only_embed': True,
            'reg_env_embed': False,
        }
        args.loss_args = {
            'lr': 0.003,
            'lr_KD': 0.005,
            'gama': 0.17,
            'invariant_coe': 1,
            'env_aware_coe': 9.988658447411407,
            'variant_coe': 5,
            'env_coe': 5,
            'L2_coe': 10,
            'L1_coe': 0.05,
            'student_coe': 1,
            'stu_L2_coe': 0.1,
            'stu_L1_coe': 0.01,
            'alpha': 1.9053711444718746,
            'use_class_re_weight': True,
            'use_recommend_re_weight': False,
        }
    elif args.dataset == 'coat':
        args.training_args = {
            'batch_size': 128,
            'batch_size_KD': 128,
            'T_epochs': 500,
            'S_epochs': 500,
            'cluster_interval': 5,
            'evaluate_interval': 1,
            'patience': 60,
        }
        args.model_args = {
            'env_num': 2,
            'factor_num': 40,
            'reg_only_embed': True,
            'reg_env_embed': False,
        }
        args.loss_args = {
            'lr': 0.0001,
            'lr_KD': 0.0001,
            'gama': 0.1,
            'invariant_coe': 2,
            'env_aware_coe': 5,
            'variant_coe': 5,
            'env_coe': 1,
            'L2_coe': 0.01,
            'L1_coe': 0.01,
            'student_coe': 5,
            'stu_L2_coe': 0.01,
            'stu_L1_coe': 0.01,
            'alpha': 2,
            'use_class_re_weight': True,
            'use_recommend_re_weight': False,
        }
    elif 'KuaiRand' in args.dataset:
        args.training_args = {
            'batch_size': 1024,
            'batch_size_KD': 8192,
            'T_epochs': 600,
            'S_epochs': 600,
            'cluster_interval': 5,
            'evaluate_interval': 1,
            'patience': 80,
        }
        args.model_args = {
            'env_num': 2,
            'factor_num': 40,
            'reg_only_embed': True,
            'reg_env_embed': False,
        }
        args.loss_args = {
            'lr': 0.001,
            'lr_KD': 0.001,
            'gama': 0.3,
            'invariant_coe': 1,
            'env_aware_coe': 10,
            'variant_coe': 5,
            'env_coe': 1,
            'L2_coe': 0.001,
            'L1_coe': 0.001,
            'student_coe': 1,
            'stu_L2_coe': 0.001,
            'stu_L1_coe': 0.0001,
            'alpha': 1,
            'use_class_re_weight': True,
            'use_recommend_re_weight': False,
        }


def mini_batch(batch_size: int, *tensors):
    if len(tensors) == 1:
        tensor = tensors[0]
        for i in range(0, len(tensor), batch_size):
            yield tensor[i:i + batch_size]
    else:
        for i in range(0, len(tensors[0]), batch_size):
            yield tuple(x[i:i + batch_size] for x in tensors)


def _init_eps(envs_num):
    base_eps = 1e-10
    eps_list = [base_eps * (1e-1 ** idx) for idx in range(envs_num)]
    temp = torch.Tensor(eps_list)
    eps_random_tensor = torch.Tensor(list(itertools.permutations(temp)))
    return eps_random_tensor


def train_and_eval(bias_train, bias_val, unif_val, test_data, biased_test_data, m, n, args, device='cpu'):
    users_train = bias_train._indices()[0]
    items_train = bias_train._indices()[1]
    scores_train = bias_train._values().float()

    N = users_train.shape[0]
    envs_num = args.model_args['env_num']

    envs = torch.LongTensor(np.random.randint(0, envs_num, N)).to(device)

    const_env_tensor_list = []
    for env in range(envs_num):
        envs_tensor = torch.LongTensor(np.full(N, env, dtype=int)).to(device)
        const_env_tensor_list.append(envs_tensor)

    eps_random_tensor = _init_eps(envs_num).to(device)

    sample_weights = torch.zeros(N).to(device)
    class_weights = torch.zeros(envs_num).to(device)

    val_loader = utils.data_loader.DataLoader(
        utils.data_loader.Interactions(unif_val),
        batch_size=args.training_args['batch_size']
    )
    biased_val_loader = utils.data_loader.DataLoader(
        utils.data_loader.Interactions(bias_val),
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

    T_model = InvPrefImplicit_changed(
        user_num=m,
        item_num=n,
        env_num=args.model_args['env_num'],
        factor_num=args.model_args['factor_num'],
        reg_only_embed=args.model_args['reg_only_embed'],
        reg_env_embed=args.model_args['reg_env_embed'],
    ).to(device)

    S_model = KD_Debias_student(
        user_num=m,
        item_num=n,
        factor_num=args.model_args['factor_num'],
    ).to(device)

    T_optimizer = torch.optim.Adam(T_model.parameters(), lr=args.loss_args['lr'])
    S_optimizer = torch.optim.Adam(S_model.parameters(), lr=args.loss_args['lr_KD'])

    recommend_loss_type = nn.BCELoss
    cluster_distance_func = nn.BCELoss(reduction='none')
    env_loss_type = nn.NLLLoss

    batch_num = math.ceil(N / args.training_args['batch_size'])

    if args.loss_args['alpha'] is None:
        alpha = 0.
        update_alpha = True
    else:
        alpha = args.loss_args['alpha']
        update_alpha = False

    use_class_re_weight = args.loss_args['use_class_re_weight']
    use_recommend_re_weight = args.loss_args['use_recommend_re_weight']

    def stat_envs():
        nonlocal sample_weights, class_weights
        result = {}
        class_rate_np = np.zeros(envs_num)
        for env in range(envs_num):
            cnt = int(torch.sum(envs == env))
            result[env] = cnt
            class_rate_np[env] = min(cnt + 1, scores_train.shape[0] - 1)
        class_rate_np = class_rate_np / scores_train.shape[0]
        class_weights = torch.Tensor(class_rate_np).to(device)
        sample_weights = class_weights[envs]
        return result

    def cluster():
        nonlocal envs
        T_model.eval()
        new_env_tensors_list = []
        for batch_u, batch_i, batch_s in mini_batch(
                args.training_args['batch_size'], users_train, items_train, scores_train):
            distances_list = []
            for env_idx in range(envs_num):
                envs_tensor = const_env_tensor_list[env_idx][0:batch_u.shape[0]]
                cluster_pred = T_model.cluster_predict(batch_u, batch_i, envs_tensor)
                distances = cluster_distance_func(cluster_pred, batch_s).reshape(-1, 1)
                distances_list.append(distances)

            each_envs_distances = torch.cat(distances_list, dim=1)
            sort_random_index = np.random.randint(0, eps_random_tensor.shape[0], each_envs_distances.shape[0])
            random_eps = eps_random_tensor[sort_random_index]
            each_envs_distances = each_envs_distances + random_eps
            new_envs = torch.argmin(each_envs_distances, dim=1)
            new_env_tensors_list.append(new_envs)

        all_new_envs = torch.cat(new_env_tensors_list, dim=0)
        diff_num = int(torch.sum((envs - all_new_envs) != 0))
        envs = all_new_envs
        T_model.train()
        return diff_num

    def run_eval(loader, model_fn, name=""):
        preds, trues, users_list, items_list = [], [], [], []
        ut_dict, pt_dict = {}, {}
        with torch.no_grad():
            for u, i, r in loader:
                u, i, r = u.to(device), i.to(device), r.to(device).float()
                scores = model_fn(u, i)
                preds.append(scores)
                trues.append(r)
                users_list.append(u)
                items_list.append(i)
                for idx in range(len(u)):
                    uid = u[idx].item()
                    if uid not in ut_dict:
                        ut_dict[uid] = []
                        pt_dict[uid] = []
                    ut_dict[uid].append(r[idx].item())
                    pt_dict[uid].append(scores[idx].item())

        preds = torch.cat(preds)
        trues = torch.cat(trues)
        users_t = torch.cat(users_list)
        items_t = torch.cat(items_list)

        metric_list = ['BCE', 'AUC', 'UAUC', 'UNDCG', 'Recall_Precision_NDCG@']

        res = utils.metrics.evaluate(
            preds, trues, metric_list,
            users=users_t, items=items_t,
            UAUC=(ut_dict, pt_dict),
            UNDCG=(ut_dict, pt_dict)
        )
        
        if name:
            print(f"\n{'='*60}")
            print(f"[{name} Test Results]")
            print(f"{'='*60}")
            for k, v in res.items():
                print(f"  {k}: {v:.4f}")
        
        return res

    print('*' * 80)
    print('Phase 1: Train Teacher Model (InvPrefImplicit_changed)')
    print('*' * 80)

    epoch_cnt = 0
    stopping_args = Stop_args(patience=args.training_args['patience'],
                              max_epochs=args.training_args['T_epochs'])
    early_stopping = EarlyStopping(T_model, **stopping_args)

    stat_envs()

    teacher_train_start = time.time()
    teacher_epoch_times = []

    while epoch_cnt < args.training_args['T_epochs']:
        teacher_epoch_start = time.time()
        T_model.train()
        loss_sum = 0.0
        n_batches = 0
        for batch_index, (batch_u, batch_i, batch_s, batch_envs, batch_sw) in enumerate(
                mini_batch(args.training_args['batch_size'],
                           users_train, items_train, scores_train, envs, sample_weights)):

            if update_alpha:
                p = float(batch_index + (epoch_cnt + 1) * batch_num) / float(
                    (epoch_cnt + 1) * batch_num)
                alpha = 2. / (1. + np.exp(-10. * p)) - 1.

            invariant_score, variant_score, env_aware_score, env_outputs = T_model(
                batch_u, batch_i, batch_envs, alpha)

            if use_class_re_weight:
                env_loss = env_loss_type(reduction='none')
            else:
                env_loss = env_loss_type()

            if use_recommend_re_weight:
                rec_loss = recommend_loss_type(reduction='none')
            else:
                rec_loss = recommend_loss_type()

            invariant_loss = rec_loss(invariant_score, batch_s)
            variant_loss = rec_loss(variant_score, batch_s)
            env_aware_loss = rec_loss(env_aware_score, batch_s)
            envs_loss = env_loss(env_outputs, batch_envs)

            if use_class_re_weight:
                envs_loss = torch.mean(envs_loss * batch_sw)
            if use_recommend_re_weight:
                invariant_loss = torch.mean(invariant_loss * batch_sw)
                variant_loss = torch.mean(variant_loss * batch_sw)
                env_aware_loss = torch.mean(env_aware_loss * batch_sw)

            L2_reg = T_model.get_L2_reg(batch_u, batch_i, batch_envs)
            L1_reg = T_model.get_L1_reg(batch_u, batch_i, batch_envs)

            loss = (invariant_loss * args.loss_args['invariant_coe']
                    + env_aware_loss * args.loss_args['env_aware_coe']
                    + variant_loss * args.loss_args['variant_coe']
                    + envs_loss * args.loss_args['env_coe']
                    + L2_reg * args.loss_args['L2_coe']
                    + L1_reg * args.loss_args['L1_coe'])

            T_optimizer.zero_grad()
            loss.backward()
            T_optimizer.step()

            loss_sum += loss.item()
            n_batches += 1

        epoch_cnt += 1
        teacher_epoch_times.append(time.time() - teacher_epoch_start)

        if epoch_cnt % args.training_args['cluster_interval'] == 0:
            diff_num = cluster()
            env_stat = stat_envs()
            print(f'  [Cluster] epoch {epoch_cnt} diff={diff_num} env_cnt={env_stat}')

        if epoch_cnt % args.training_args['evaluate_interval'] == 0:
            T_model.eval()
            val_res = run_eval(val_loader, T_model.predict)
            T_model.train()

            print(f'[Teacher] Epoch {epoch_cnt:3d} | Loss: {loss_sum/n_batches:.4f} | '
                  f'Val AUC: {val_res.get("AUC", 0):.4f} UAUC: {val_res.get("UAUC", 0):.4f}')

            if early_stopping.check([val_res['AUC']], epoch_cnt):
                print(f'Early stopping at epoch {epoch_cnt}')
                break

    teacher_train_total = time.time() - teacher_train_start
    teacher_avg_epoch = sum(teacher_epoch_times) / max(len(teacher_epoch_times), 1)

    T_model.load_state_dict(early_stopping.best_state)
    print(f'\n[Teacher] Best epoch: {early_stopping.best_epoch} | Best AUC: {early_stopping.remembered_vals[0]:.4f}')
    print(f'[Teacher Timing] Total: {teacher_train_total:.1f}s ({teacher_train_total/60:.2f}min) | Epochs: {len(teacher_epoch_times)} | Avg/epoch: {teacher_avg_epoch:.2f}s')

    T_model.eval()
    run_eval(test_loader, T_model.predict, "Teacher Test (Unbiased)")
    run_eval(biased_test_loader, T_model.predict, "Teacher Test (Biased)")
    print('*' * 80)

    print('*' * 80)
    print('Phase 2: Train Student Model (KD_Debias_student)')
    print('*' * 80)

    for param in T_model.parameters():
        param.requires_grad = False
    T_model.eval()

    epoch_cnt = 0
    stopping_args_s = Stop_args(patience=args.training_args['patience'],
                                max_epochs=args.training_args['S_epochs'])
    early_stopping_s = EarlyStopping(S_model, **stopping_args_s)

    gama_base = args.loss_args['gama']

    student_train_start = time.time()
    student_epoch_times = []

    while epoch_cnt < args.training_args['S_epochs']:
        student_epoch_start = time.time()
        S_model.train()
        loss_sum = 0.0
        n_batches = 0

        gama = gama_base

        for batch_u, batch_i, batch_s, batch_envs, batch_sw in mini_batch(
                args.training_args['batch_size_KD'],
                users_train, items_train, scores_train, envs, sample_weights):

            with torch.no_grad():
                invariant_score, variant_score, env_aware_score, _ = T_model(
                    batch_u, batch_i, batch_envs, alpha)

            stu_score = S_model(batch_u, batch_i)

            dis = torch.abs(env_aware_score - invariant_score)
            W_inv = torch.pow(1 - dis, gama)
            W_env = 1 - W_inv

            y_star = W_inv * invariant_score + W_env * env_aware_score

            student_loss = nn.MSELoss(reduction='mean')(stu_score, y_star)
            L2_reg = S_model.get_L2_reg(batch_u, batch_i)
            L1_reg = S_model.get_L1_reg(batch_u, batch_i)

            loss = (student_loss * args.loss_args['student_coe']
                    + L2_reg * args.loss_args['stu_L2_coe']
                    + L1_reg * args.loss_args['stu_L1_coe'])

            S_optimizer.zero_grad()
            loss.backward()
            S_optimizer.step()

            loss_sum += loss.item()
            n_batches += 1

        epoch_cnt += 1
        student_epoch_times.append(time.time() - student_epoch_start)

        S_model.eval()
        val_res = run_eval(val_loader, S_model.predict)
        S_model.train()

        avg_loss = loss_sum / n_batches
        print(f'[Student] Epoch {epoch_cnt:3d} | Loss: {avg_loss:.4f} | '
              f'Val AUC: {val_res.get("AUC", 0):.4f} UAUC: {val_res.get("UAUC", 0):.4f}')

        if early_stopping_s.check([val_res['AUC']], epoch_cnt):
            print(f'Early stopping at epoch {epoch_cnt}')
            break

    student_train_total = time.time() - student_train_start
    student_avg_epoch = sum(student_epoch_times) / max(len(student_epoch_times), 1)

    S_model.load_state_dict(early_stopping_s.best_state)
    print(f'\n[Student] Best epoch: {early_stopping_s.best_epoch} | Best AUC: {early_stopping_s.remembered_vals[0]:.4f}')
    print(f'[Student Timing] Total: {student_train_total:.1f}s ({student_train_total/60:.2f}min) | Epochs: {len(student_epoch_times)} | Avg/epoch: {student_avg_epoch:.2f}s')

    S_model.eval()
    s_test_res = run_eval(test_loader, S_model.predict, "Student Test (Unbiased)")
    s_biased_res = run_eval(biased_test_loader, S_model.predict, "Student Test (Biased)")

    pipeline_total = teacher_train_total + student_train_total
    print(f"\n{'='*70}")
    print(f"[Timing Summary] KD-Debias Training")
    print(f"{'='*70}")
    print(f"  {'Stage':<20} {'Total(s)':>10} {'Total(min)':>12} {'Epochs':>8} {'Avg/epoch(s)':>14}")
    print(f"  {'-'*20} {'-'*10} {'-'*12} {'-'*8} {'-'*14}")
    print(f"  {'Teacher (Phase 1)':<20} {teacher_train_total:>10.1f} {teacher_train_total/60:>12.2f} {len(teacher_epoch_times):>8d} {teacher_avg_epoch:>14.2f}")
    print(f"  {'Student (Phase 2)':<20} {student_train_total:>10.1f} {student_train_total/60:>12.2f} {len(student_epoch_times):>8d} {student_avg_epoch:>14.2f}")
    print(f"  {'Pipeline Total':<20} {pipeline_total:>10.1f} {pipeline_total/60:>12.2f}")
    print(f"{'='*70}")

    return s_test_res


if __name__ == '__main__':
    args = arguments.parse_args()
    para(args)
    setup_seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    print(f'Dataset: {args.dataset}')

    print(f"\nLoading dataset: {args.dataset}")
    if args.dataset == 'KuaiRand':
        import utils.load_kuairand as load_kuairand
        data_dict = load_kuairand.load_kuairand(device=device, verbose=True, random_seed=args.seed)
        bias_train = data_dict['train_biased']
        bias_val = data_dict['val_biased']
        unif_train = data_dict['train_unbiased']
        unif_val = data_dict['val_unbiased']
        test_unbiased = data_dict['test_unbiased']
        test_biased = data_dict['test_biased']
    else:
        bias_train, bias_val, unif_train, unif_val, test_unbiased, test_biased = \
            load_dataset_custom.load_dataset(
                data_name=args.dataset,
                unif_ratio=0.5,
                seed=args.seed,
                device=device
            )

    m, n = bias_train.shape

    print(f'  Biased Train:   {bias_train._nnz()} samples')
    print(f'  Biased Val:     {bias_val._nnz()} samples')
    print(f'  Unbiased Train: {unif_train._nnz()} samples')
    print(f'  Unbiased Val:   {unif_val._nnz()} samples')
    print(f'  Unbiased Test:  {test_unbiased._nnz()} samples')
    print(f'  Biased Test:    {test_biased._nnz()} samples')
    print(f'  Users: {m}, Items: {n}')

    train_and_eval(
        bias_train=bias_train,
        bias_val=bias_val,
        unif_val=unif_val,
        test_data=test_unbiased,
        biased_test_data=test_biased,
        m=m,
        n=n,
        args=args,
        device=device
    )
