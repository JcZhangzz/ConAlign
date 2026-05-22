# coding: utf-8

import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from model import MetaModule, MetaEmbed, to_var
import utils.load_dataset_new as load_dataset_custom
import utils.load_kuairand as load_kuairand
import utils.data_loader
import utils.metrics
from utils.early_stop import EarlyStopping, Stop_args
import arguments


def setup_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)



class DualTowerMF(nn.Module):
    
    def __init__(self, n_user, n_item, dim=32, dropout=0.0, alignment_weight=0.1):
        super().__init__()
        
        self.dim = dim
        self.alignment_weight = alignment_weight
        
        self.user_latent_biased = nn.Embedding(n_user, dim)
        self.item_latent_biased = nn.Embedding(n_item, dim)
        self.user_bias_biased = nn.Embedding(n_user, 1)
        self.item_bias_biased = nn.Embedding(n_item, 1)

        nn.init.kaiming_normal_(self.user_latent_biased.weight, mode='fan_out', a=0)
        nn.init.kaiming_normal_(self.item_latent_biased.weight, mode='fan_out', a=0)
        nn.init.kaiming_normal_(self.user_bias_biased.weight, mode='fan_out', a=0)
        nn.init.kaiming_normal_(self.item_bias_biased.weight, mode='fan_out', a=0)
        
        self.user_latent_unbiased = nn.Embedding(n_user, dim)
        self.item_latent_unbiased = nn.Embedding(n_item, dim)
        self.user_bias_unbiased = nn.Embedding(n_user, 1)
        self.item_bias_unbiased = nn.Embedding(n_item, 1)
        
        nn.init.kaiming_normal_(self.user_latent_unbiased.weight, mode='fan_out', a=0)
        nn.init.kaiming_normal_(self.item_latent_unbiased.weight, mode='fan_out', a=0)
        nn.init.kaiming_normal_(self.user_bias_unbiased.weight, mode='fan_out', a=0)
        nn.init.kaiming_normal_(self.item_bias_unbiased.weight, mode='fan_out', a=0)
        
  
        
        self.dropout = nn.Dropout(p=dropout)
    
    def forward(self, users, items):

        u_latent_biased = self.dropout(self.user_latent_biased(users))
        i_latent_biased = self.dropout(self.item_latent_biased(items))
        u_bias_biased = self.user_bias_biased(users)
        i_bias_biased = self.item_bias_biased(items)
        

        biased_logits = torch.sum(u_latent_biased * i_latent_biased, dim=1) + u_bias_biased.squeeze() + i_bias_biased.squeeze()
        biased_probs = torch.sigmoid(biased_logits)
        
        u_latent_unbiased = self.dropout(self.user_latent_unbiased(users))
        i_latent_unbiased = self.dropout(self.item_latent_unbiased(items))
        u_bias_unbiased = self.user_bias_unbiased(users)
        i_bias_unbiased = self.item_bias_unbiased(items)
        
        unbiased_logits = torch.sum(u_latent_unbiased * i_latent_unbiased, dim=1) + u_bias_unbiased.squeeze() + i_bias_unbiased.squeeze()
        unbiased_probs = torch.sigmoid(unbiased_logits)
        
        user_latent_alignment = F.l1_loss(u_latent_unbiased, u_latent_biased.detach())
        item_latent_alignment = F.l1_loss(i_latent_unbiased, i_latent_biased.detach())
        user_bias_alignment = F.l1_loss(u_bias_unbiased, u_bias_biased.detach())
        item_bias_alignment = F.l1_loss(i_bias_unbiased, i_bias_biased.detach())
        alignment_loss = (user_latent_alignment + item_latent_alignment + 
                         user_bias_alignment + item_bias_alignment) / 2
        
        return biased_probs, unbiased_probs, alignment_loss
    
    def l2_norm(self, users, items):
        users_unique = torch.unique(users)
        items_unique = torch.unique(items)
        
        l2_loss = (
            torch.sum(self.user_latent_biased(users_unique) ** 2) / len(users_unique) +
            torch.sum(self.item_latent_biased(items_unique) ** 2) / len(items_unique)
        ) / 2
        return l2_loss



def para(args):
    if args.dataset == 'yahooR3':
        args.training_args = {
            'batch_size': 1024,  
            'pretrain_epochs': 500,  
            'finetune_epochs': 500,  
            'patience': 60,
            'block_batch': [6000, 500]
        }
        args.model_args = {
            'dim': 10,
            'learning_rate_1': 0.01,
            'learning_rate_2': 0.01,
            'weight_decay_1': 0.001,
            'weight_decay_2': 0.001,
            'alignment_weight': 1100,
            'adaptive_alignment': True  
        }
    elif args.dataset == 'coat':
        args.training_args = {
            'batch_size': 128,  
            'pretrain_epochs': 500, 
            'finetune_epochs': 500,  
            'patience': 60,
            'block_batch': [64, 64]
        }
        args.model_args = {
            'dim': 10,
            'learning_rate_1': 0.01,
            'learning_rate_2': 0.001,
            'weight_decay_1': 0.01,
            'weight_decay_2': 0.01,
            'alignment_weight': 10,
            'adaptive_alignment': True  
        }
    elif 'KuaiRand' in args.dataset:
        args.training_args = {
            'batch_size': 1024,  
            'pretrain_epochs': 500,  
            'finetune_epochs': 500,  
            'patience': 60,
            'block_batch': [500, 10000]
        }
        args.model_args = {
            'dim': 10,
            'learning_rate_1': 0.01,
            'learning_rate_2': 0.01,
            'weight_decay_1': 0.0001,
            'weight_decay_2': 0.0001,
            'alignment_weight': 42725,
            'adaptive_alignment': True  
        }


def train_and_eval(train_data, biased_val_data, unif_train_data, unif_val_data, 
                   test_data, biased_test_data, args, device='cuda'):
    
    users_unif = unif_train_data._indices()[0]
    items_unif = unif_train_data._indices()[1]
    y_unif = unif_train_data._values().float()
    

    train_loader = utils.data_loader.Block(
        train_data,
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
    model = DualTowerMF(
        n_user, n_item,
        dim=args.model_args['dim'],
        alignment_weight=args.model_args.get('alignment_weight', 0.1)
    ).to(device)
    

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.model_args['learning_rate_1'],
        weight_decay=0  #
    )
    

    criterion = nn.BCELoss(reduction='none')
    

    import os
    script_dir = os.path.dirname(os.path.abspath(__file__))
    save_path = os.path.join(script_dir, f"best_model_{args.dataset}_dualtower.pth")
    is_test_only = getattr(args, 'test_only', False)
    
    if not is_test_only:
        print("=" * 60)
        print("Two-Stage Training Strategy")
        print(f"Model: DualTowerMF (Simple MF + Alignment Loss)")
        print(f"Dataset: {args.dataset}")
        print(f"Users: {n_user}, Items: {n_item}")
        print(f"Stage 1 (Pretrain Biased Tower): {args.training_args['pretrain_epochs']} epochs")
        print(f"Stage 2 (Finetune Unbiased Tower): {args.training_args['finetune_epochs']} epochs")
        print("=" * 60)
        
        print("\n" + "=" * 60)
        print("STAGE 1: Pretraining Biased Tower (Only Biased Data)")
        print("=" * 60)
        stage1_start = time.time()
        stage1_epoch_times = []

        stopping_args_stage1 = Stop_args(
            patience=args.training_args['patience'],
            max_epochs=args.training_args['pretrain_epochs']
        )
        early_stopping_stage1 = EarlyStopping(model, **stopping_args_stage1)
        
        for epoch in range(args.training_args['pretrain_epochs']):
            epoch_start = time.time()
            model.train()
            train_preds = []
            train_trues = []
            
            total_l2_loss = 0.0
            n_batches = 0
            for u_batch_idx, u_batch in enumerate(train_loader.User_loader):
                for i_batch_idx, i_batch in enumerate(train_loader.Item_loader):
                    users_train, items_train, y_train = train_loader.get_batch(u_batch, i_batch, device=device)
                    y_train = y_train.float()
                    
                    optimizer.zero_grad()
                    
                    biased_probs, _, _ = model(users_train, items_train)
                    
                    loss_biased = criterion(biased_probs, y_train).sum()
                    
                    users_unique = torch.unique(users_train)
                    items_unique = torch.unique(items_train)
                    l2_loss_biased = (
                        torch.sum(model.user_latent_biased(users_unique) ** 2) +
                        torch.sum(model.item_latent_biased(items_unique) ** 2)
                    ) / 2
                    
                    total_loss = loss_biased + args.model_args['weight_decay_1'] * l2_loss_biased
                    
                    total_loss.backward()
                    optimizer.step()
                    
                    total_l2_loss += l2_loss_biased.item()
                    n_batches += 1
                    
                    train_preds.append(biased_probs.detach())
                    train_trues.append(y_train.detach())
            
            train_preds = torch.cat(train_preds)
            train_trues = torch.cat(train_trues)
            train_res = utils.metrics.evaluate(train_preds, train_trues, ['BCE', 'AUC'])
            
            model.eval()
            with torch.no_grad():
                val_preds = []
                val_trues = []
                for u, i, r in biased_val_loader:  
                    biased_pred, _, _ = model(u.to(device), i.to(device))
                    val_preds.append(biased_pred)
                    val_trues.append(r.to(device).float())
                
                val_preds = torch.cat(val_preds)
                val_trues = torch.cat(val_trues)
                val_res = utils.metrics.evaluate(val_preds, val_trues, ['BCE', 'AUC'])
            
            avg_l2_loss = total_l2_loss / max(n_batches, 1)
            reg_term = args.model_args['weight_decay_1'] * avg_l2_loss
            
            with torch.no_grad():
                user_emb_norm = torch.norm(model.user_latent_biased.weight).item()
                item_emb_norm = torch.norm(model.item_latent_biased.weight).item()
                avg_user_norm = user_emb_norm / model.user_latent_biased.weight.shape[0]
                avg_item_norm = item_emb_norm / model.item_latent_biased.weight.shape[0]
            
            epoch_time = time.time() - epoch_start
            stage1_epoch_times.append(epoch_time)

            print(f"[Stage 1] Epoch {epoch:3d} | Train BCE: {train_res['BCE']:.4f} AUC: {train_res['AUC']:.4f} | "
                  f"Val BCE: {val_res['BCE']:.4f} AUC: {val_res['AUC']:.4f}")
            
            if early_stopping_stage1.check([val_res['AUC']], epoch):
                print(f"Early stopping at epoch {epoch}")
                break
        
        model.load_state_dict(early_stopping_stage1.best_state)
        stage1_total = time.time() - stage1_start
        stage1_avg_epoch = sum(stage1_epoch_times) / max(len(stage1_epoch_times), 1)
        print(f"\n[Stage 1] Best model loaded (Val AUC: {early_stopping_stage1.remembered_vals[0]:.4f})")
        
        # Transfer: copy the trained biased-tower embeddings to the unbiased tower
        print("\n" + "=" * 60)
        print("TRANSFERRING: Copying Biased Tower Embeddings to Unbiased Tower")
        print("=" * 60)
        
        with torch.no_grad():
            model.user_latent_unbiased.weight.copy_(model.user_latent_biased.weight)
            model.item_latent_unbiased.weight.copy_(model.item_latent_biased.weight)
            model.user_bias_unbiased.weight.copy_(model.user_bias_biased.weight)
            model.item_bias_unbiased.weight.copy_(model.item_bias_biased.weight)
        
        print("✓ User latent embeddings transferred")
        print("✓ Item latent embeddings transferred")
        print("✓ User bias transferred")
        print("✓ Item bias transferred")
        
        # Stage 2: freeze the biased tower and train only the unbiased tower
        print("\n" + "=" * 60)
        print("STAGE 2: Finetuning Unbiased Tower (Only Unbiased Data)")
        print("=" * 60)
        
        for param in model.user_latent_biased.parameters():
            param.requires_grad = False
        for param in model.item_latent_biased.parameters():
            param.requires_grad = False
        for param in model.user_bias_biased.parameters():
            param.requires_grad = False
        for param in model.item_bias_biased.parameters():
            param.requires_grad = False
        
        print("✓ Biased tower parameters frozen (requires_grad=False)")
        
        optimizer = torch.optim.SGD(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=args.model_args['learning_rate_2'],
            weight_decay=0  
        )
        
        stopping_args_stage2 = Stop_args(
            patience=args.training_args['patience'],
            max_epochs=args.training_args['finetune_epochs']
        )
        early_stopping_stage2 = EarlyStopping(model, **stopping_args_stage2)
        
        is_kuairand = ('KuaiRand' in args.dataset)
        current_alignment_weight = args.model_args['alignment_weight']

        
        stage2_start = time.time()
        stage2_epoch_times = []

        for epoch in range(args.training_args['finetune_epochs']):
            epoch_start = time.time()
            model.train()
            train_preds = []
            train_trues = []
            total_alignment_loss = 0.0
            n_unif_batches = 0
            n_alignment_used = 0  
            
            # Merge biased and unbiased data and create source indicators
            total_l2_loss = 0.0
            
            users_bias = train_data._indices()[0]
            items_bias = train_data._indices()[1]
            y_bias = train_data._values().float()
            
            is_unbiased_bias = torch.zeros(len(users_bias), dtype=torch.bool)
            is_unbiased_unif = torch.ones(len(users_unif), dtype=torch.bool)
            
            users_all = torch.cat([users_bias, users_unif])
            items_all = torch.cat([items_bias, items_unif])
            y_all = torch.cat([y_bias, y_unif])
            is_unbiased_all = torch.cat([is_unbiased_bias, is_unbiased_unif])
            
            shuffle_indices = torch.randperm(len(users_all))
            users_shuffled = users_all[shuffle_indices]
            items_shuffled = items_all[shuffle_indices]
            y_shuffled = y_all[shuffle_indices]
            is_unbiased_shuffled = is_unbiased_all[shuffle_indices]
            
            batch_size = args.training_args['batch_size']  #
            n_batches_total = (len(users_all) + batch_size - 1) // batch_size
            
            for i in range(n_batches_total):
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, len(users_all))
                
                batch_users = users_shuffled[start_idx:end_idx].to(device)
                batch_items = items_shuffled[start_idx:end_idx].to(device)
                batch_y = y_shuffled[start_idx:end_idx].to(device)
                batch_is_unbiased = is_unbiased_shuffled[start_idx:end_idx].to(device)
                
                optimizer.zero_grad()
                
                biased_probs, unbiased_probs, alignment_loss = model(batch_users, batch_items)
                
                unbiased_mask = batch_is_unbiased
                n_unbiased = unbiased_mask.sum().item()
                
                total_loss = torch.tensor(0.0, device=device)
                
                use_alignment = True
                if args.model_args.get('adaptive_alignment', False):
                    with torch.no_grad():
                        loss_unbiased_all = criterion(unbiased_probs, batch_y).mean()
                        loss_biased_all = criterion(biased_probs, batch_y).mean()
                    # Use alignment only when the unbiased tower performs worse
                    use_alignment = (loss_unbiased_all.item() > loss_biased_all.item())
                
                # Only the prediction loss of unbiased data is added to the final loss
                if n_unbiased > 0:
                    unbiased_pred = unbiased_probs[unbiased_mask]
                    unbiased_true = batch_y[unbiased_mask]
                    loss_unbiased = criterion(unbiased_pred, unbiased_true).sum()
                    total_loss += loss_unbiased
                    
                    train_preds.append(unbiased_pred.detach())
                    train_trues.append(unbiased_true.detach())
                
                aw_for_this_batch = current_alignment_weight if is_kuairand else model.alignment_weight
                if use_alignment:
                    total_loss += aw_for_this_batch * alignment_loss
                    total_alignment_loss += alignment_loss.item()
                    n_alignment_used += 1
                
                users_unique = torch.unique(batch_users)
                items_unique = torch.unique(batch_items)
                l2_loss = (
                    torch.sum(model.user_latent_unbiased(users_unique) ** 2) +
                    torch.sum(model.item_latent_unbiased(items_unique) ** 2)
                ) / 2
                
                total_loss += args.model_args['weight_decay_2'] * l2_loss
                
                total_loss.backward()
                optimizer.step()
                
                total_l2_loss += l2_loss.item()
                n_unif_batches += 1
            
            if len(train_preds) > 0:
                train_preds = torch.cat(train_preds)
                train_trues = torch.cat(train_trues)
                train_res = utils.metrics.evaluate(train_preds, train_trues, ['BCE', 'AUC'])
            else:
                train_res = {'BCE': 0.0, 'AUC': 0.5}
            
            model.eval()
            with torch.no_grad():
                val_preds = []
                val_trues = []
                for u, i, r in unbiased_val_loader:
                    _, unbiased_pred, _ = model(u.to(device), i.to(device))
                    val_preds.append(unbiased_pred)
                    val_trues.append(r.to(device).float())
                
                val_preds = torch.cat(val_preds)
                val_trues = torch.cat(val_trues)
                val_res = utils.metrics.evaluate(val_preds, val_trues, ['BCE', 'AUC'])
            
            avg_l2_loss = total_l2_loss / max(n_unif_batches, 1)
            avg_alignment = total_alignment_loss / max(n_alignment_used, 1) if n_alignment_used > 0 else 0.0
            alignment_ratio = n_alignment_used / max(n_unif_batches, 1)
            reg_term = args.model_args['weight_decay_2'] * avg_l2_loss
            
            with torch.no_grad():
                user_emb_norm = torch.norm(model.user_latent_unbiased.weight).item()
                item_emb_norm = torch.norm(model.item_latent_unbiased.weight).item()
                avg_user_norm = user_emb_norm / model.user_latent_unbiased.weight.shape[0]
                avg_item_norm = item_emb_norm / model.item_latent_unbiased.weight.shape[0]
            
            if is_kuairand:
                current_alignment_weight += 0
            
            epoch_time = time.time() - epoch_start
            stage2_epoch_times.append(epoch_time)

            print(f"[Stage 2] Epoch {epoch:3d} | Train BCE: {train_res['BCE']:.4f} AUC: {train_res['AUC']:.4f} | "
                  f"Val BCE: {val_res['BCE']:.4f} AUC: {val_res['AUC']:.4f}")
            
            if early_stopping_stage2.check([val_res['AUC']], epoch):
                print(f"Early stopping at epoch {epoch}")
                break
        
        model.load_state_dict(early_stopping_stage2.best_state)
        stage2_total = time.time() - stage2_start
        stage2_avg_epoch = sum(stage2_epoch_times) / max(len(stage2_epoch_times), 1)
        print(f"\n[Stage 2] Best model loaded (Val AUC: {early_stopping_stage2.remembered_vals[0]:.4f})")

        torch.save(model.state_dict(), save_path)
        print(f"\nFinal model saved to {save_path}")
    
    else:
        model.load_state_dict(torch.load(save_path, map_location=device))
        print(f"Model loaded from {save_path}")
    
    model.eval()
    
    def run_test(loader, name):
        biased_preds, unbiased_preds, trues = [], [], []
        users_list, items_list = [], []
        
        biased_ut_dict, biased_pt_dict = {}, {}
        unbiased_ut_dict, unbiased_pt_dict = {}, {}
        
        with torch.no_grad():
            for u, i, r in loader:
                biased_pred, unbiased_pred, _ = model(u.to(device), i.to(device))
                biased_preds.append(biased_pred)
                unbiased_preds.append(unbiased_pred)
                trues.append(r.to(device).float())
                users_list.append(u.to(device))
                items_list.append(i.to(device))
                
                for idx, user_id in enumerate(u):
                    uid = user_id.item()
                    rating = r[idx].item()
                    biased_score = biased_pred[idx].item()
                    unbiased_score = unbiased_pred[idx].item()
                    
                    if uid not in biased_ut_dict:
                        biased_ut_dict[uid] = []
                        biased_pt_dict[uid] = []
                    biased_ut_dict[uid].append(rating)
                    biased_pt_dict[uid].append(biased_score)
                    
                    if uid not in unbiased_ut_dict:
                        unbiased_ut_dict[uid] = []
                        unbiased_pt_dict[uid] = []
                    unbiased_ut_dict[uid].append(rating)
                    unbiased_pt_dict[uid].append(unbiased_score)
        
        biased_preds = torch.cat(biased_preds)
        unbiased_preds = torch.cat(unbiased_preds)
        trues = torch.cat(trues)
        users_all = torch.cat(users_list)
        items_all = torch.cat(items_list)
        
        print(f"\n{'='*60}")
        print(f"[{name} Test Results]")
        print(f"{'='*60}")
        
        items_num = torch.max(items_all).item() + 1
        
        metric_list = ['BCE', 'AUC', 'UAUC', 'UNDCG', 'Recall_Precision_NDCG@']
        print(f"\nDataset '{args.dataset}' (num_items: {items_num})")
        print(f"   Metrics: BCE, AUC, UAUC, UNDCG, Precision, Recall, NDCG")
        
        biased_res = utils.metrics.evaluate(
            biased_preds, trues, metric_list,
            users=users_all, items=items_all,
            UAUC=(biased_ut_dict, biased_pt_dict),
            UNDCG=(biased_ut_dict, biased_pt_dict)
        )
        print(f"\nBiased Tower Predictions:")
        for k, v in biased_res.items():
            print(f"  {k}: {v:.4f}")
        
        unbiased_res = utils.metrics.evaluate(
            unbiased_preds, trues, metric_list,
            users=users_all, items=items_all,
            UAUC=(unbiased_ut_dict, unbiased_pt_dict),
            UNDCG=(unbiased_ut_dict, unbiased_pt_dict)
        )
        print(f"\nUnbiased Tower Predictions:")
        for k, v in unbiased_res.items():
            print(f"  {k}: {v:.4f}")
        
        return biased_res, unbiased_res
    
    run_test(test_loader, "Unbiased Test Data (Random)")
    
    run_test(biased_test_loader, "Biased Test Data (Standard)")



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
