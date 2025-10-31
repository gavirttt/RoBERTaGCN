import os
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.optim import Adam
from scipy import sparse
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_fscore_support
import pandas as pd

from utils import set_seed
from data.build_graph import build_text_graph
from models.bertgcn import BertGCN
from train import sparse_to_torch_sparse_tensor, read_csv_data, prefinetune_bert
from math import ceil


def train_one_fold(model, texts, labels, train_idx, val_idx, A_torch, args, device, fold_num):
    """
    Train on train_idx, evaluate on val_idx
    Graph includes ALL nodes (transductive), but loss only on train_idx
    """
    ndocs = len(texts)
    nwords = A_torch.shape[0] - ndocs
    n_classes = max([lab for lab in labels if lab is not None]) + 1
    
    # Setup optimizer with different LRs
    gcn_params = list(model.gcn.parameters()) + list(model.aux_clf.parameters())
    bert_params = list(model.encoder.model.parameters())
    
    all_params = [
        {'params': gcn_params, 'lr': args.lr_gcn},
        {'params': bert_params, 'lr': args.lr_bert}
    ]
    optimizer = Adam(all_params, weight_decay=args.weight_decay)
    loss_fn = nn.CrossEntropyLoss()
    
    best_val_f1 = 0.0
    best_epoch_metrics = None
    
    print(f"\n{'='*70}")
    print(f"Fold {fold_num} Training: {len(train_idx)} train, {len(val_idx)} val")
    print(f"{'='*70}")
    
    for epoch in range(1, args.epochs + 1):
        model.train()
        
        # Build memory bank for ALL documents
        membank = np.zeros((ndocs, args.feat_dim), dtype=np.float32)
        with torch.no_grad():
            for i in range(0, ndocs, args.bert_batch):
                texts_batch = texts[i:i+args.bert_batch]
                feats = model.encoder.encode_batch(texts_batch, device=device, max_len=args.max_len)
                membank[i:i+args.bert_batch] = feats.cpu().numpy()
        
        X_docs = torch.tensor(membank, dtype=torch.float32, device=device)
        X_words = torch.zeros((nwords, args.feat_dim), dtype=torch.float32, device=device)
        X_full_base = torch.cat([X_docs, X_words], dim=0)
        
        # Training loop - only use train_idx for loss
        train_idx_shuffled = train_idx.copy()
        np.random.shuffle(train_idx_shuffled)
        losses = []
        
        for i in range(0, len(train_idx_shuffled), args.batch_size):
            batch_idx = train_idx_shuffled[i:i+args.batch_size]
            texts_batch = [texts[j] for j in batch_idx]
            
            # Update embeddings for batch
            feats_batch = model.encoder.encode_batch(texts_batch, device=device, max_len=args.max_len)
            logits_bert = model.aux_clf(feats_batch)
            
            # Construct X with updated batch
            X = X_full_base.clone()
            X = X.detach()
            X[batch_idx, :] = feats_batch  # Gradients flow through batch
            
            # GCN forward
            gcn_logits_all = model.gcn_forward(A_torch, X)
            doc_logits_batch = gcn_logits_all[batch_idx, :]
            
            # Loss only on train_idx within batch
            target = torch.tensor([labels[j] for j in batch_idx], dtype=torch.long, device=device)
            gcn_pred = doc_logits_batch
            bert_pred = logits_bert
            
            # Equation 6: Z = λZ_GCN + (1-λ)Z_BERT
            logits = args.lmbda * gcn_pred + (1.0 - args.lmbda) * bert_pred
            loss = loss_fn(logits, target)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            losses.append(loss.item())
        
        avg_loss = float(np.mean(losses)) if losses else 0.0
        
        # Evaluation on validation set
        if epoch % 5 == 0 or epoch == args.epochs:
            val_metrics = evaluate_fold(model, texts, labels, val_idx, A_torch, args, device)
            
            print(f"Epoch {epoch:3d} | Loss: {avg_loss:.4f} | "
                  f"Val Acc: {val_metrics['accuracy']:.4f} | "
                  f"Val Macro-F1: {val_metrics['macro_f1']:.4f}")
            
            # Track best model
            if val_metrics['macro_f1'] > best_val_f1:
                best_val_f1 = val_metrics['macro_f1']
                best_epoch_metrics = val_metrics
    
    return best_epoch_metrics


def evaluate_fold(model, texts, labels, eval_idx, A_torch, args, device):
    """Evaluate on eval_idx (validation or test set)"""
    model.eval()
    ndocs = len(texts)
    nwords = A_torch.shape[0] - ndocs
    n_classes = max([lab for lab in labels if lab is not None]) + 1
    
    with torch.no_grad():
        # Recompute memory bank
        membank_eval = np.zeros((ndocs, args.feat_dim), dtype=np.float32)
        for i in range(0, ndocs, args.bert_batch):
            texts_batch = texts[i:i+args.bert_batch]
            feats = model.encoder.encode_batch(texts_batch, device=device, max_len=args.max_len)
            membank_eval[i:i+args.bert_batch] = feats.cpu().numpy()
        
        X_docs = torch.tensor(membank_eval, dtype=torch.float32, device=device)
        X_words = torch.zeros((nwords, args.feat_dim), dtype=torch.float32, device=device)
        X_full = torch.cat([X_docs, X_words], dim=0)
        
        # GCN predictions
        gcn_logits_all = model.gcn_forward(A_torch, X_full)
        doc_logits = gcn_logits_all[:ndocs, :]
        
        # BERT predictions
        bert_logits = []
        for i in range(0, ndocs, args.bert_batch):
            bl = model.aux_clf(model.encoder.encode_batch(texts[i:i+args.bert_batch], device=device, max_len=args.max_len))
            bert_logits.append(bl.cpu())
        bert_logits = torch.cat(bert_logits, dim=0).to(device)
        
        # Final predictions (Equation 6)
        final_logits = args.lmbda * doc_logits + (1.0 - args.lmbda) * bert_logits
        
        # Evaluate only on eval_idx
        y_true = np.array([labels[i] for i in eval_idx])
        y_pred = torch.argmax(final_logits[eval_idx, :], dim=1).cpu().numpy()
        
        # Calculate metrics (Paper uses macro-averaging)
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, average=None, zero_division=0
        )
        
        macro_precision = precision.mean()
        macro_recall = recall.mean()
        macro_f1 = f1.mean()
        
        accuracy = (y_pred == y_true).mean()
        
        metrics = {
            'accuracy': accuracy,
            'macro_precision': macro_precision,
            'macro_recall': macro_recall,
            'macro_f1': macro_f1,
            'per_class_precision': precision,
            'per_class_recall': recall,
            'per_class_f1': f1,
            'y_true': y_true,
            'y_pred': y_pred
        }
        
        return metrics


def run_10fold_cv(args):
    """Main 10-fold cross-validation function"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    set_seed(args.seed)
    
    print(f"Using device: {device}")
    print(f"Running 10-Fold Cross-Validation")
    
    # Load data
    ids, texts, labels, label_map = read_csv_data(args.data, quickrun=args.quickrun)
    ndocs = len(texts)
    
    # Get labeled indices and their labels
    labeled_idx = [i for i, y in enumerate(labels) if y is not None]
    y_labeled = [labels[i] for i in labeled_idx]
    n_classes = max(y_labeled) + 1
    
    print(f"\nDataset: {ndocs} total docs, {len(labeled_idx)} labeled, {n_classes} classes")
    print(f"Label distribution: {np.bincount(y_labeled)}")
    
    # Build graph once (transductive - includes all data)
    print("\n" + "="*70)
    print("Building text graph (all data)...")
    print("="*70)
    A_norm, vocab, doc_word = build_text_graph(
        texts, 
        max_features=args.max_vocab, 
        min_df=args.min_df, 
        window_size=args.window_size
    )
    nwords = doc_word.shape[1]
    A_torch = sparse_to_torch_sparse_tensor(A_norm).coalesce().to(device)
    
    # 10-fold stratified split (only on labeled data)
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=args.seed)
    
    fold_results = []
    all_y_true = []
    all_y_pred = []
    
    for fold, (train_indices, val_indices) in enumerate(skf.split(labeled_idx, y_labeled), 1):
        # Map back to original document indices
        train_idx = [labeled_idx[i] for i in train_indices]
        val_idx = [labeled_idx[i] for i in val_indices]
        
        # Initialize fresh model for each fold
        model = BertGCN(
            encoder_name=args.encoder, 
            feat_dim=args.feat_dim, 
            gcn_hid=args.gcn_hid, 
            n_classes=n_classes, 
            dropout=args.dropout
        ).to(device)
        
        # Pre-finetune BERT on training split
        train_texts = [texts[i] for i in train_idx]
        train_labels = [labels[i] for i in train_idx]
        
        # Create temporary labels list for pre-finetuning
        temp_labels = [None] * len(texts)
        for i, idx in enumerate(train_idx):
            temp_labels[idx] = train_labels[i]
        
        model = prefinetune_bert(model, texts, temp_labels, label_map, device, args)
        
        # Train and evaluate this fold
        metrics = train_one_fold(model, texts, labels, train_idx, val_idx, A_torch, args, device, fold)
        
        fold_results.append(metrics)
        all_y_true.extend(metrics['y_true'])
        all_y_pred.extend(metrics['y_pred'])
        
        print(f"\nFold {fold} Results:")
        print(f"  Accuracy:        {metrics['accuracy']:.4f}")
        print(f"  Macro-Precision: {metrics['macro_precision']:.4f}")
        print(f"  Macro-Recall:    {metrics['macro_recall']:.4f}")
        print(f"  Macro-F1:        {metrics['macro_f1']:.4f}")
    
    # Aggregate results across folds
    print("\n" + "="*70)
    print("10-Fold Cross-Validation Results")
    print("="*70)
    
    accuracies = [m['accuracy'] for m in fold_results]
    macro_precisions = [m['macro_precision'] for m in fold_results]
    macro_recalls = [m['macro_recall'] for m in fold_results]
    macro_f1s = [m['macro_f1'] for m in fold_results]
    
    # Paper format: mean ± std
    print(f"\nAccuracy:        {np.mean(accuracies):.4f} ± {np.std(accuracies):.4f}")
    print(f"Macro-Precision: {np.mean(macro_precisions):.4f} ± {np.std(macro_precisions):.4f}")
    print(f"Macro-Recall:    {np.mean(macro_recalls):.4f} ± {np.std(macro_recalls):.4f}")
    print(f"Macro-F1:        {np.mean(macro_f1s):.4f} ± {np.std(macro_f1s):.4f}")
    
    # Overall confusion matrix
    print("\n" + "="*70)
    print("Overall Confusion Matrix (All Folds)")
    print("="*70)
    cm = confusion_matrix(all_y_true, all_y_pred)
    print(cm)
    
    # Classification report
    print("\n" + "="*70)
    print("Overall Classification Report")
    print("="*70)
    print(classification_report(
        all_y_true, all_y_pred, 
        target_names=[f'Class {i}' for i in range(n_classes)],
        zero_division=0
    ))
    
    # Save results
    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)
        results_df = pd.DataFrame({
            'fold': range(1, 11),
            'accuracy': accuracies,
            'macro_precision': macro_precisions,
            'macro_recall': macro_recalls,
            'macro_f1': macro_f1s
        })
        results_df.to_csv(os.path.join(args.save_dir, '10fold_results.csv'), index=False)
        print(f"\nResults saved to {args.save_dir}/10fold_results.csv")
    
    return fold_results