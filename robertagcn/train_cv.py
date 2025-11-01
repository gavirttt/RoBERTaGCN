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
from trainer import train_epoch, evaluate_model, prefinetune_bert, refresh_memory_bank
from data.data_loader import read_separate_csv_data
from early_stopping import EarlyStopping
from math import ceil


def train_one_fold(model, texts, labels, train_idx, val_idx, A_torch, vocab, config, device, fold_num):
    """Train on train_idx, evaluate on val_idx with early stopping - FIXED VERSION"""
    ndocs = len(texts)
    nwords = A_torch.shape[0] - ndocs
    n_classes = max([lab for lab in labels if lab is not None]) + 1
    
    # Setup optimizer with different learning rates
    gcn_params = list(model.gcn.parameters()) + list(model.aux_clf.parameters())
    bert_params = list(model.encoder.model.parameters())
    
    all_params = [
        {'params': gcn_params, 'lr': config.get('lr_gcn', 1e-3)},
        {'params': bert_params, 'lr': config.get('lr_bert', 1e-5)}
    ]
    optimizer = Adam(all_params, weight_decay=config.get('weight_decay', 1e-5))
    loss_fn = nn.NLLLoss()
    
    # Initialize early stopping for this fold
    early_stopping = EarlyStopping(
        patience=config.get('early_stopping_patience', 5),
        verbose=config.get('verbose', False),
        delta=config.get('early_stopping_delta', 0),
        save_path=os.path.join(config.get('save_dir', 'checkpoints'), f'best_model_fold{fold_num}.pt'),
        mode='max'  # Using validation F1 for early stopping
    )
    
    best_val_f1 = 0.0
    best_epoch_metrics = None
    fold_training_history = []
    
    print(f"\n{'='*70}")
    print(f"Fold {fold_num} Training: {len(train_idx)} train, {len(val_idx)} val")
    print(f"Early stopping patience: {config.get('early_stopping_patience', 5)}")
    print(f"LR (GCN): {config.get('lr_gcn', 1e-3)}, LR (BERT): {config.get('lr_bert', 1e-5)}")
    print(f"{'='*70}")
    
    # Initialize memory bank for this fold
    model.initialize_memory_bank(ndocs, config.get('feat_dim', 768), device)
    
    # Initialize memory bank with current BERT embeddings
    refresh_memory_bank(model, texts, device, config)
    
    # Cache word features
    model.encode_and_cache_words(vocab, device, config)
    
    # FIXED: Use the unified training approach that only uses labeled data
    for epoch in range(1, config.get('epochs', 10) + 1):
        # Train for one epoch using only labeled training indices
        train_metrics = train_epoch(
            model, texts, labels, train_idx, A_torch, vocab, 
            optimizer, loss_fn, device, config, epoch, f'Fold {fold_num}'
        )
        
        # Evaluate on validation set
        val_metrics = evaluate_model(
            model, texts, labels, val_idx, A_torch, vocab, 
            n_classes, device, config
        )
        
        val_loss = val_metrics['loss']
        val_f1 = val_metrics['macro_f1']
        
        # Store training history
        epoch_history = {
            'epoch': epoch,
            'train_loss': train_metrics['loss'],
            'val_loss': val_loss,
            'val_accuracy': val_metrics['accuracy'],
            'val_f1': val_f1
        }
        fold_training_history.append(epoch_history)
        
        print(f"Fold {fold_num} Epoch {epoch:3d}/{config.get('epochs', 10)} | "
              f"Train Loss: {train_metrics['loss']:.4f} | "
              f"Val Loss: {val_loss:.4f} | "
              f"Val Acc: {val_metrics['accuracy']:.4f} | "
              f"Val F1: {val_f1:.4f}")
        
        # Early stopping check
        early_stopping(val_f1, model, epoch, config)
        
        if early_stopping.early_stop:
            print(f"Early stopping triggered for fold {fold_num} at epoch {epoch}")
            break
        
        # Update best metrics
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch_metrics = val_metrics
    
    # Load the best model for this fold before returning
    if early_stopping.early_stop and os.path.exists(early_stopping.save_path):
        print(f"Loading best model for fold {fold_num} from {early_stopping.save_path}")
        checkpoint = torch.load(early_stopping.save_path)
        model.load_state_dict(checkpoint['model_state'])
    
    # If no early stopping, use the last epoch's best metrics
    if best_epoch_metrics is None:
        best_epoch_metrics = val_metrics
    
    # Add training history to results
    best_epoch_metrics['training_history'] = fold_training_history
    best_epoch_metrics['stopped_epoch'] = epoch if early_stopping.early_stop else config.get('epochs', 10)
    best_epoch_metrics['best_epoch'] = checkpoint['epoch'] if early_stopping.early_stop else config.get('epochs', 10)
    
    return best_epoch_metrics


def run_Kfold_cv(config):
    """K-fold cross-validation using config dictionary with early stopping - OPTIMIZED"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    set_seed(config.get('seed', 42))
    
    print("="*70)
    print("BertGCN K-Fold Cross-Validation with Early Stopping")
    print("="*70)
    print(f"Device: {device}")
    print(f"Early stopping patience: {config.get('early_stopping_patience', 5)}")
    
    # Load data
    ids, texts, labels, label_map = read_separate_csv_data(
        labeled_path=config['labeled_data'],
        unlabeled_path=config.get('unlabeled_data'),
        column_mapping=config.get('column_mapping', {}),
        quickrun=(config.get('mode') == 'quickrun')
    )
    
    ndocs = len(texts)
    
    # Get labeled indices
    labeled_idx = [i for i, y in enumerate(labels) if y is not None]
    y_labeled = [labels[i] for i in labeled_idx]
    n_classes = max(y_labeled) + 1
    
    print(f"\nCV Setup: {len(labeled_idx)} labeled docs, {n_classes} classes")
    print(f"Label distribution: {np.bincount(y_labeled)}")
    
    # Build graph (transductive - includes all data)
    print("\n" + "="*70)
    print("Building graph (all data)...")
    print("="*70)
    
    A_norm, vocab, doc_word = build_text_graph(
        texts,
        max_features=config.get('max_vocab', 20000),
        min_df=config.get('min_df', 2),
        window_size=config.get('window_size', 20),
    )
    
    nwords = doc_word.shape[1]
    A_torch = sparse_to_torch_sparse_tensor(A_norm).coalesce().to(device)
    print(f"Graph: {ndocs + nwords} nodes\n")
    
    # K-fold stratified split
    skf = StratifiedKFold(n_splits=config.get('fold',10), shuffle=True, random_state=config.get('seed', 42))
    
    fold_results = []
    all_y_true = []
    all_y_pred = []
    training_histories = []
    
    for fold, (train_indices, val_indices) in enumerate(skf.split(labeled_idx, y_labeled), 1):
        # Map to original indices
        train_idx = [labeled_idx[i] for i in train_indices]
        val_idx = [labeled_idx[i] for i in val_indices]
        
        # Initialize model
        model = BertGCN(
            encoder_name=config.get('encoder', 'jcblaise/roberta-tagalog-base'),
            feat_dim=config.get('feat_dim', 768),
            gcn_hid=config.get('gcn_hid', 256),
            n_classes=n_classes,
            dropout=config.get('dropout', 0.5)
        ).to(device)
        
        # Pre-finetune BERT on training split
        temp_labels = [None] * len(texts)
        for idx in train_idx:
            temp_labels[idx] = labels[idx]
        
        model = prefinetune_bert(model, texts, temp_labels, label_map, device, config)
        
        # Train and evaluate fold with early stopping
        metrics = train_one_fold(model, texts, labels, train_idx, val_idx, 
                                A_torch, vocab, config, device, fold)
        
        fold_results.append(metrics)
        all_y_true.extend(metrics['y_true'])
        all_y_pred.extend(metrics['y_pred'])
        training_histories.append(metrics.get('training_history', []))
        
        print(f"\n{'='*70}")
        print(f"Fold {fold} Results:")
        print(f"  Stopped at epoch: {metrics.get('stopped_epoch', 'N/A')}")
        print(f"  Best epoch: {metrics.get('best_epoch', 'N/A')}")
        print(f"  Accuracy:  {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['macro_precision']:.4f}")
        print(f"  Recall:    {metrics['macro_recall']:.4f}")
        print(f"  F1:        {metrics['macro_f1']:.4f}")
        print(f"{'='*70}")
    
    # Aggregate results
    print("\n" + "="*70)
    print("K-FOLD CROSS-VALIDATION RESULTS WITH EARLY STOPPING")
    print("="*70)
    
    accuracies = [m['accuracy'] for m in fold_results]
    precisions = [m['macro_precision'] for m in fold_results]
    recalls = [m['macro_recall'] for m in fold_results]
    f1s = [m['macro_f1'] for m in fold_results]
    stopped_epochs = [m.get('stopped_epoch', config.get('epochs', 10)) for m in fold_results]
    
    print(f"\nAverage epochs trained: {np.mean(stopped_epochs):.1f} ± {np.std(stopped_epochs):.1f}")
    print(f"Accuracy:  {np.mean(accuracies):.4f} ± {np.std(accuracies):.4f}")
    print(f"Precision: {np.mean(precisions):.4f} ± {np.std(precisions):.4f}")
    print(f"Recall:    {np.mean(recalls):.4f} ± {np.std(recalls):.4f}")
    print(f"F1-Score:  {np.mean(f1s):.4f} ± {np.std(f1s):.4f}")
    
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
    
    # Save results with training histories
    if config.get('save_dir'):
        os.makedirs(config.get('save_dir'), exist_ok=True)
        
        # Save fold results
        results_df = pd.DataFrame({
            'fold': range(1, config.get('fold',10)+1),
            'accuracy': accuracies,
            'precision': precisions,
            'recall': recalls,
            'f1': f1s,
            'stopped_epoch': stopped_epochs,
            'best_epoch': [m.get('best_epoch', config.get('epochs', 10)) for m in fold_results]
        })
        
        save_path = os.path.join(config.get('save_dir'), '10fold_results.csv')
        results_df.to_csv(save_path, index=False)
        print(f"\nResults saved: {save_path}")
        
        # Save training histories
        history_df = pd.DataFrame()
        for fold_num, history in enumerate(training_histories, 1):
            for epoch_data in history:
                epoch_data['fold'] = fold_num
                history_df = pd.concat([history_df, pd.DataFrame([epoch_data])], ignore_index=True)
        
        history_path = os.path.join(config.get('save_dir'), 'training_histories.csv')
        history_df.to_csv(history_path, index=False)
        print(f"Training histories saved: {history_path}")
    
    print("\n" + "="*70)
    print("Cross-validation completed!")
    print("="*70)

    return fold_results


def sparse_to_torch_sparse_tensor(sparse_mx):
    """Convert scipy sparse matrix to torch sparse tensor"""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = sparse_mx.shape
    return torch.sparse_coo_tensor(indices, values, torch.Size(shape))