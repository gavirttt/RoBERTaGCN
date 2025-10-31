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
from train import (sparse_to_torch_sparse_tensor, prefinetune_bert, evaluate_model_with_loss)
from data.data_loader import read_separate_csv_data
from early_stopping import EarlyStopping  # Add this import
from math import ceil


def train_one_fold(model, texts, labels, train_idx, val_idx, A_torch, vocab, config, device, fold_num):
    """Train on train_idx, evaluate on val_idx with early stopping"""
    ndocs = len(texts)
    nwords = A_torch.shape[0] - ndocs
    n_classes = max([lab for lab in labels if lab is not None]) + 1
    
    # Setup optimizer
    gcn_params = list(model.gcn.parameters()) + list(model.aux_clf.parameters())
    bert_params = list(model.encoder.model.parameters())
    
    all_params = [
        {'params': gcn_params, 'lr': config.get('lr_gcn', 1e-3)},
        {'params': bert_params, 'lr': config.get('lr_bert', 1e-5)}
    ]
    optimizer = Adam(all_params, weight_decay=config.get('weight_decay', 1e-5))
    loss_fn = nn.CrossEntropyLoss()
    
    # Initialize early stopping for this fold
    early_stopping = EarlyStopping(
        patience=config.get('early_stopping_patience', 5),
        verbose=config.get('verbose', False),
        delta=config.get('early_stopping_delta', 0),
        save_path=os.path.join(config.get('save_dir', 'checkpoints'), f'best_model_fold{fold_num}.pt')
    )
    
    best_val_f1 = 0.0
    best_epoch_metrics = None
    fold_training_history = []
    
    print(f"\n{'='*70}")
    print(f"Fold {fold_num} Training: {len(train_idx)} train, {len(val_idx)} val")
    print(f"Early stopping patience: {config.get('early_stopping_patience', 5)}")
    print(f"{'='*70}")
    
    for epoch in range(1, config.get('epochs', 10) + 1):
        model.train()
        
        # Build memory bank for ALL documents
        membank = np.zeros((ndocs, config.get('feat_dim', 768)), dtype=np.float32)
        with torch.no_grad():
            for i in range(0, ndocs, config.get('bert_batch', 32)):
                texts_batch = texts[i:i+config.get('bert_batch', 32)]
                feats = model.encoder.encode_batch(texts_batch, device=device, 
                                                   max_len=config.get('max_len', 64))
                membank[i:i+config.get('bert_batch', 32)] = feats.cpu().numpy()
        
        X_docs = torch.tensor(membank, dtype=torch.float32, device=device)
        
        # Initialize word features using BERT encoder
        # vocab is available from run_Kfold_cv scope
        X_words = model.encoder.encode_words(vocab, device=device, 
                                             max_len=config.get('max_len', 64), 
                                             batch_size=config.get('bert_batch', 32))
        
        X_full_base = torch.cat([X_docs, X_words], dim=0)
        
        # Training loop
        train_idx_shuffled = train_idx.copy()
        np.random.shuffle(train_idx_shuffled)
        losses = []
        
        pbar = tqdm(range(0, len(train_idx_shuffled), config.get('batch_size', 32)), 
                   desc=f'Fold {fold_num} Epoch {epoch}', leave=False)
        
        for i in pbar:
            batch_idx = train_idx_shuffled[i:i+config.get('batch_size', 32)]
            texts_batch = [texts[j] for j in batch_idx]
            
            # Update embeddings
            feats_batch = model.encoder.encode_batch(texts_batch, device=device, 
                                                     max_len=config.get('max_len', 64))
            logits_bert = model.aux_clf(feats_batch)
            
            # Construct X
            X = X_full_base.clone()
            X = X.detach()
            X[batch_idx, :] = feats_batch
            
            # GCN forward
            gcn_logits_all = model.gcn_forward(A_torch, X)
            doc_logits_batch = gcn_logits_all[batch_idx, :]
            
            # Loss
            target = torch.tensor([labels[j] for j in batch_idx], dtype=torch.long, device=device)
            gcn_pred = doc_logits_batch
            bert_pred = logits_bert
            
            logits = config.get('lmbda', 0.7) * gcn_pred + \
                    (1.0 - config.get('lmbda', 0.7)) * bert_pred
            loss = loss_fn(logits, target)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            losses.append(loss.item())
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_loss = float(np.mean(losses)) if losses else 0.0
        
        # Evaluation every epoch for early stopping
        val_metrics = evaluate_fold_with_loss(model, texts, labels, val_idx, A_torch, vocab, config, device)
        val_loss = val_metrics['val_loss']
        
        # Store training history
        epoch_history = {
            'epoch': epoch,
            'train_loss': avg_loss,
            'val_loss': val_loss,
            'val_accuracy': val_metrics['accuracy'],
            'val_f1': val_metrics['macro_f1']
        }
        fold_training_history.append(epoch_history)
        
        print(f"Fold {fold_num} Epoch {epoch:3d}/{config.get('epochs', 10)} | "
              f"Train Loss: {avg_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | "
              f"Val Acc: {val_metrics['accuracy']:.4f} | "
              f"Val F1: {val_metrics['macro_f1']:.4f}")
        
        # Early stopping check
        early_stopping(val_loss, model, epoch, config)
        
        if early_stopping.early_stop:
            print(f"Early stopping triggered for fold {fold_num} at epoch {epoch}")
            break
        
        # Update best metrics
        if val_metrics['macro_f1'] > best_val_f1:
            best_val_f1 = val_metrics['macro_f1']
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


def evaluate_fold_with_loss(model, texts, labels, eval_idx, A_torch, vocab, config, device):
    """Evaluate on eval_idx and return metrics including loss"""
    model.eval()
    ndocs = len(texts)
    nwords = A_torch.shape[0] - ndocs
    n_classes = max([lab for lab in labels if lab is not None]) + 1
    
    with torch.no_grad():
        # Recompute memory bank for documents
        membank_eval = np.zeros((ndocs, config.get('feat_dim', 768)), dtype=np.float32)
        for i in range(0, ndocs, config.get('bert_batch', 32)):
            texts_batch = texts[i:i+config.get('bert_batch', 32)]
            feats = model.encoder.encode_batch(texts_batch, device=device, 
                                              max_len=config.get('max_len', 64))
            membank_eval[i:i+config.get('bert_batch', 32)] = feats.cpu().numpy()
        
        X_docs = torch.tensor(membank_eval, dtype=torch.float32, device=device)
        
        # Recompute word features (vocab is available from run_Kfold_cv scope)
        X_words_eval = model.encoder.encode_words(vocab, device=device, 
                                                  max_len=config.get('max_len', 64), 
                                                  batch_size=config.get('bert_batch', 32))
        
        X_full = torch.cat([X_docs, X_words_eval], dim=0)
        
        # Predictions
        gcn_logits_all = model.gcn_forward(A_torch, X_full)
        doc_logits = gcn_logits_all[:ndocs, :]
        
        bert_logits = []
        for i in range(0, ndocs, config.get('bert_batch', 32)):
            bl = model.aux_clf(model.encoder.encode_batch(
                texts[i:i+config.get('bert_batch', 32)], 
                device=device, 
                max_len=config.get('max_len', 64)
            ))
            bert_logits.append(bl.cpu())
        bert_logits = torch.cat(bert_logits, dim=0).to(device)
        
        final_logits = config.get('lmbda', 0.7) * doc_logits + \
                      (1.0 - config.get('lmbda', 0.7)) * bert_logits
        
        y_true = np.array([labels[i] for i in eval_idx])
        y_pred = torch.argmax(final_logits[eval_idx, :], dim=1).cpu().numpy()
        
        # Calculate validation loss
        loss_fn = nn.CrossEntropyLoss()
        y_true_tensor = torch.tensor(y_true, dtype=torch.long, device=device)
        y_pred_logits = final_logits[eval_idx, :]
        val_loss = loss_fn(y_pred_logits, y_true_tensor).item()
        
        # Metrics
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, average=None, zero_division=0
        )
        
        metrics = {
            'val_loss': val_loss,
            'accuracy': (y_pred == y_true).mean(),
            'macro_precision': precision.mean(),
            'macro_recall': recall.mean(),
            'macro_f1': f1.mean(),
            'per_class_precision': precision,
            'per_class_recall': recall,
            'per_class_f1': f1,
            'y_true': y_true,
            'y_pred': y_pred
        }
        
        return metrics


def evaluate_fold(model, texts, labels, eval_idx, A_torch, vocab, config, device):
    """Original evaluate_fold without loss calculation (for backward compatibility)"""
    metrics = evaluate_fold_with_loss(model, texts, labels, eval_idx, A_torch, vocab, config, device)
    # Remove val_loss for compatibility with existing code
    metrics.pop('val_loss', None)
    return metrics


def run_Kfold_cv(config):
    """K-fold cross-validation using config dictionary with early stopping"""
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
