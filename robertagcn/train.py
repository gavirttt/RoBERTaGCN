import os
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.optim import Adam
from scipy import sparse

from utils import set_seed
from data.build_graph import build_text_graph
from models.bertgcn import BertGCN
from trainer import train_epoch, evaluate_model, prefinetune_bert, refresh_memory_bank
from early_stopping import EarlyStopping
from data.data_loader import read_separate_csv_data


def sparse_to_torch_sparse_tensor(sparse_mx):
    """Convert scipy sparse matrix to torch sparse tensor"""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = sparse_mx.shape
    return torch.sparse_coo_tensor(indices, values, torch.Size(shape))


def run_training(config):
    """Main training function using unified trainer"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    set_seed(config.get('seed', 42))
    
    print("="*70)
    print("BertGCN Training (Optimized Implementation)")
    print("="*70)
    print(f"Device: {device}")
    print(f"Mode: {config.get('mode', 'train')}")
    
    # Load data
    ids, texts, labels, label_map = read_separate_csv_data(
        labeled_path=config['labeled_data'],
        unlabeled_path=config.get('unlabeled_data'),
        column_mapping=config.get('column_mapping', {}),
        quickrun=(config.get('mode') == 'quickrun')
    )
    
    ndocs = len(texts)
    
    # Get labeled and unlabeled indices
    labeled_idx = [i for i, y in enumerate(labels) if y is not None]
    unlabeled_idx = [i for i, y in enumerate(labels) if y is None]
    
    # For training, we need labeled data
    if len(labeled_idx) == 0:
        raise ValueError("No labeled data found for training!")
    
    y_labeled = [labels[i] for i in labeled_idx]
    n_classes = max(y_labeled) + 1
    
    print(f"\nData: {ndocs} total docs ({len(labeled_idx)} labeled, {len(unlabeled_idx)} unlabeled)")
    print(f"Classes: {n_classes}")
    print(f"Label distribution: {np.bincount(y_labeled)}")
    
    # Build graph
    print("\n" + "="*70)
    print("Building graph (optimized)...")
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
    
    # Initialize model
    model = BertGCN(
        encoder_name=config.get('encoder', 'jcblaise/roberta-tagalog-base'),
        feat_dim=config.get('feat_dim', 768),
        gcn_hid=config.get('gcn_hid', 256),
        n_classes=n_classes,
        dropout=config.get('dropout', 0.5)
    ).to(device)
    
    # Setup optimizer with different learning rates
    gcn_params = list(model.gcn.parameters()) + list(model.aux_clf.parameters())
    bert_params = list(model.encoder.model.parameters())
    
    all_params = [
        {'params': gcn_params, 'lr': config.get('lr_gcn', 1e-3)},
        {'params': bert_params, 'lr': config.get('lr_bert', 1e-5)}
    ]
    optimizer = Adam(all_params, weight_decay=config.get('weight_decay', 1e-5))
    loss_fn = nn.NLLLoss()
    
    # Pre-finetune BERT
    model = prefinetune_bert(model, texts, labels, label_map, device, config)
    
    # Initialize memory bank
    model.initialize_memory_bank(ndocs, config.get('feat_dim', 768), device)
    
    # Initialize memory bank with BERT embeddings
    refresh_memory_bank(model, texts, device, config)
    
    # Cache word features
    model.encode_and_cache_words(vocab, device, config)
    
    # Setup early stopping
    early_stopping = EarlyStopping(
        patience=config.get('early_stopping_patience', 5),
        verbose=config.get('verbose', False),
        delta=config.get('early_stopping_delta', 0),
        save_path=os.path.join(config.get('save_dir', 'checkpoints'), 'best_model.pt'),
        mode='max'  # Using validation F1 for early stopping
    )
    
    # Training loop
    print("\n" + "="*70)
    print("Starting main training...")
    print("="*70)
    
    best_val_f1 = 0.0
    training_history = []
    
    for epoch in range(1, config.get('epochs', 10) + 1):
        # Train for one epoch
        train_metrics = train_epoch(
            model, texts, labels, labeled_idx, A_torch, vocab, 
            optimizer, loss_fn, device, config, epoch, 'Training'
        )
        
        # Evaluate on validation set (using labeled data)
        val_metrics = evaluate_model(
            model, texts, labels, labeled_idx, A_torch, vocab, 
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
        training_history.append(epoch_history)
        
        print(f"Epoch {epoch:3d}/{config.get('epochs', 10)} | "
              f"Train Loss: {train_metrics['loss']:.4f} | "
              f"Val Loss: {val_loss:.4f} | "
              f"Val Acc: {val_metrics['accuracy']:.4f} | "
              f"Val F1: {val_f1:.4f}")
        
        # Early stopping check
        early_stopping(val_f1, model, epoch, config)
        
        if early_stopping.early_stop:
            print(f"Early stopping triggered at epoch {epoch}")
            break
        
        # Update best metrics
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
    
    # Load best model
    if early_stopping.early_stop and os.path.exists(early_stopping.save_path):
        print(f"Loading best model from {early_stopping.save_path}")
        checkpoint = torch.load(early_stopping.save_path)
        model.load_state_dict(checkpoint['model_state'])
    
    print(f"\nTraining completed. Best validation F1: {best_val_f1:.4f}")
    
    return {
        'model': model,
        'vocab': vocab,
        'A_torch': A_torch,
        'training_history': training_history,
        'best_val_f1': best_val_f1
    }


if __name__ == "__main__":
    # Example configuration
    config = {
        'labeled_data': 'data/labeled.csv',
        'unlabeled_data': 'data/unlabeled.csv',
        'encoder': 'jcblaise/roberta-tagalog-base',
        'feat_dim': 768,
        'gcn_hid': 256,
        'n_classes': 2,
        'dropout': 0.5,
        'lr_bert': 1e-5,
        'lr_gcn': 1e-3,
        'batch_size': 32,
        'bert_batch': 32,
        'epochs': 10,
        'early_stopping_patience': 5,
        'max_vocab': 20000,
        'window_size': 20,
        'lmbda': 0.7,
        'prefinetune': True,
        'prefinetune_epochs': 2,
        'prefinetune_lr': 1e-6,
        'seed': 42
    }
    
    results = run_training(config)