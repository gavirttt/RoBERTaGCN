import os
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.optim import Adam
from scipy import sparse
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd # Re-adding pandas as it's used in plot_confusion_matrix
from utils import set_seed
from data.build_graph import build_text_graph
from models.bertgcn import BertGCN
from early_stopping import EarlyStopping
from data.data_loader import read_separate_csv_data # Import the new data loader functions

from math import ceil


def sparse_to_torch_sparse_tensor(sparse_mx):
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = sparse_mx.shape
    return torch.sparse_coo_tensor(indices, values, torch.Size(shape))


def prefinetune_bert(model, texts, labels, label_map, device, config):
    """Pre-finetune BERT on labeled data before joint training"""
    if not config.get('prefinetune', True):
        print("Pre-finetuning disabled by configuration")
        return model
    
    print("="*70)
    print("Pre-finetuning BERT on labeled data...")
    print("="*70)
    
    labeled_texts = [texts[i] for i, label in enumerate(labels) if label is not None]
    labeled_labels = [label for label in labels if label is not None]
    
    if len(labeled_texts) == 0:
        print("No labeled data for pre-finetuning, skipping...")
        return model
    
    print(f"Pre-finetuning on {len(labeled_texts)} labeled examples")
    
    labeled_labels_tensor = torch.tensor(labeled_labels, dtype=torch.long, device=device)
    
    bert_lr = config.get('prefinetune_lr', 1e-6)
    epochs = config.get('prefinetune_epochs', 2)
    bert_batch = config.get('bert_batch', 32)
    
    bert_optimizer = Adam(model.encoder.model.parameters(), lr=bert_lr, 
                          weight_decay=config.get('weight_decay', 1e-5))
    loss_fn = nn.CrossEntropyLoss()
    
    model.train()
    n_batches = ceil(len(labeled_texts) / bert_batch)
    
    for epoch in range(epochs):
        total_loss = 0
        indices = list(range(len(labeled_texts)))
        np.random.shuffle(indices)
        
        pbar = tqdm(range(0, len(labeled_texts), bert_batch), 
                   desc=f'Pre-finetune Epoch {epoch+1}/{epochs}')
        
        for i in pbar:
            batch_indices = indices[i:i+bert_batch]
            batch_texts = [labeled_texts[idx] for idx in batch_indices]
            batch_labels = labeled_labels_tensor[batch_indices]
            
            feats = model.encoder.encode_batch(batch_texts, device=device, 
                                              max_len=config.get('max_len', 64))
            logits = model.aux_clf(feats)
            
            loss = loss_fn(logits, batch_labels)
            
            bert_optimizer.zero_grad()
            loss.backward()
            bert_optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_loss = total_loss / n_batches
        print(f"Pre-finetune Epoch {epoch+1}/{epochs} avg_loss={avg_loss:.4f}")
    
    print("BERT pre-finetuning completed\n")
    return model


def run_training(config):
    """Main training function using config dictionary"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    set_seed(config.get('seed', 42))
    
    print("="*70)
    print("BertGCN Training")
    print("="*70)
    print(f"Device: {device}")
    print(f"Mode: {config.get('mode', 'train')}")
    
    # Load data (, social_edges)
    ids, texts, labels, label_map = read_separate_csv_data(
        labeled_path=config['labeled_data'],
        unlabeled_path=config.get('unlabeled_data'),
        column_mapping=config.get('column_mapping', {}),
        quickrun=(config.get('mode') == 'quickrun')
    )
    
    ndocs = len(texts)
    labeled_idx = [i for i, y in enumerate(labels) if y is not None]
    unlabeled_idx = [i for i, y in enumerate(labels) if y is None]
    
    if len(labeled_idx) > 0:
        n_classes = max([lab for lab in labels if lab is not None]) + 1
    else:
        raise ValueError("No labeled data found!")
    
    # Build graph
    print("\n" + "="*70)
    print("Building heterogeneous graph...")
    print("="*70)
    
    A_norm, vocab, doc_word = build_text_graph(
        texts,
        max_features=config.get('max_vocab', 20000),
        min_df=config.get('min_df', 2),
        window_size=config.get('window_size', 20)
    )
    
    nwords = doc_word.shape[1]
    N_total = ndocs + nwords
    A_torch = sparse_to_torch_sparse_tensor(A_norm).coalesce().to(device)
    
    print(f"\nGraph: {N_total} nodes ({ndocs} docs + {nwords} words)")
    print("="*70)
    
    # Initialize model
    model = BertGCN(
        encoder_name=config.get('encoder', 'jcblaise/roberta-tagalog-base'),
        feat_dim=config.get('feat_dim', 768),
        gcn_hid=config.get('gcn_hid', 256),
        n_classes=n_classes,
        dropout=config.get('dropout', 0.5)
    ).to(device)
    
    print(f"\nModel initialized: {config.get('encoder')}")
    print(f"  Feature dim: {config.get('feat_dim')}")
    print(f"  GCN hidden: {config.get('gcn_hid')}")
    print(f"  Classes: {n_classes}")
    print(f"  Dropout: {config.get('dropout')}")
    
    # Pre-finetune BERT
    model = prefinetune_bert(model, texts, labels, label_map, device, config)
    
    # Setup optimizer
    gcn_params = list(model.gcn.parameters()) + list(model.aux_clf.parameters())
    bert_params = list(model.encoder.model.parameters())
    
    all_params = [
        {'params': gcn_params, 'lr': config.get('lr_gcn', 1e-3)},
        {'params': bert_params, 'lr': config.get('lr_bert', 1e-5)}
    ]
    optimizer = Adam(all_params, weight_decay=config.get('weight_decay', 1e-5))
    loss_fn = nn.CrossEntropyLoss()
    
    # Initialize early stopping
    early_stopping = EarlyStopping(
        patience=config.get('early_stopping_patience', 5),
        verbose=True,
        delta=config.get('early_stopping_delta', 0),
        save_path=os.path.join(config.get('save_dir', 'checkpoints'), 'best_model.pt')
    )
    
    print("\n" + "="*70)
    print("Starting training...")
    print("="*70)
    print(f"Epochs: {config.get('epochs', 10)}")
    print(f"Batch size: {config.get('batch_size', 32)}")
    print(f"BERT batch: {config.get('bert_batch', 32)}")
    print(f"LR (GCN): {config.get('lr_gcn', 1e-3)}")
    print(f"LR (BERT): {config.get('lr_bert', 1e-5)}")
    print(f"Lambda: {config.get('lmbda', 0.7)}")
    print(f"Social weight: {config.get('social_weight', 1.0)}")
    print(f"Early stopping patience: {config.get('early_stopping_patience', 5)}")
    print("="*70 + "\n")
    
    doc_indices = list(range(ndocs))
    
    for epoch in range(1, config.get('epochs', 10) + 1):
        model.train()
        
        # Build memory bank
        print(f"Epoch {epoch}/{config.get('epochs', 10)}: Building memory bank...")
        membank = np.zeros((ndocs, config.get('feat_dim', 768)), dtype=np.float32)
        
        with torch.no_grad():
            for i in tqdm(range(0, ndocs, config.get('bert_batch', 32)), 
                         desc='Memory Bank', leave=False):
                texts_batch = texts[i:i+config.get('bert_batch', 32)]
                feats = model.encoder.encode_batch(texts_batch, device=device, 
                                                   max_len=config.get('max_len', 64))
                membank[i:i+config.get('bert_batch', 32)] = feats.cpu().numpy()
        
        # Build base feature matrix
        X_docs = torch.tensor(membank, dtype=torch.float32, device=device)
        
        # Initialize word features using BERT encoder
        print(f"Epoch {epoch}/{config.get('epochs', 10)}: Encoding word features...")
        X_words = model.encoder.encode_words(vocab, device=device, 
                                             max_len=config.get('max_len', 64), 
                                             batch_size=config.get('bert_batch', 32))
        
        X_full_base = torch.cat([X_docs, X_words], dim=0)
        
        # Training loop
        np.random.shuffle(doc_indices)
        losses = []
        
        pbar = tqdm(range(0, ndocs, config.get('batch_size', 32)), 
                   desc=f'Epoch {epoch}', leave=True)
        
        for i in pbar:
            batch_idx = doc_indices[i:i+config.get('batch_size', 32)]
            texts_batch = [texts[j] for j in batch_idx]
            
            # Update embeddings for batch
            feats_batch = model.encoder.encode_batch(texts_batch, device=device, 
                                                     max_len=config.get('max_len', 64))
            logits_bert = model.aux_clf(feats_batch)
            
            # Construct X with updated batch
            X = X_full_base.clone()
            X = X.detach()
            X[batch_idx, :] = feats_batch  # Gradients flow through batch
            
            # GCN forward
            gcn_logits_all = model.gcn_forward(A_torch, X)
            doc_logits_batch = gcn_logits_all[batch_idx, :]
            
            # Loss only on labeled within batch
            batch_labels = [labels[j] for j in batch_idx]
            labeled_locs = [k for k, v in enumerate(batch_labels) if v is not None]
            
            if len(labeled_locs) == 0:
                continue
            
            target = torch.tensor([batch_labels[k] for k in labeled_locs], 
                                 dtype=torch.long, device=device)
            gcn_pred = doc_logits_batch[labeled_locs]
            bert_pred = logits_bert[labeled_locs]
            
            # Equation 6: Z = λZ_GCN + (1-λ)Z_BERT
            logits = config.get('lmbda', 0.7) * gcn_pred + \
                    (1.0 - config.get('lmbda', 0.7)) * bert_pred
            loss = loss_fn(logits, target)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            losses.append(loss.item())
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_loss = float(np.mean(losses)) if losses else 0.0
        print(f"Epoch {epoch} completed | Avg Loss: {avg_loss:.4f}")
        
        # Evaluation - get validation loss for early stopping
        val_loss = evaluate_model_with_loss(model, texts, labels, labeled_idx, A_torch, 
                                           vocab, n_classes, device, config, epoch)
        
        # Early stopping check
        early_stopping(val_loss, model, epoch, config)
        
        if early_stopping.early_stop:
            print(f"\nEarly stopping triggered at epoch {epoch}")
            break
        
        # Save checkpoint (optional - you might want to disable this if using early stopping)
        if not config.get('save_best_only', False):
            save_checkpoint(model, label_map, vocab, config, epoch)
    
    print("\n" + "="*70)
    print("Training completed!")
    if early_stopping.early_stop:
        print(f"Best model saved at: {early_stopping.save_path}")
        print(f"Best validation loss: {early_stopping.val_loss_min:.6f}")
    print("="*70)

def evaluate_model_with_loss(model, texts, labels, labeled_idx, A_torch, vocab, 
                           n_classes, device, config, epoch):
    """Evaluate model and return validation loss"""
    model.eval()
    ndocs = len(texts)
    nwords = A_torch.shape[0] - ndocs
    
    with torch.no_grad():
        # Recompute memory bank for documents
        membank_eval = np.zeros((ndocs, config.get('feat_dim', 768)), dtype=np.float32)
        
        for i in tqdm(range(0, ndocs, config.get('bert_batch', 32)), 
                     desc='Eval Docs', leave=False):
            texts_batch = texts[i:i+config.get('bert_batch', 32)]
            feats = model.encoder.encode_batch(texts_batch, device=device, 
                                              max_len=config.get('max_len', 64))
            membank_eval[i:i+config.get('bert_batch', 32)] = feats.cpu().numpy()
        
        X_docs = torch.tensor(membank_eval, dtype=torch.float32, device=device)
        
        # Recompute word features
        X_words_eval = model.encoder.encode_words(vocab, device=device, 
                                                  max_len=config.get('max_len', 64), 
                                                  batch_size=config.get('bert_batch', 32))
        
        X_full = torch.cat([X_docs, X_words_eval], dim=0)
        
        # GCN predictions
        gcn_logits_all = model.gcn_forward(A_torch, X_full)
        doc_logits = gcn_logits_all[:ndocs, :]
        
        # BERT predictions
        bert_logits = []
        for i in range(0, ndocs, config.get('bert_batch', 32)):
            bl = model.aux_clf(model.encoder.encode_batch(
                texts[i:i+config.get('bert_batch', 32)], 
                device=device, 
                max_len=config.get('max_len', 64)
            ))
            bert_logits.append(bl.cpu())
        bert_logits = torch.cat(bert_logits, dim=0).to(device)
        
        # Final predictions
        final_logits = config.get('lmbda', 0.7) * doc_logits + \
                      (1.0 - config.get('lmbda', 0.7)) * bert_logits
        
        # Calculate validation loss
        loss_fn = nn.CrossEntropyLoss()
        val_loss = 0.0
        
        if len(labeled_idx) > 0:
            y_true = torch.tensor([labels[i] for i in labeled_idx], dtype=torch.long, device=device)
            y_pred_logits = final_logits[labeled_idx, :]
            val_loss = loss_fn(y_pred_logits, y_true).item()
            
            # Also print metrics (optional)
            y_pred = torch.argmax(y_pred_logits, dim=1).cpu().numpy()
            acc = (y_pred == y_true.cpu().numpy()).mean()
            
            print(f"\n{'='*70}")
            print(f"Epoch {epoch} Evaluation Results")
            print(f"{'='*70}")
            print(f"Validation Loss: {val_loss:.4f}")
            print(f"Accuracy: {acc*100:.2f}%")
            print(f"{'='*70}")
    
    return val_loss

def evaluate_model(model, texts, labels, labeled_idx, A_torch, vocab, 
                   n_classes, device, config, epoch):
    """Evaluate model on labeled data"""
    model.eval()
    ndocs = len(texts)
    nwords = A_torch.shape[0] - ndocs # Need nwords here for X_words_eval
    
    with torch.no_grad():
        # Recompute memory bank for documents
        membank_eval = np.zeros((ndocs, config.get('feat_dim', 768)), dtype=np.float32)
        
        for i in tqdm(range(0, ndocs, config.get('bert_batch', 32)), 
                     desc='Eval Docs', leave=False):
            texts_batch = texts[i:i+config.get('bert_batch', 32)]
            feats = model.encoder.encode_batch(texts_batch, device=device, 
                                              max_len=config.get('max_len', 64))
            membank_eval[i:i+config.get('bert_batch', 32)] = feats.cpu().numpy()
        
        X_docs = torch.tensor(membank_eval, dtype=torch.float32, device=device)
        
        # Recompute word features
        X_words_eval = model.encoder.encode_words(vocab, device=device, 
                                                  max_len=config.get('max_len', 64), 
                                                  batch_size=config.get('bert_batch', 32))
        
        X_full = torch.cat([X_docs, X_words_eval], dim=0)
        
        # GCN predictions
        gcn_logits_all = model.gcn_forward(A_torch, X_full)
        doc_logits = gcn_logits_all[:ndocs, :]
        
        # BERT predictions
        bert_logits = []
        for i in range(0, ndocs, config.get('bert_batch', 32)):
            bl = model.aux_clf(model.encoder.encode_batch(
                texts[i:i+config.get('bert_batch', 32)], 
                device=device, 
                max_len=config.get('max_len', 64)
            ))
            bert_logits.append(bl.cpu())
        bert_logits = torch.cat(bert_logits, dim=0).to(device)
        
        # Final predictions
        final_logits = config.get('lmbda', 0.7) * doc_logits + \
                      (1.0 - config.get('lmbda', 0.7)) * bert_logits
        
        if len(labeled_idx) > 0:
            y_true = np.array([labels[i] for i in labeled_idx])
            y_pred = torch.argmax(final_logits[labeled_idx, :], dim=1).cpu().numpy()
            acc = (y_pred == y_true).mean()
            
            print(f"\n{'='*70}")
            print(f"Epoch {epoch} Evaluation Results")
            print(f"{'='*70}")
            print(f"Accuracy: {acc*100:.2f}%\n")
            
            print("Classification Report:")
            print(classification_report(y_true, y_pred, 
                                       target_names=[f'Class {i}' for i in range(n_classes)],
                                       zero_division=0))
            
            cm = confusion_matrix(y_true, y_pred)
            print("\nConfusion Matrix:")
            print(cm)
            
            # Plot confusion matrix
            if config.get('plot_cm', False):
                plot_confusion_matrix(cm, n_classes, config, epoch)


def plot_confusion_matrix(cm, n_classes, config, epoch):
    """Plot and save confusion matrix"""
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
               xticklabels=[f'Class {i}' for i in range(n_classes)],
               yticklabels=[f'Class {i}' for i in range(n_classes)])
    plt.title(f'Confusion Matrix - Epoch {epoch}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    
    save_dir = config.get('save_dir', 'checkpoints')
    os.makedirs(save_dir, exist_ok=True)
    
    save_path = os.path.join(save_dir, f'confusion_matrix_epoch{epoch}.png')
    plt.savefig(save_path, dpi=150)
    print(f"Confusion matrix saved: {save_path}")
    plt.close()


def save_checkpoint(model, label_map, vocab, config, epoch):
    """Save model checkpoint"""
    if config.get('mode') == 'quickrun':
        return  # Don't save in quickrun mode
    
    save_dir = config.get('save_dir', 'checkpoints')
    os.makedirs(save_dir, exist_ok=True)
    
    checkpoint = {
        'model_state': model.state_dict(),
        'label_map': label_map,
        'vocab': vocab,
        'config': config,
        'epoch': epoch
    }
    
    save_path = os.path.join(save_dir, f'checkpoint_epoch{epoch}.pt')
    torch.save(checkpoint, save_path)
    print(f"Checkpoint saved: {save_path}")
