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
import pandas as pd

from utils import set_seed
from data.build_graph import build_text_graph
from models.bertgcn import BertGCN

from math import ceil


def sparse_to_torch_sparse_tensor(sparse_mx):
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = sparse_mx.shape
    return torch.sparse_coo_tensor(indices, values, torch.Size(shape))


def load_tweets_from_csv(csv_path, column_mapping, has_labels=True):
    """
    Load tweets from CSV with flexible column mapping
    
    Args:
        csv_path: Path to CSV file
        column_mapping: Dict mapping standard names to actual column names
        has_labels: Whether this CSV contains labels
    
    Returns:
        DataFrame with standardized columns
    """
    df = pd.read_csv(csv_path, keep_default_na=False)
    
    # Standardize column names
    rename_map = {}
    for std_name, actual_name in column_mapping.items():
        if actual_name in df.columns:
            rename_map[actual_name] = std_name
    
    df = df.rename(columns=rename_map)
    
    # Ensure required columns exist
    if 'id' not in df.columns:
        df['id'] = df.index.astype(str)
    
    if 'text' not in df.columns:
        raise ValueError(f"Text column not found in {csv_path}")
    
    # Handle labels
    if has_labels:
        if 'label' not in df.columns:
            raise ValueError(f"Label column not found in {csv_path}")
    else:
        df['label'] = None
    
    # Handle social columns (optional)
    if 'author' not in df.columns:
        df['author'] = None
    if 'reply_to' not in df.columns:
        df['reply_to'] = None
    
    # Clean up data types
    df['id'] = df['id'].astype(str)
    df['text'] = df['text'].astype(str)
    df['author'] = df['author'].astype(str)
    df['reply_to'] = df['reply_to'].astype(str)
    
    # Clean empty strings
    df['author'] = df['author'].replace(['', 'nan', 'None'], None)
    df['reply_to'] = df['reply_to'].replace(['', 'nan', 'None'], None)
    
    return df


def read_separate_csv_data(labeled_path, unlabeled_path, column_mapping, quickrun=False):
    """
    Read labeled and unlabeled data from separate CSV files
    
    Args:
        labeled_path: Path to labeled CSV
        unlabeled_path: Path to unlabeled CSV (can be None)
        column_mapping: Dict for column name mapping
        quickrun: Whether to sample small subset
    
    Returns:
        ids, texts, labels, label_map, social_edges
    """
    print("="*70)
    print("Loading data from separate files...")
    print("="*70)
    
    # Load labeled data
    print(f"Loading labeled data from: {labeled_path}")
    labeled_df = load_tweets_from_csv(labeled_path, column_mapping, has_labels=True)
    print(f"  Loaded {len(labeled_df)} labeled tweets")
    
    # Load unlabeled data (if provided)
    if unlabeled_path:
        print(f"Loading unlabeled data from: {unlabeled_path}")
        unlabeled_df = load_tweets_from_csv(unlabeled_path, column_mapping, has_labels=False)
        print(f"  Loaded {len(unlabeled_df)} unlabeled tweets")
    else:
        print("No unlabeled data provided (supervised-only mode)")
        unlabeled_df = pd.DataFrame()
    
    # Quick run sampling
    if quickrun:
        print("\nQUICKRUN MODE: Sampling data...")
        sample_each = 100
        if len(labeled_df) > sample_each:
            labeled_df = labeled_df.sample(n=sample_each, random_state=42)
        if len(unlabeled_df) > sample_each:
            unlabeled_df = unlabeled_df.sample(n=sample_each, random_state=42)
        print(f"  Sampled {len(labeled_df)} labeled + {len(unlabeled_df)} unlabeled")
    
    # Combine datasets
    if not unlabeled_df.empty:
        combined_df = pd.concat([labeled_df, unlabeled_df], ignore_index=True)
    else:
        combined_df = labeled_df
    
    print(f"\nCombined dataset: {len(combined_df)} total tweets")
    
    # Normalize labels
    def normalize_label(label):
        if pd.isna(label) or label in ['', 'nan', 'None', None]:
            return None
        try:
            label_float = float(label)
            return int(label_float)
        except (ValueError, TypeError):
            return None
    
    combined_df['label'] = combined_df['label'].apply(normalize_label)
    
    # Extract data
    texts = combined_df['text'].tolist()
    ids = combined_df['id'].tolist()
    authors = combined_df['author'].tolist()
    reply_to = combined_df['reply_to'].tolist()
    
    # Build label map and labels list
    label_map = {}
    labels = []
    for lab in combined_df['label']:
        if pd.isna(lab) or lab is None:
            labels.append(None)
        else:
            if lab not in label_map:
                label_map[lab] = len(label_map)
            labels.append(label_map[lab])
    
    # Statistics
    labeled_count = sum(1 for y in labels if y is not None)
    unlabeled_count = len(labels) - labeled_count
    
    print(f"\nDataset statistics:")
    print(f"  Total tweets: {len(labels)}")
    print(f"  Labeled: {labeled_count}")
    print(f"  Unlabeled: {unlabeled_count}")
    print(f"  Classes: {label_map}")
    
    if labeled_count > 0:
        label_counts = {}
        for lab in labels:
            if lab is not None:
                label_counts[lab] = label_counts.get(lab, 0) + 1
        print(f"  Label distribution: {label_counts}")
    
    # Build social edges dictionary
    social_edges = {
        'authors': authors,
        'reply_to': reply_to,
        'doc_ids': ids
    }
    
    # Social statistics
    author_counts = {}
    reply_count = 0
    
    for author in authors:
        if author and author not in ['None', 'nan', '']:
            author_counts[author] = author_counts.get(author, 0) + 1
    
    reply_count = sum(1 for r in reply_to if r and r not in ['None', 'nan', ''])
    
    multi_tweet_authors = sum(1 for count in author_counts.values() if count > 1)
    
    print(f"\nSocial network statistics:")
    print(f"  Unique authors: {len(author_counts)}")
    print(f"  Authors with multiple tweets: {multi_tweet_authors}")
    print(f"  Reply relationships: {reply_count}")
    
    return ids, texts, labels, label_map, social_edges


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
    print("BertGCN Training with Social Edges")
    print("="*70)
    print(f"Device: {device}")
    print(f"Mode: {config.get('mode', 'train')}")
    
    # Load data
    ids, texts, labels, label_map, social_edges = read_separate_csv_data(
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
    
    # Build graph with social edges
    print("\n" + "="*70)
    print("Building heterogeneous graph with social edges...")
    print("="*70)
    
    A_norm, vocab, doc_word = build_text_graph(
        texts,
        max_features=config.get('max_vocab', 20000),
        min_df=config.get('min_df', 2),
        window_size=config.get('window_size', 20),
        social_edges=social_edges,
        social_weight=config.get('social_weight', 1.0)
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
        X_words = torch.zeros((nwords, config.get('feat_dim', 768)), 
                             dtype=torch.float32, device=device)
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
        
        # Evaluation
        evaluate_model(model, texts, labels, labeled_idx, A_torch, 
                      X_words, n_classes, device, config, epoch)
        
        # Save checkpoint
        save_checkpoint(model, label_map, vocab, config, epoch)
    
    print("\n" + "="*70)
    print("Training completed!")
    print("="*70)


def evaluate_model(model, texts, labels, labeled_idx, A_torch, X_words, 
                   n_classes, device, config, epoch):
    """Evaluate model on labeled data"""
    model.eval()
    ndocs = len(texts)
    
    with torch.no_grad():
        # Recompute memory bank
        membank_eval = np.zeros((ndocs, config.get('feat_dim', 768)), dtype=np.float32)
        
        for i in tqdm(range(0, ndocs, config.get('bert_batch', 32)), 
                     desc='Eval', leave=False):
            texts_batch = texts[i:i+config.get('bert_batch', 32)]
            feats = model.encoder.encode_batch(texts_batch, device=device, 
                                              max_len=config.get('max_len', 64))
            membank_eval[i:i+config.get('bert_batch', 32)] = feats.cpu().numpy()
        
        X_docs = torch.tensor(membank_eval, dtype=torch.float32, device=device)
        X_full = torch.cat([X_docs, X_words], dim=0)
        
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