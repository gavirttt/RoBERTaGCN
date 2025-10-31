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


def read_csv_data(path, quickrun=False):
    import pandas as pd
    df = pd.read_csv(path, keep_default_na=False)

    if quickrun and len(df) > 200:
        labeled_df = df[df['sentiment1'].notna() & (df['sentiment1'] != '')]
        unlabeled_df = df[df['sentiment1'].isna() | (df['sentiment1'] == '')]
        
        sample_each = 100
        if len(labeled_df) > sample_each:
            labeled_df = labeled_df.sample(n=sample_each, random_state=42)
        if len(unlabeled_df) > sample_each:
            unlabeled_df = unlabeled_df.sample(n=sample_each, random_state=42)
            
        df = pd.concat([labeled_df, unlabeled_df], ignore_index=True)
        print(f"QUICKRUN: Sampled {len(labeled_df)} labeled + {len(unlabeled_df)} unlabeled = {len(df)} total")

    # Extract document IDs
    if 'tweet_ids' in df.columns:
        df['id'] = df['tweet_ids'].astype(str)
    elif 'pseudo_id' in df.columns:
        df['id'] = df['pseudo_id'].astype(str)
    if 'id' not in df.columns:
        df['id'] = df.index.astype(str)

    # Extract labels
    if 'sentiment1' in df.columns:
        df['label'] = df['sentiment1']
    if 'label' not in df.columns:
        df['label'] = ''
    df['label'] = df['label'].replace('', np.nan)
    
    def normalize_label(label):
        if pd.isna(label) or label == '':
            return None
        try:
            label_float = float(label)
            return int(label_float)
        except (ValueError, TypeError):
            return None
    
    df['label'] = df['label'].apply(normalize_label)
    
    # Extract texts, labels, and IDs
    texts = df['text'].astype(str).tolist()
    labels = df['label'].tolist()
    ids = df['id'].astype(str).tolist()
    
    # Extract social information
    authors = df['pseudo_author_userName'].astype(str).tolist() if 'pseudo_author_userName' in df.columns else None
    reply_to = df['pseudo_inReplyToUsername'].astype(str).tolist() if 'pseudo_inReplyToUsername' in df.columns else None
    
    # Build label map
    label_map = {}
    y = []
    for lab in labels:
        if pd.isna(lab):
            y.append(None)
        else:
            if lab not in label_map:
                label_map[lab] = len(label_map)
            y.append(label_map[lab])
    
    labeled_count = sum(1 for y_val in y if y_val is not None)
    unlabeled_count = len(y) - labeled_count
    print(f"Dataset: {len(y)} total, {labeled_count} labeled, {unlabeled_count} unlabeled")
    print(f"Classes: {label_map}")
    
    # Create social edges dictionary
    social_edges = None
    if authors is not None:
        social_edges = {
            'authors': authors,
            'reply_to': reply_to if reply_to is not None else [None] * len(texts),
            'doc_ids': ids
        }
        
        # Count social interactions
        author_counts = {}
        reply_count = 0
        for author in authors:
            if author and str(author).lower() not in ['', 'nan', 'none']:
                author_counts[author] = author_counts.get(author, 0) + 1
        
        if reply_to:
            reply_count = sum(1 for r in reply_to if r and str(r).lower() not in ['', 'nan', 'none'])
        
        multi_tweet_authors = sum(1 for count in author_counts.values() if count > 1)
        print(f"Social info: {len(author_counts)} unique authors, "
              f"{multi_tweet_authors} authors with multiple tweets, "
              f"{reply_count} replies")
    
    return ids, texts, y, label_map, social_edges


def prefinetune_bert(model, texts, labels, label_map, device, args):
    """Pre-finetune BERT on labeled data before joint training (Section 3.3)"""
    print("======================================================================\nPre-finetuning BERT on labeled data...")
    
    labeled_texts = [texts[i] for i, label in enumerate(labels) if label is not None]
    labeled_labels = [label for label in labels if label is not None]
    
    if len(labeled_texts) == 0:
        print("No labeled data for pre-finetuning, skipping...")
        return model
    
    labeled_labels_tensor = torch.tensor(labeled_labels, dtype=torch.long, device=device)
    
    # Paper uses Adam for BERT, very small LR
    bert_optimizer = Adam(model.encoder.model.parameters(), lr=1e-6, weight_decay=args.weight_decay)
    loss_fn = nn.CrossEntropyLoss()
    
    model.train()
    n_batches = ceil(len(labeled_texts) / args.bert_batch)
    
    for epoch in range(2):  # Short pre-finetuning
        total_loss = 0
        indices = list(range(len(labeled_texts)))
        np.random.shuffle(indices)
        
        for i in tqdm(range(0, len(labeled_texts), args.bert_batch), desc=f'Pre-finetune Epoch {epoch+1}'):
            batch_indices = indices[i:i+args.bert_batch]
            batch_texts = [labeled_texts[idx] for idx in batch_indices]
            batch_labels = labeled_labels_tensor[batch_indices]
            
            feats = model.encoder.encode_batch(batch_texts, device=device, max_len=args.max_len)
            logits = model.aux_clf(feats)
            
            loss = loss_fn(logits, batch_labels)
            
            bert_optimizer.zero_grad()
            loss.backward()
            bert_optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / n_batches
        print(f"Pre-finetune Epoch {epoch+1} avg_loss={avg_loss:.4f}")
    
    print("BERT pre-finetuning completed")
    return model


def run_training(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    set_seed(args.seed)

    print(f"Using device: {device}")
    if args.quickrun:
        print("QUICKRUN MODE: Fast testing with minimal data")

    ids, texts, labels, label_map, social_edges = read_csv_data(args.data, quickrun=args.quickrun)
    ndocs = len(texts)
    labeled_idx = [i for i, y in enumerate(labels) if y is not None]
    unlabeled_idx = [i for i, y in enumerate(labels) if y is None]
    if len(labeled_idx) > 0:
        n_classes = max([lab for lab in labels if lab is not None]) + 1
    else:
        n_classes = 2
    print("===================================\nDataset Characteristics")
    print(f"Documents: {ndocs}, labeled: {len(labeled_idx)}, classes: {n_classes}")

    # Build graph with social edges
    print("===================================\nBuilding text graph with social edges...")
    social_weight = getattr(args, 'social_weight', 1.0)
    A_norm, vocab, doc_word = build_text_graph(
        texts, 
        max_features=args.max_vocab, 
        min_df=args.min_df, 
        window_size=args.window_size,
        social_edges=social_edges,
        social_weight=social_weight
    )
    nwords = doc_word.shape[1]
    N_total = ndocs + nwords
    A_torch = sparse_to_torch_sparse_tensor(A_norm).coalesce().to(device)
    print(f"Graph built: {N_total} nodes ({ndocs} docs + {nwords} words)\n===================================")

    # Model
    model = BertGCN(
        encoder_name=args.encoder, 
        feat_dim=args.feat_dim, 
        gcn_hid=args.gcn_hid, 
        n_classes=n_classes, 
        dropout=args.dropout
    ).to(device)

    # PRE-FINETUNE BERT (Section 3.3)
    model = prefinetune_bert(model, texts, labels, label_map, device, args)

    # CRITICAL: Paper uses Adam for BERT, but the key is the LR difference
    gcn_params = list(model.gcn.parameters()) + list(model.aux_clf.parameters())
    bert_params = list(model.encoder.model.parameters())
    
    all_params = [
        {'params': gcn_params, 'lr': args.lr_gcn},
        {'params': bert_params, 'lr': args.lr_bert}
    ]
    optimizer = Adam(all_params, weight_decay=args.weight_decay)
    
    loss_fn = nn.CrossEntropyLoss()

    doc_indices = list(range(ndocs))
    print("======================================================================")
    for epoch in range(1, args.epochs + 1):
        model.train()
        
        # Build memory bank
        print(f"Epoch {epoch}: Computing memory bank with current BERT...")
        membank = np.zeros((ndocs, args.feat_dim), dtype=np.float32)
        with torch.no_grad():
            for i in tqdm(range(0, ndocs, args.bert_batch), desc='Build Memory Bank'):
                texts_batch = texts[i:i+args.bert_batch]
                feats = model.encoder.encode_batch(texts_batch, device=device, max_len=args.max_len)
                membank[i:i+args.bert_batch] = feats.cpu().numpy()
        
        # Build base X for this epoch
        X_docs = torch.tensor(membank, dtype=torch.float32, device=device)
        X_words = torch.zeros((nwords, args.feat_dim), dtype=torch.float32, device=device)
        X_full_base = torch.cat([X_docs, X_words], dim=0)

        # Shuffle and train
        np.random.shuffle(doc_indices)
        losses = []
        
        for i in tqdm(range(0, ndocs, args.batch_size), desc=f'Epoch {epoch} Training'):
            batch_idx = doc_indices[i:i+args.batch_size]
            texts_batch = [texts[j] for j in batch_idx]
            
            feats_batch = model.encoder.encode_batch(texts_batch, device=device, max_len=args.max_len)
            logits_bert = model.aux_clf(feats_batch)

            # Construct X with updated batch rows
            X = X_full_base.clone()
            X[batch_idx, :] = feats_batch
            X = X.detach()
            X[batch_idx, :] = feats_batch  # These have gradients

            # GCN forward pass
            gcn_logits_all = model.gcn_forward(A_torch, X)
            doc_logits_batch = gcn_logits_all[batch_idx, :]

            # Compute loss only on labeled within batch
            batch_labels = [labels[j] for j in batch_idx]
            labeled_locs = [k for k, v in enumerate(batch_labels) if v is not None]
            if len(labeled_locs) == 0:
                continue

            target = torch.tensor([batch_labels[k] for k in labeled_locs], dtype=torch.long, device=device)
            gcn_pred = doc_logits_batch[labeled_locs]
            bert_pred = logits_bert[labeled_locs]
            
            # Paper Equation 6: Z = λZ_GCN + (1-λ)Z_BERT
            logits = args.lmbda * gcn_pred + (1.0 - args.lmbda) * bert_pred
            loss = loss_fn(logits, target)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.append(loss.item())

        avg_loss = float(np.mean(losses)) if losses else 0.0
        print(f"Epoch {epoch} avg_loss={avg_loss:.4f}")

        # Evaluation on labeled nodes
        model.eval()
        with torch.no_grad():
            # Recompute memory bank for evaluation
            membank_eval = np.zeros((ndocs, args.feat_dim), dtype=np.float32)
            for i in tqdm(range(0, ndocs, args.bert_batch), desc=f'Epoch {epoch} Eval Memory'):
                texts_batch = texts[i:i+args.bert_batch]
                feats = model.encoder.encode_batch(texts_batch, device=device, max_len=args.max_len)
                membank_eval[i:i+args.bert_batch] = feats.cpu().numpy()
            
            X_docs = torch.tensor(membank_eval, dtype=torch.float32, device=device)
            X_full = torch.cat([X_docs, X_words], dim=0)
            gcn_logits_all = model.gcn_forward(A_torch, X_full)
            doc_logits = gcn_logits_all[:ndocs, :]

            # Compute BERT logits for all docs
            bert_logits = []
            for i in range(0, ndocs, args.bert_batch):
                bl = model.aux_clf(model.encoder.encode_batch(texts[i:i+args.bert_batch], device=device, max_len=args.max_len))
                bert_logits.append(bl.cpu())
            bert_logits = torch.cat(bert_logits, dim=0).to(device)

            final_logits = args.lmbda * doc_logits + (1.0 - args.lmbda) * bert_logits
            if len(labeled_idx) > 0:
                y_true = np.array([labels[i] for i in labeled_idx])
                y_pred = torch.argmax(final_logits[labeled_idx, :], dim=1).cpu().numpy()
                acc = (y_pred == y_true).mean()
                print(f"Eval accuracy on labeled nodes: {acc*100:.2f}%")
                
                print("\nClassification Report:")
                print(classification_report(y_true, y_pred, target_names=[f'Class {i}' for i in range(n_classes)], zero_division=0))
                
                cm = confusion_matrix(y_true, y_pred)
                print("\nConfusion Matrix:")
                print(cm)
                
                if args.plot_cm:
                    plt.figure(figsize=(8, 6))
                    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                            xticklabels=[f'Pred {i}' for i in range(n_classes)],
                            yticklabels=[f'True {i}' for i in range(n_classes)])
                    plt.title(f'Confusion Matrix - Epoch {epoch}')
                    plt.ylabel('True Label')
                    plt.xlabel('Predicted Label')
                    plt.tight_layout()
                    
                    os.makedirs(args.save_dir, exist_ok=True)
                    plt.savefig(os.path.join(args.save_dir, f'confusion_matrix_epoch{epoch}.png'))
                    print(f"Confusion matrix saved to {args.save_dir}/confusion_matrix_epoch{epoch}.png")
                    plt.close()

        # Save checkpoint
        if args.save_dir and not args.quickrun:
            os.makedirs(args.save_dir, exist_ok=True)
            torch.save({
                'model_state': model.state_dict(), 
                'label_map': label_map,
                'vocab': vocab,
                'args': vars(args)
            }, os.path.join(args.save_dir, f'checkpoint_epoch{epoch}.pt'))

    print('Training finished======================================================================')
    if args.quickrun:
        print('Quickrun completed successfully!')