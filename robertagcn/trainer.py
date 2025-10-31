import os
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
from sklearn.metrics import precision_recall_fscore_support
from math import ceil


def train_epoch(model, texts, labels, train_idx, A_torch, vocab, 
                optimizer, loss_fn, device, config, epoch, prefix=''):
    """Unified training for one epoch"""
    model.train()
    
    # Refresh memory bank at epoch start (Paper Section 3.3)
    if config.get('refresh_memory', True):
        refresh_memory_bank(model, texts, device, config)
    
    # Shuffle training indices
    np.random.shuffle(train_idx)
    
    losses = []
    batch_size = config.get('batch_size', 32)
    num_batches = ceil(len(train_idx) / batch_size)
    
    pbar = tqdm(range(num_batches), desc=f'{prefix} Epoch {epoch}', leave=True)
    
    for batch_num in pbar:
        start_idx = batch_num * batch_size
        end_idx = min((batch_num + 1) * batch_size, len(train_idx))
        batch_idx = train_idx[start_idx:end_idx]
        
        # Forward pass
        log_probs = model(
            idx=batch_idx, A_torch=A_torch, vocab=vocab,
            texts=texts, device=device, config=config, update_memory=True
        )
        
        # Compute loss (all examples in batch are labeled)
        target_labels = torch.tensor([labels[i] for i in batch_idx], 
                                    dtype=torch.long, device=device)
        
        loss = loss_fn(log_probs, target_labels)
        
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        losses.append(loss.item())
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    return {'loss': np.mean(losses) if losses else 0.0}


def evaluate_model(model, texts, labels, eval_idx, A_torch, vocab, 
                   n_classes, device, config):
    """Unified model evaluation"""
    model.eval()
    
    with torch.no_grad():
        log_probs = model(
            idx=eval_idx, A_torch=A_torch, vocab=vocab,
            texts=texts, device=device, config=config, update_memory=False
        )
        
        # Calculate validation loss
        loss_fn = nn.NLLLoss()
        y_true = torch.tensor([labels[i] for i in eval_idx], dtype=torch.long, device=device)
        val_loss = loss_fn(log_probs, y_true).item()
        
        # Get predictions
        y_pred = torch.argmax(torch.exp(log_probs), dim=1).cpu().numpy()
        y_true_np = y_true.cpu().numpy()
        
        # Calculate metrics
        accuracy = (y_pred == y_true_np).mean()
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true_np, y_pred, average='macro', zero_division=0
        )
        
        per_class_precision, per_class_recall, per_class_f1, _ = precision_recall_fscore_support(
            y_true_np, y_pred, average=None, zero_division=0
        )
        
        metrics = {
            'loss': val_loss,
            'accuracy': accuracy,
            'macro_precision': precision,
            'macro_recall': recall,
            'macro_f1': f1,
            'per_class_precision': per_class_precision,
            'per_class_recall': per_class_recall,
            'per_class_f1': per_class_f1,
            'y_true': y_true_np,
            'y_pred': y_pred
        }
        
        return metrics


def refresh_memory_bank(model, texts, device, config):
    """Refresh memory bank with current BERT embeddings"""
    if not model.memory_initialized:
        return
    
    print("Refreshing memory bank...")
    ndocs = len(texts)
    
    with torch.no_grad():
        for i in tqdm(range(0, ndocs, config.get('bert_batch', 32)), 
                     desc='Memory Refresh', leave=False):
            batch_idx = list(range(i, min(i + config.get('bert_batch', 32), ndocs)))
            batch_texts = [texts[j] for j in batch_idx]
            batch_embeddings = model.encoder.encode_batch(
                batch_texts, device=device, max_len=config.get('max_len', 64)
            )
            model.update_memory_batch(batch_idx, batch_embeddings)


def prefinetune_bert(model, texts, labels, label_map, device, config):
    """Pre-finetune BERT on labeled data"""
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
    
    # Only optimize BERT parameters during pre-finetuning
    bert_optimizer = torch.optim.Adam(
        model.encoder.model.parameters(), 
        lr=bert_lr, 
        weight_decay=config.get('weight_decay', 1e-5)
    )
    loss_fn = nn.CrossEntropyLoss()
    
    model.train()
    
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
            
            # Only use BERT forward pass
            feats = model.encoder.encode_batch(
                batch_texts, device=device, max_len=config.get('max_len', 64)
            )
            logits = model.aux_clf(feats)
            
            loss = loss_fn(logits, batch_labels)
            
            bert_optimizer.zero_grad()
            loss.backward()
            bert_optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_loss = total_loss / ceil(len(labeled_texts) / bert_batch)
        print(f"Pre-finetune Epoch {epoch+1}/{epochs} avg_loss={avg_loss:.4f}")
    
    print("BERT pre-finetuning completed\n")
    return model