import torch
import argparse
import pandas as pd
import numpy as np
from tqdm import tqdm
import os

from train import read_csv_data, sparse_to_torch_sparse_tensor
from data.build_graph import build_text_graph
from models.bertgcn import BertGCN

def load_model_and_predict(checkpoint_path, data_path, device='cuda'):
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    args = checkpoint['args']
    label_map = checkpoint['label_map']
    
    print(f"Loaded checkpoint from epoch {checkpoint_path}")
    
    # Reload data
    ids, texts, labels, label_map_loaded = read_csv_data(data_path)
    ndocs = len(texts)
    
    # Rebuild graph (same as training)
    print("Rebuilding graph...")
    A_norm, vocab, doc_word = build_text_graph(
        texts, 
        max_features=args['max_vocab'], 
        min_df=args['min_df'], 
        window_size=args['window_size']
    )
    nwords = doc_word.shape[1]
    A_torch = sparse_to_torch_sparse_tensor(A_norm).coalesce().to(device)
    
    # Recreate model
    model = BertGCN(
        encoder_name=args['encoder'],
        feat_dim=args['feat_dim'],
        gcn_hid=args['gcn_hid'], 
        n_classes=len(label_map),
        dropout=args['dropout']
    ).to(device)
    
    # Load trained weights
    model.load_state_dict(checkpoint['model_state'])
    model.eval()
    
    # Get predictions for all nodes
    print("Generating predictions for all nodes...")
    with torch.no_grad():
        # Compute memory bank
        membank = np.zeros((ndocs, args['feat_dim']), dtype=np.float32)
        for i in tqdm(range(0, ndocs, args['bert_batch']), desc='Computing embeddings'):
            texts_batch = texts[i:i+args['bert_batch']]
            feats = model.encoder.encode_batch(texts_batch, device=device, max_len=args['max_len'])
            membank[i:i+args['bert_batch']] = feats.cpu().numpy()
        
        X_docs = torch.tensor(membank, dtype=torch.float32, device=device)
        X_words = torch.zeros((nwords, args['feat_dim']), dtype=torch.float32, device=device)
        X_full = torch.cat([X_docs, X_words], dim=0)
        
        # GCN predictions
        gcn_logits_all = model.gcn_forward(A_torch, X_full)
        doc_logits = gcn_logits_all[:ndocs, :]
        
        # BERT predictions
        bert_logits = []
        for i in range(0, ndocs, args['bert_batch']):
            bl = model.aux_clf(model.encoder.encode_batch(
                texts[i:i+args['bert_batch']], device=device, max_len=args['max_len']
            ))
            bert_logits.append(bl.cpu())
        bert_logits = torch.cat(bert_logits, dim=0).to(device)
        
        # Combined predictions
        final_logits = args['lmbda'] * doc_logits + (1.0 - args['lmbda']) * bert_logits
        final_predictions = torch.argmax(final_logits, dim=1).cpu().numpy()
        confidence = torch.softmax(final_logits, dim=1).max(dim=1)[0].cpu().numpy()
    
    # Create results DataFrame
    results_df = pd.DataFrame({
        'id': ids,
        'text': texts,
        'true_label': labels,
        'predicted_label': final_predictions,
        'confidence': confidence,
        'status': ['labeled' if lab is not None else 'unlabeled' for lab in labels]
    })
    
    return results_df

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True, help='Path to checkpoint file')
    parser.add_argument('--data', required=True, help='Path to data CSV')
    parser.add_argument('--output', default='predictions.csv', help='Output CSV path')
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    results_df = load_model_and_predict(args.checkpoint, args.data, device)
    
    # Save results
    results_df.to_csv(args.output, index=False)
    print(f"Predictions saved to {args.output}")
    
    # Print summary
    labeled_mask = results_df['status'] == 'labeled'
    unlabeled_mask = results_df['status'] == 'unlabeled'
    
    print(f"\nSummary:")
    print(f"   Labeled documents: {labeled_mask.sum()}")
    print(f"   Unlabeled documents: {unlabeled_mask.sum()}")
    
    if labeled_mask.any():
        accuracy = (results_df[labeled_mask]['true_label'] == results_df[labeled_mask]['predicted_label']).mean()
        print(f"   Accuracy on labeled: {accuracy*100:.2f}%")
    
    print(f"   Prediction distribution:")
    print(results_df['predicted_label'].value_counts().sort_index())
