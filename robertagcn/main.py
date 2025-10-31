import argparse
from train import run_training
from train_cv import run_10fold_cv


def parse_args():
    p = argparse.ArgumentParser(description='BertGCN: Transductive Text Classification')
    
    # Data
    p.add_argument('--data', required=True, help='CSV file with id,text,label')
    p.add_argument('--encoder', default='jcblaise/roberta-tagalog-base', 
                   help='Hugging Face encoder model')
    
    # Training
    p.add_argument('--epochs', type=int, default=10)
    p.add_argument('--batch_size', type=int, default=64, 
                   help='Mini-batch size for graph training')
    p.add_argument('--bert_batch', type=int, default=64,
                   help='Batch size for BERT encoding')
    p.add_argument('--seed', type=int, default=42)
    
    # Model architecture
    p.add_argument('--max_len', type=int, default=64,
                   help='Max sequence length for BERT')
    p.add_argument('--feat_dim', type=int, default=768,
                   help='Feature dimension (BERT output dimension)')
    p.add_argument('--gcn_hid', type=int, default=256,
                   help='GCN hidden dimension')
    p.add_argument('--dropout', type=float, default=0.5,
                   help='Dropout rate (paper uses 0.5)')
    
    # Optimization
    p.add_argument('--lr_gcn', type=float, default=1e-3,
                   help='Learning rate for GCN (paper: 1e-3)')
    p.add_argument('--lr_bert', type=float, default=1e-5,
                   help='Learning rate for BERT (paper: 1e-5, much smaller)')
    p.add_argument('--weight_decay', type=float, default=1e-5)
    p.add_argument('--lmbda', type=float, default=0.7,
                   help='Interpolation weight λ in Equation 6 (paper: 0.7 for 20NG)')
    
    # Graph construction
    p.add_argument('--max_vocab', type=int, default=20000,
                   help='Maximum vocabulary size')
    p.add_argument('--min_df', type=int, default=2,
                   help='Minimum document frequency for words')
    p.add_argument('--window_size', type=int, default=20,
                   help='Sliding window size for PMI calculation (paper: 20)')
    
    # Output
    p.add_argument('--save_dir', default='checkpoints')
    p.add_argument('--plot_cm', action='store_true', 
                   help='Plot and save confusion matrix')
    
    # Testing
    p.add_argument('--quickrun', action='store_true', 
                   help='Run quick test with small data and 1 epoch')
    p.add_argument('--cv', action='store_true',
                   help='Run 10-fold cross-validation (as in paper)')
    
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    
    # Override settings for quickrun
    if args.quickrun:
        print("QUICKRUN MODE: Adjusting parameters for fast testing...")
        args.epochs = 1
        args.batch_size = 8
        args.bert_batch = 8
        args.max_vocab = 1000
        args.max_len = 32
        print(f"   Epochs: {args.epochs}, Batch size: {args.batch_size}, Vocab: {args.max_vocab}")
    
    if args.cv:
        run_10fold_cv(args)
    else:
        run_training(args)
