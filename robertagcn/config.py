"""
Configuration file for BertGCN with Social Edges
Edit these parameters instead of using command-line arguments
"""

# ============================================================================
# DATA FILES
# ============================================================================
LABELED_DATA = 'data/labeled_tweets.clean.csv'      # CSV with annotated tweets
UNLABELED_DATA = 'data/unlabeled_tweets.clean.csv'  # CSV with unannotated tweets (optional)

# Alternative: Use combined file (set UNLABELED_DATA = None)
# LABELED_DATA = 'data/combined_tweets.clean.csv'
# UNLABELED_DATA = None

# ============================================================================
# MODEL CONFIGURATION
# ============================================================================
ENCODER_MODEL = 'jcblaise/roberta-tagalog-base'  # Hugging Face model name
FEAT_DIM = 768          # Feature dimension (must match encoder output)
GCN_HIDDEN = 256        # GCN hidden layer dimension
DROPOUT = 0.5           # Dropout rate (paper: 0.5)

# ============================================================================
# TRAINING PARAMETERS
# ============================================================================
EPOCHS = 30
BATCH_SIZE = 48         # Mini-batch size for GCN training
BERT_BATCH_SIZE = 48    # Batch size for BERT encoding
MAX_LENGTH = 128        # Max sequence length for tokenization
SEED = 42               # Random seed for reproducibility

# ============================================================================
# OPTIMIZATION
# ============================================================================
LR_GCN = 1e-3           # Learning rate for GCN (paper: 1e-3)
LR_BERT = 5e-6          # Learning rate for BERT (paper: 1e-5, use smaller for stability)
WEIGHT_DECAY = 1e-4     # L2 regularization
LAMBDA = 0.6            # Interpolation weight λ: Z = λ*Z_GCN + (1-λ)*Z_BERT (paper: 0.7)

# ============================================================================
# GRAPH CONSTRUCTION
# ============================================================================
MAX_VOCAB = 15000       # Maximum vocabulary size
MIN_DOC_FREQ = 2        # Minimum document frequency for words
WINDOW_SIZE = 20        # Sliding window for PMI calculation (paper: 20)
SOCIAL_WEIGHT = 1.0     # Weight for social interaction edges (0 = disable)

# ============================================================================
# EARLY STOPPING PARAMETERS
# ============================================================================
EARLY_STOPPING_PATIENCE = 5           # Number of epochs to wait for improvement
EARLY_STOPPING_DELTA = 0              # Minimum change to qualify as improvement
SAVE_BEST_ONLY = True                 # Save only the best model

# ============================================================================
# OUTPUT
# ============================================================================
SAVE_DIR = 'checkpoints'
PLOT_CONFUSION_MATRIX = True  # Save confusion matrix plots

# ============================================================================
# EXPERIMENT MODE
# ============================================================================
RUN_MODE = 'train'      # Options: 'train', 'cv', 'quickrun'
FOLD = 5                 # Number of folds for cross-validation
# 'train' = normal training
# 'cv' = 10-fold cross-validation
# 'quickrun' = fast test with minimal data

# ============================================================================
# CSV COLUMN MAPPING
# ============================================================================
# Map your CSV columns to expected fields
COLUMN_MAPPING = {
    'id': 'pseudo_id',                          # Document ID column
    'text': 'text',                             # Text content column
    'label': 'predicted_label',                 # Label column (for labeled data)
    'author': 'pseudo_author_userName',         # Author username column
    'reply_to': 'pseudo_inReplyToUsername',     # Reply-to username column
}

# ============================================================================
# ADVANCED SETTINGS
# ============================================================================
# Pre-finetuning
PREFINETUNE_BERT = True         # Pre-finetune BERT before joint training (recommended)
PREFINETUNE_EPOCHS = 2          # Number of pre-finetuning epochs
PREFINETUNE_LR = 1e-6           # Learning rate for pre-finetuning

# Evaluation
EVAL_EVERY_N_EPOCHS = 5         # Evaluate every N epochs during CV
SAVE_BEST_ONLY = False          # Save only best model (by validation F1)

# Memory optimization
LOW_MEMORY_MODE = False         # Use slower but more memory-efficient operations

# ============================================================================
# PRESET CONFIGURATIONS
# ============================================================================

def get_config(preset='default'):
    """
    Get a preset configuration
    
    Available presets:
    - 'default': Balanced configuration for most datasets
    - 'quickrun': Fast testing configuration
    - 'low_resource': For limited GPU memory (e.g., MX450)
    - 'high_quality': Best quality, requires good GPU
    - 'social_heavy': Emphasizes social edges
    """
    
    config = {
        'labeled_data': LABELED_DATA,
        'unlabeled_data': UNLABELED_DATA,
        'encoder': ENCODER_MODEL,
        'feat_dim': FEAT_DIM,
        'gcn_hid': GCN_HIDDEN,
        'dropout': DROPOUT,
        'epochs': EPOCHS,
        'batch_size': BATCH_SIZE,
        'bert_batch': BERT_BATCH_SIZE,
        'max_len': MAX_LENGTH,
        'seed': SEED,
        'lr_gcn': LR_GCN,
        'lr_bert': LR_BERT,
        'weight_decay': WEIGHT_DECAY,
        'lmbda': LAMBDA,
        'max_vocab': MAX_VOCAB,
        'min_df': MIN_DOC_FREQ,
        'window_size': WINDOW_SIZE,
        'social_weight': SOCIAL_WEIGHT,
        'save_dir': SAVE_DIR,
        'plot_cm': PLOT_CONFUSION_MATRIX,
        'mode': RUN_MODE,
        'column_mapping': COLUMN_MAPPING,
        'prefinetune': PREFINETUNE_BERT,
        'prefinetune_epochs': PREFINETUNE_EPOCHS,
        'prefinetune_lr': PREFINETUNE_LR,
        'early_stopping_patience': EARLY_STOPPING_PATIENCE,
        'early_stopping_delta': EARLY_STOPPING_DELTA,
        'save_best_only': SAVE_BEST_ONLY,
        'fold': FOLD,
    }
    
    if preset == 'quickrun':
        config.update({
            'epochs': 1,
            'batch_size': 8,
            'bert_batch': 8,
            'max_vocab': 1000,
            'max_len': 32,
            'prefinetune': False,
            'early_stopping_patience': 2,  # Quick stopping for quickrun
        })
    
    elif preset == 'low_resource':
        config.update({
            'batch_size': 16,
            'bert_batch': 16,
            'max_len': 64,
            'gcn_hid': 128,
            'max_vocab': 10000,
        })
    
    elif preset == 'high_quality':
        config.update({
            'epochs': 50,
            'batch_size': 64,
            'bert_batch': 64,
            'max_len': 256,
            'gcn_hid': 512,
            'max_vocab': 32000,
            'lr_bert': 2e-6,
            'dropout': 0.3,
        })
    
    elif preset == 'social_heavy':
        config.update({
            'social_weight': 2.0,
            'lmbda': 0.8,  # Favor GCN (which includes social edges)
            'epochs': 40,
        })
    
    return config


# ============================================================================
# VALIDATION
# ============================================================================

def validate_config(config):
    """Validate configuration parameters"""
    import os
    
    errors = []
    
    # Check files exist
    if not os.path.exists(config['labeled_data']):
        errors.append(f"Labeled data file not found: {config['labeled_data']}")
    
    if config['unlabeled_data'] and not os.path.exists(config['unlabeled_data']):
        errors.append(f"Unlabeled data file not found: {config['unlabeled_data']}")
    
    # Check parameter ranges
    if config['lmbda'] < 0 or config['lmbda'] > 1:
        errors.append(f"Lambda must be in [0, 1], got {config['lmbda']}")
    
    if config['dropout'] < 0 or config['dropout'] > 1:
        errors.append(f"Dropout must be in [0, 1], got {config['dropout']}")
    
    if config['social_weight'] < 0:
        errors.append(f"Social weight must be >= 0, got {config['social_weight']}")
    
    if config['epochs'] < 1:
        errors.append(f"Epochs must be >= 1, got {config['epochs']}")
    
    if errors:
        raise ValueError("Configuration validation failed:\n" + "\n".join(f"  - {e}" for e in errors))
    
    return True


if __name__ == '__main__':
    # Test configuration loading
    print("Default configuration:")
    config = get_config('default')
    for key, value in config.items():
        if key != 'column_mapping':
            print(f"  {key}: {value}")
    
    print("\nValidating configuration...")
    try:
        validate_config(config)
        print("✓ Configuration is valid")
    except ValueError as e:
        print(f"✗ Configuration error:\n{e}")