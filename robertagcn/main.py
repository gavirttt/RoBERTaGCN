"""
BertGCN with Social Edges - Main Entry Point

Usage:
    python main.py                    # Use config.py settings
    python main.py --preset quickrun  # Use preset configuration
    python main.py --config custom_config.py  # Use custom config file
"""

import argparse
import importlib.util
import sys
from train import run_training
from train_cv import run_Kfold_cv


def load_config_from_file(config_path):
    """Load configuration from a Python file"""
    spec = importlib.util.spec_from_file_location("config", config_path)
    config_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config_module)
    return config_module


def parse_args():
    parser = argparse.ArgumentParser(
        description='BertGCN: Transductive Text Classification with Social Edges',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                           # Use default config.py
  python main.py --preset quickrun         # Quick test run
  python main.py --preset high_quality     # Best quality settings
  python main.py --config my_config.py     # Custom config file
  
Configuration presets:
  default       : Balanced configuration (recommended)
  quickrun      : Fast testing with minimal data
  low_resource  : For limited GPU memory
  high_quality  : Best quality, requires good GPU
  social_heavy  : Emphasizes social network edges
        """
    )
    
    parser.add_argument('--config', type=str, default='config.py',
                       help='Path to configuration file (default: config.py)')
    
    parser.add_argument('--preset', type=str, default=None,
                       choices=['default', 'quickrun', 'low_resource', 
                               'high_quality', 'social_heavy'],
                       help='Use a preset configuration')
    
    parser.add_argument('--mode', type=str, default=None,
                       choices=['train', 'cv', 'quickrun'],
                       help='Override run mode (train/cv/quickrun)')
    
    parser.add_argument('--labeled-data', type=str, default=None,
                       help='Override labeled data path')
    
    parser.add_argument('--unlabeled-data', type=str, default=None,
                       help='Override unlabeled data path')
    
    parser.add_argument('--social-weight', type=float, default=None,
                       help='Override social edge weight')
    
    parser.add_argument('--epochs', type=int, default=None,
                       help='Override number of epochs')
    
    parser.add_argument('--save-dir', type=str, default=None,
                       help='Override save directory')
    
    parser.add_argument('--validate', action='store_true',
                       help='Validate configuration and exit')
    
    return parser.parse_args()


def merge_config_with_args(config_dict, args):
    """Merge command-line arguments into configuration"""
    # Override with command-line arguments
    if args.mode:
        config_dict['mode'] = args.mode
    
    if args.labeled_data:
        config_dict['labeled_data'] = args.labeled_data
    
    if args.unlabeled_data:
        config_dict['unlabeled_data'] = args.unlabeled_data
    
    if args.social_weight is not None:
        config_dict['social_weight'] = args.social_weight
    
    if args.epochs is not None:
        config_dict['epochs'] = args.epochs
    
    if args.save_dir:
        config_dict['save_dir'] = args.save_dir
    
    return config_dict


def print_config_summary(config):
    """Print configuration summary"""
    print("\n" + "="*70)
    print("CONFIGURATION SUMMARY")
    print("="*70)
    
    print("\n[Data]")
    print(f"  Labeled:   {config['labeled_data']}")
    print(f"  Unlabeled: {config.get('unlabeled_data', 'None')}")
    
    print("\n[Model]")
    print(f"  Encoder:     {config['encoder']}")
    print(f"  Feature dim: {config['feat_dim']}")
    print(f"  GCN hidden:  {config['gcn_hid']}")
    print(f"  Dropout:     {config['dropout']}")
    
    print("\n[Training]")
    print(f"  Mode:        {config.get('mode', 'train')}")
    print(f"  Epochs:      {config['epochs']}")
    print(f"  Batch size:  {config['batch_size']}")
    print(f"  BERT batch:  {config['bert_batch']}")
    print(f"  Max length:  {config['max_len']}")
    
    print("\n[Optimization]")
    print(f"  LR (GCN):      {config['lr_gcn']}")
    print(f"  LR (BERT):     {config['lr_bert']}")
    print(f"  Weight decay:  {config['weight_decay']}")
    print(f"  Lambda:        {config['lmbda']}")
    
    print("\n[Graph]")
    print(f"  Max vocab:     {config['max_vocab']}")
    print(f"  Min doc freq:  {config['min_df']}")
    print(f"  Window size:   {config['window_size']}")
    print(f"  Social weight: {config['social_weight']}")
    
    print("\n[Output]")
    print(f"  Save dir:    {config['save_dir']}")
    print(f"  Plot CM:     {config.get('plot_cm', False)}")
    
    print("\n" + "="*70 + "\n")


def main():
    args = parse_args()
    
    # Load configuration
    print(f"Loading configuration from: {args.config}")
    
    try:
        config_module = load_config_from_file(args.config)
        
        # Get configuration (use preset if specified)
        if args.preset:
            print(f"Using preset: {args.preset}")
            config = config_module.get_config(args.preset)
        else:
            config = config_module.get_config('default')
        
        # Merge with command-line arguments
        config = merge_config_with_args(config, args)
        
        # Validate configuration
        if hasattr(config_module, 'validate_config'):
            config_module.validate_config(config)
            print("✓ Configuration validated successfully")
        
        # Print configuration summary
        print_config_summary(config)
        
        # If --validate flag, exit here
        if args.validate:
            print("Configuration validation complete. Exiting.")
            return
        
    except FileNotFoundError:
        print(f"Error: Configuration file not found: {args.config}")
        print("Please create a config.py file or specify --config path")
        sys.exit(1)
    
    except ValueError as e:
        print(f"Error: Configuration validation failed")
        print(str(e))
        sys.exit(1)
    
    except Exception as e:
        print(f"Error loading configuration: {e}")
        sys.exit(1)
    
    # Run based on mode
    mode = config.get('mode', 'train')
    
    try:
        if mode == 'cv':
            print("Starting 10-fold cross-validation...")
            run_Kfold_cv(config)
        elif mode in ['train', 'quickrun']:
            if mode == 'quickrun':
                print("Running in QUICKRUN mode (fast testing)")
            else:
                print("Starting training...")
            run_training(config)
        else:
            print(f"Error: Unknown mode '{mode}'")
            sys.exit(1)
    
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user")
        sys.exit(0)
    
    except Exception as e:
        print(f"\nError during training: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()