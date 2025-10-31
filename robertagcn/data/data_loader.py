import pandas as pd
import numpy as np
import os

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
    
    # Clean up data types
    df['id'] = df['id'].astype(str)
    df['text'] = df['text'].astype(str)
    
    return df


def read_separate_csv_data(labeled_path, unlabeled_path, column_mapping, quickrun=False):
    """
    Read labeled and unlabeled data from separate CSV files
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
    
    # 1. Convert to integers with robust handling
    def normalize_label(label):
        # For unlabeled data, return None
        if pd.isna(label) or str(label).strip() in ['', 'nan', 'none', 'null', 'na']:
            return None
        
        try:
            # Handle string representations of numbers
            label_str = str(label).strip()
            # Remove any extra characters and convert
            cleaned = label_str.replace(' ', '')
            return int(float(cleaned))
        except (ValueError, TypeError) as e:
            print(f"Warning: Could not convert label '{label}' to integer: {e}")
            return None

    # Apply normalization
    combined_df['label'] = combined_df['label'].apply(normalize_label)
    
    # Explicitly convert any remaining np.nan to None
    combined_df['label'] = combined_df['label'].replace({np.nan: None})

    # Debug: Check label distribution
    print("\nLabel distribution after normalization:")
    print(combined_df['label'].value_counts(dropna=False))

    # 2. Create label mapping ONLY from labeled data (non-None values)
    labeled_mask = combined_df['label'].notna()
    labeled_labels = combined_df.loc[labeled_mask, 'label'].tolist()

    if not labeled_labels:
        raise ValueError("No valid labeled data found! Check your labeled CSV file.")

    unique_labels = sorted(set(labeled_labels))
    label_map = {label: idx for idx, label in enumerate(unique_labels)}

    # 3. Apply the consistent mapping
    labels = []
    for lab in combined_df['label']:
        if lab is not None:
            labels.append(label_map[lab])
        else:
            labels.append(None)

    # Verify the mapping
    print(f"\nLabel mapping verification:")
    for orig_label, mapped_idx in sorted(label_map.items()):
        print(f"  Original: {orig_label} → Mapped index: {mapped_idx}")

    # Statistics
    labeled_count = sum(1 for y in labels if y is not None)
    unlabeled_count = len(labels) - labeled_count

    print(f"\nFinal dataset statistics:")
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
    
    # Extract ids and texts from the combined DataFrame
    ids = combined_df['id'].tolist()
    texts = combined_df['text'].tolist()
    
    return ids, texts, labels, label_map
