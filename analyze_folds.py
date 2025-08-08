import polars as pl
import joblib
import numpy as np
import os

PREPROCESSED_DIR = "preprocessed"

# Load processed data and splits
def load_processed():
    """Load the processed training data and CV splits"""
    train = pl.read_parquet(os.path.join(PREPROCESSED_DIR, "train_processed.parquet"))
    cv_splits_dict = joblib.load(os.path.join(PREPROCESSED_DIR, "cv_splits.joblib"))
    return train, cv_splits_dict

def get_binary_bfrb_target(gesture: str) -> int:
    """Convert gesture to binary target (BFRB vs non-BFRB)"""
    bfrb_gestures = {
        'hair_pulling', 'skin_picking', 'nail_biting', 'teeth_grinding',
        'lip_biting', 'cheek_biting', 'nose_picking', 'thumb_sucking'
    }
    return 1 if gesture in bfrb_gestures else 0

def analyze_gesture_distribution(df: pl.DataFrame, task: str = "binary"):
    """Analyze gesture distribution for either binary or multiclass task"""
    if task == "binary":
        # Add binary labels
        df = df.with_columns(
            pl.col('gesture').map_elements(get_binary_bfrb_target, return_dtype=pl.Int32).alias('is_bfrb')
        )
        return df.group_by('is_bfrb').agg(
            pl.len().alias('count'),
            pl.col('gesture').n_unique().alias('unique_gestures')
        )
    else:
        # Only show BFRB gestures
        return df.filter(
            pl.col('gesture').map_elements(get_binary_bfrb_target, return_dtype=pl.Int32) == 1
        ).group_by('gesture').agg(pl.len().alias('count'))

def analyze_folds():
    train, cv_splits_dict = load_processed()
    
    print("=== Overall Dataset Statistics ===")
    print(f"Total samples: {train.shape[0]}")
    print(f"Total unique subjects: {train['subject'].n_unique()}")
    print(f"Total unique gestures: {train['gesture'].n_unique()}")
    
    print("\n=== Binary Classification (BFRB vs non-BFRB) ===")
    binary_dist = analyze_gesture_distribution(train, "binary")
    print("\nOverall distribution:")
    print(binary_dist)
    
    print("\n--- Binary Classification Fold Analysis ---")
    binary_splits = cv_splits_dict['binary_splits']
    binary_targets = cv_splits_dict['binary_targets']
    
    for i, (train_idx, val_idx) in enumerate(binary_splits):
        train_fold = train[train_idx]
        val_fold = train[val_idx]
        print(f"\nFold {i+1}")
        print("Train set:")
        print(analyze_gesture_distribution(train_fold, "binary"))
        print("Validation set:")
        print(analyze_gesture_distribution(val_fold, "binary"))
        print(f"Train subjects: {train_fold['subject'].n_unique()}, Val subjects: {val_fold['subject'].n_unique()}")
        overlap = set(train_fold['subject'].to_list()) & set(val_fold['subject'].to_list())
        print(f"Subject overlap between train/val: {len(overlap)}")
    
    # Use only BFRB samples for multi-class analysis
    bfrb_train = train.filter(pl.col("sequence_type") == "Target")

    print("\n=== BFRB Multi-class Classification ===")
    bfrb_dist = analyze_gesture_distribution(bfrb_train, "multiclass")
    print("\nOverall BFRB gesture distribution:")
    print(bfrb_dist)

    print("\n--- BFRB Multi-class Fold Analysis ---")
    bfrb_splits = cv_splits_dict['bfrb_splits']
    bfrb_targets = cv_splits_dict['bfrb_targets']

    if len(bfrb_splits) > 0:
        for i, (train_idx, val_idx) in enumerate(bfrb_splits):
            train_fold = bfrb_train[train_idx]
            val_fold = bfrb_train[val_idx]
            print(f"\nFold {i+1}")
            print("Train set:")
            print(analyze_gesture_distribution(train_fold, "multiclass"))
            print("Validation set:")
            print(analyze_gesture_distribution(val_fold, "multiclass"))
            print(f"Train subjects: {train_fold['subject'].n_unique()}, Val subjects: {val_fold['subject'].n_unique()}")
            overlap = set(train_fold['subject'].to_list()) & set(val_fold['subject'].to_list())
            print(f"Subject overlap between train/val: {len(overlap)}")
    else:
        print("No BFRB-specific folds found")

if __name__ == "__main__":
    analyze_folds()
