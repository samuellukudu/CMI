import polars as pl
import numpy as np
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import StratifiedGroupKFold
from typing import Tuple, List
import os
import joblib

from scipy.spatial.transform import Rotation as R

# Paths
TRAIN_PATH = "input/train.csv"
TEST_PATH = "input/test.csv"
PREPROCESSED_DIR = "preprocessed"

# 1. Data Loading
def load_data(train_path: str, test_path: str) -> Tuple[pl.DataFrame, pl.DataFrame]:
    train = pl.read_csv(train_path)
    test = pl.read_csv(test_path)
    return train, test

# 1b. Downcast dtypes for memory efficiency
def downcast_dtypes(df: pl.DataFrame) -> pl.DataFrame:
    for col, dtype in zip(df.columns, df.dtypes):
        if dtype == pl.Float64:
            df = df.with_columns(pl.col(col).cast(pl.Float32))
        elif dtype == pl.Int64:
            df = df.with_columns(pl.col(col).cast(pl.Int32))
    return df

# 2. Identify Feature Columns
def get_feature_columns(df: pl.DataFrame) -> Tuple[List[str], List[str], List[str]]:
    imu_cols = [col for col in df.columns if col.startswith("acc_") or col.startswith("rot_")]
    thm_cols = [col for col in df.columns if col.startswith("thm_")]
    tof_cols = [col for col in df.columns if col.startswith("tof_")]
    return imu_cols, thm_cols, tof_cols

# 3. Generate Observed Masks (before filling missing values)
def generate_observed_masks(df: pl.DataFrame, thm_cols: List[str], tof_cols: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate observed/missing masks before imputation.
    Returns binary masks where 1 = observed, 0 = missing.
    """
    thm_observed = np.ones((df.shape[0], len(thm_cols)), dtype=np.float32)
    tof_observed = np.ones((df.shape[0], len(tof_cols)), dtype=np.float32)
    
    # THM: missing values are null/NaN
    for i, col in enumerate(thm_cols):
        col_data = df[col].to_numpy()
        thm_observed[:, i] = (~np.isnan(col_data)).astype(np.float32)
    
    # TOF: missing values are -1
    for i, col in enumerate(tof_cols):
        col_data = df[col].to_numpy()
        tof_observed[:, i] = (col_data != -1).astype(np.float32)
    
    return thm_observed, tof_observed

# 3. Fill Missing Values
def fill_missing_thm(df: pl.DataFrame, thm_cols: List[str]) -> pl.DataFrame:
    for col in thm_cols:
        median_val = df[col].median()
        df = df.with_columns(pl.col(col).fill_null(median_val))
    return df

def fill_missing_tof(df: pl.DataFrame, tof_cols: List[str]) -> pl.DataFrame:
    for col in tof_cols:
        median_val = df.filter(pl.col(col) != -1)[col].median()
        df = df.with_columns(
            pl.when(pl.col(col) == -1).then(median_val).otherwise(pl.col(col)).alias(col)
        )
    return df

# --- THM Feature Engineering ---
def add_thm_features_polars(df: pl.DataFrame) -> pl.DataFrame:
    """
    Add thermopile (THM) features including:
    - Rate of temperature change for each sensor
    - Spatial gradients between adjacent sensors
    - Basic statistical features
    """
    # Sort by sequence and timestamp if available
    if 'sequence_id' in df.columns:
        df = df.sort(['sequence_id', 'timestamp']) if 'timestamp' in df.columns else df.sort('sequence_id')
    
    thm_cols = [col for col in df.columns if col.startswith("thm_")]
    
    # 1. Rate of Temperature Change
    if 'sequence_id' in df.columns:
        for col in thm_cols:
            df = df.with_columns([
                pl.col(col).diff().over(['sequence_id']).forward_fill().fill_null(0).alias(f"{col}_rate")
            ])
    else:
        for col in thm_cols:
            df = df.with_columns([
                pl.col(col).diff().forward_fill().fill_null(0).alias(f"{col}_rate")
            ])
    
    # 2. Spatial Gradients (assuming sensors are arranged 1-5)
    # Calculate gradients between adjacent sensors
    for i in range(1, 5):  # 4 gradients between 5 sensors
        df = df.with_columns([
            (pl.col(f"thm_{i}") - pl.col(f"thm_{i+1}")).alias(f"thm_grad_{i}_{i+1}")
        ])
    
    # 3. Basic Statistical Features
    # Calculate statistics across sensors for each row
    df = df.with_columns([
        pl.sum_horizontal(*thm_cols).alias("thm_mean"),  # This will be our mean after division
        pl.min_horizontal(*thm_cols).alias("thm_min"),
        pl.max_horizontal(*thm_cols).alias("thm_max")
    ])
    
    # Update mean and calculate range
    df = df.with_columns([
        (pl.col("thm_mean") / len(thm_cols)).alias("thm_mean"),  # Now it's actually the mean
        (pl.col("thm_max") - pl.col("thm_min")).alias("thm_range")
    ])
    
    # Calculate standard deviation manually
    # First calculate squared differences from mean for each column
    squared_diff_exprs = [(pl.col(col) - pl.col("thm_mean"))**2 for col in thm_cols]
    # Then sum them and divide by n-1
    df = df.with_columns([
        (pl.sum_horizontal(*squared_diff_exprs) / (len(thm_cols) - 1)).sqrt().alias("thm_std")
    ])
    
    # Drop intermediate columns
    df = df.drop(["thm_min", "thm_max"])
    
    return df

# --- IMU Feature Engineering ---
def process_imu_features(acc_data: np.ndarray, quat_data: np.ndarray, time_delta=1/200) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Process all IMU features in a single pass.
    Returns: (linear_accel, linear_acc_mag, linear_acc_mag_jerk, angular_vel, angular_dist)
    """
    num_samples = acc_data.shape[0]
    gravity_world = np.array([0, 0, 9.81])
    
    # Initialize output arrays
    linear_accel = acc_data.copy()
    angular_vel = np.zeros((num_samples, 3))
    angular_dist = np.zeros(num_samples)
    
    # Create masks for valid quaternions
    valid_mask = ~(np.all(np.isnan(quat_data), axis=1) | np.all(np.isclose(quat_data, 0), axis=1))
    valid_pairs = valid_mask[:-1] & valid_mask[1:]
    
    if np.any(valid_mask):
        try:
            # Process gravity removal for all valid samples at once
            rotation = R.from_quat(quat_data[valid_mask])
            gravity_sensor_frame = rotation.apply(gravity_world, inverse=True)
            linear_accel[valid_mask] = acc_data[valid_mask] - gravity_sensor_frame
            
            if np.any(valid_pairs):
                # Process angular features for valid pairs
                rot_t = R.from_quat(quat_data[:-1][valid_pairs])
                rot_t_plus_dt = R.from_quat(quat_data[1:][valid_pairs])
                delta_rot = rot_t.inv() * rot_t_plus_dt
                
                # Calculate both angular velocity and distance from the same delta rotation
                rotvec = delta_rot.as_rotvec()
                angular_vel[:-1][valid_pairs] = rotvec / time_delta
                angular_dist[:-1][valid_pairs] = np.linalg.norm(rotvec, axis=1)
        except Exception:
            pass
    
    # Calculate magnitude-based features
    linear_acc_mag = np.sqrt((linear_accel**2).sum(axis=1))
    linear_acc_mag_jerk = np.diff(linear_acc_mag, prepend=linear_acc_mag[0])
    
    return linear_accel, linear_acc_mag, linear_acc_mag_jerk, angular_vel, angular_dist

def add_imu_features_polars(df: pl.DataFrame) -> pl.DataFrame:
    # Sort by sequence and timestamp if available
    if 'sequence_id' in df.columns:
        df = df.sort(['sequence_id', 'timestamp']) if 'timestamp' in df.columns else df.sort('sequence_id')
    
    # Basic IMU features using Polars expressions
    df = df.with_columns([
        (pl.col('acc_x')**2 + pl.col('acc_y')**2 + pl.col('acc_z')**2).sqrt().alias('acc_mag'),
        (2 * pl.col('rot_w').clip(-1, 1).arccos()).alias('rot_angle')
    ])

    # Compute features that need windowing
    if 'sequence_id' in df.columns:
        df = df.with_columns([
            pl.col('acc_mag').diff().over(['sequence_id']).forward_fill().fill_null(0).alias('acc_mag_jerk'),
            pl.col('rot_angle').diff().over(['sequence_id']).forward_fill().fill_null(0).alias('rot_angle_vel')
        ])
    else:
        df = df.with_columns([
            pl.col('acc_mag').diff().forward_fill().fill_null(0).alias('acc_mag_jerk'),
            pl.col('rot_angle').diff().forward_fill().fill_null(0).alias('rot_angle_vel')
        ])
    
    # Process data in chunks for memory efficiency
    chunk_size = 10000
    total_rows = df.shape[0]
    chunks = []
    
    for start_idx in range(0, total_rows, chunk_size):
        end_idx = min(start_idx + chunk_size, total_rows)
        chunk = df.slice(start_idx, end_idx - start_idx)
        
        # Convert necessary columns to numpy arrays once
        acc_data = chunk.select(['acc_x', 'acc_y', 'acc_z']).to_numpy()
        quat_data = chunk.select(['rot_x', 'rot_y', 'rot_z', 'rot_w']).to_numpy()
        
        # Process all IMU features in a single pass
        linear_accel, linear_acc_mag, linear_acc_mag_jerk, angular_vel, angular_dist = \
            process_imu_features(acc_data, quat_data)
        
        # Add new columns to chunk all at once
        chunk = chunk.with_columns([
            pl.Series('linear_acc_x', linear_accel[:,0]),
            pl.Series('linear_acc_y', linear_accel[:,1]),
            pl.Series('linear_acc_z', linear_accel[:,2]),
            pl.Series('linear_acc_mag', linear_acc_mag),
            pl.Series('linear_acc_mag_jerk', linear_acc_mag_jerk),
            pl.Series('angular_vel_x', angular_vel[:,0]),
            pl.Series('angular_vel_y', angular_vel[:,1]),
            pl.Series('angular_vel_z', angular_vel[:,2]),
            pl.Series('angular_distance', angular_dist)
        ])
        chunks.append(chunk)
    
    # Combine all chunks
    df = pl.concat(chunks)
    return df

# 4. Data Transformation (Scaling)
def scale_features(train: pl.DataFrame, test: pl.DataFrame, feature_cols: List[str]) -> Tuple[np.ndarray, np.ndarray, RobustScaler]:
    scaler = RobustScaler()
    train_arr = train.select(feature_cols).to_numpy()
    test_arr = test.select(feature_cols).to_numpy()
    train_scaled = scaler.fit_transform(train_arr)
    test_scaled = scaler.transform(test_arr)

    # Fill any NaNs that may have been created during feature engineering
    train_scaled = np.nan_to_num(train_scaled)
    test_scaled = np.nan_to_num(test_scaled)
    return train_scaled, test_scaled, scaler

# 5. Robust Stratified Group K-Fold for both Binary and Multi-Class
def get_binary_bfrb_target(sequence_type: str) -> int:
    """Convert sequence_type to binary target (BFRB vs non-BFRB)"""
    return 1 if sequence_type == "Target" else 0

def get_stratified_group_kfold_splits(
    df: pl.DataFrame, 
    n_splits: int = 5, 
    group_col: str = "subject", 
    target_col: str = "gesture",
    task: str = "binary"  # 'binary' for BFRB vs non-BFRB, 'multiclass' for BFRB gestures only
) -> Tuple[list, np.ndarray]:
    """
    Get stratified group k-fold splits for either binary or multiclass BFRB classification.
    
    Args:
        df: Input dataframe
        n_splits: Number of folds
        group_col: Column name for grouping (usually subject)
        target_col: Column containing gesture labels
        task: Either 'binary' or 'multiclass'
        
    Returns:
        splits: List of (train_idx, val_idx) tuples
        y: Target values used for stratification
    """
    groups = df[group_col].to_numpy()
    sequence_types = df['sequence_type'].to_numpy()
    gestures = df[target_col].to_numpy()

    if task == "binary":
        # Use sequence_type for binary target
        y = np.array([get_binary_bfrb_target(st) for st in sequence_types])
        sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)
        splits = list(sgkf.split(df, y, groups))
        return splits, y
    else:  # multiclass
        # Only use rows with sequence_type == "Target" (BFRB)
        bfrb_mask = sequence_types == "Target"
        if not any(bfrb_mask):
            return [], np.array([])
        df_bfrb = df.filter(pl.Series(bfrb_mask))
        groups = df_bfrb[group_col].to_numpy()
        y = df_bfrb[target_col].to_numpy()
        sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)
        splits = list(sgkf.split(df_bfrb, y, groups))
        return splits, y

# Save preprocessing outputs
def save_preprocessing_outputs(data: dict, out_dir: str = PREPROCESSED_DIR):
    os.makedirs(out_dir, exist_ok=True)
    # Save DataFrames as Parquet
    data['train'].write_parquet(os.path.join(out_dir, 'train_processed.parquet'))
    data['test'].write_parquet(os.path.join(out_dir, 'test_processed.parquet'))
    # Save numpy arrays
    np.save(os.path.join(out_dir, 'train_imu.npy'), data['train_imu'])
    np.save(os.path.join(out_dir, 'test_imu.npy'), data['test_imu'])
    np.save(os.path.join(out_dir, 'train_thm.npy'), data['train_thm'])
    np.save(os.path.join(out_dir, 'test_thm.npy'), data['test_thm'])
    np.save(os.path.join(out_dir, 'train_tof.npy'), data['train_tof'])
    np.save(os.path.join(out_dir, 'test_tof.npy'), data['test_tof'])
    # Save observed masks
    np.save(os.path.join(out_dir, 'train_thm_observed.npy'), data['train_thm_observed'])
    np.save(os.path.join(out_dir, 'train_tof_observed.npy'), data['train_tof_observed'])
    np.save(os.path.join(out_dir, 'test_thm_observed.npy'), data['test_thm_observed'])
    np.save(os.path.join(out_dir, 'test_tof_observed.npy'), data['test_tof_observed'])
    # Save scalers and cv_splits
    joblib.dump(data['imu_scaler'], os.path.join(out_dir, 'imu_scaler.joblib'))
    joblib.dump(data['thm_scaler'], os.path.join(out_dir, 'thm_scaler.joblib'))
    joblib.dump(data['tof_scaler'], os.path.join(out_dir, 'tof_scaler.joblib'))
    # Save both binary and BFRB-specific CV splits
    joblib.dump({
        'binary_splits': data['binary_cv_splits'],
        'binary_targets': data['binary_targets'],
        'bfrb_splits': data['bfrb_cv_splits'],
        'bfrb_targets': data['bfrb_targets']
    }, os.path.join(out_dir, 'cv_splits.joblib'))
    # Save feature column names
    joblib.dump(data['imu_cols'], os.path.join(out_dir, 'imu_cols.joblib'))
    joblib.dump(data['thm_cols'], os.path.join(out_dir, 'thm_cols.joblib'))
    joblib.dump(data['tof_cols'], os.path.join(out_dir, 'tof_cols.joblib'))

# Main preprocessing function
def preprocess():
    train, test = load_data(TRAIN_PATH, TEST_PATH)

    # Downcast dtypes for memory efficiency
    train = downcast_dtypes(train)
    test = downcast_dtypes(test)

    imu_cols, thm_cols, tof_cols = get_feature_columns(train)

    # Generate observed masks BEFORE filling missing values
    train_thm_observed, train_tof_observed = generate_observed_masks(train, thm_cols, tof_cols)
    test_thm_observed, test_tof_observed = generate_observed_masks(test, thm_cols, tof_cols)

    # Fill missing values
    train = fill_missing_thm(train, thm_cols)
    test = fill_missing_thm(test, thm_cols)
    train = fill_missing_tof(train, tof_cols)
    test = fill_missing_tof(test, tof_cols)

    # --- IMU Feature Engineering ---
    train = add_imu_features_polars(train)
    test = add_imu_features_polars(test)
    # Update imu_cols to include new features
    new_imu_features = [
        'acc_mag', 'acc_mag_jerk', 'rot_angle', 'rot_angle_vel',
        'linear_acc_x', 'linear_acc_y', 'linear_acc_z',
        'linear_acc_mag', 'linear_acc_mag_jerk',
        'angular_vel_x', 'angular_vel_y', 'angular_vel_z',
        'angular_distance'
    ]
    imu_cols = imu_cols + [f for f in new_imu_features if f not in imu_cols]

    # --- THM Feature Engineering ---
    train = add_thm_features_polars(train)
    test = add_thm_features_polars(test)
    # Update thm_cols to include new features
    new_thm_features = [
        # Rate of change features
        *[f"thm_{i}_rate" for i in range(1, 6)],
        # Spatial gradient features
        *[f"thm_grad_{i}_{i+1}" for i in range(1, 5)],
        # Statistical features
        'thm_mean', 'thm_range', 'thm_std'
    ]
    thm_cols = thm_cols + [f for f in new_thm_features if f not in thm_cols]

    # Align THM observed masks to final feature dimensionality (engineered feats are always observed)
    extra_train = np.ones((train.shape[0], len(new_thm_features)), dtype=np.float32)
    extra_test = np.ones((test.shape[0], len(new_thm_features)), dtype=np.float32)
    train_thm_observed = np.concatenate([train_thm_observed, extra_train], axis=1)
    test_thm_observed = np.concatenate([test_thm_observed, extra_test], axis=1)

    # Scale features
    train_imu, test_imu, imu_scaler = scale_features(train, test, imu_cols)
    train_thm, test_thm, thm_scaler = scale_features(train, test, thm_cols)
    train_tof, test_tof, tof_scaler = scale_features(train, test, tof_cols)

    # Robust cross-validation splits for both tasks
    binary_cv_splits, binary_targets = get_stratified_group_kfold_splits(
        train, n_splits=5, group_col="subject", target_col="gesture", task="binary"
    )
    bfrb_cv_splits, bfrb_targets = get_stratified_group_kfold_splits(
        train, n_splits=5, group_col="subject", target_col="gesture", task="multiclass"
    )

    # BFRB-only DataFrame for multi-class gesture modeling
    bfrb_train = train.filter(pl.col("sequence_type") == "Target")

    # --- Final NaN Check ---
    print("\n--- Checking for NaNs before saving ---")
    for name, arr in [('IMU', train_imu), ('THM', train_thm), ('TOF', train_tof)]:
        if np.isnan(arr).any():
            nan_counts = np.isnan(arr).sum(axis=0)
            nan_cols = [i for i, count in enumerate(nan_counts) if count > 0]
            print(f"Found {np.isnan(arr).sum()} NaNs in {name} data.")
            print(f"Columns with NaNs (indices): {nan_cols}")
        else:
            print(f"No NaNs found in {name} data.")

    # Return processed data and splits for further use
    return {
        "train": train,
        "test": test,
        "train_imu": train_imu,
        "test_imu": test_imu,
        "train_thm": train_thm,
        "test_thm": test_thm,
        "train_tof": train_tof,
        "test_tof": test_tof,
        "train_thm_observed": train_thm_observed,
        "train_tof_observed": train_tof_observed,
        "test_thm_observed": test_thm_observed,
        "test_tof_observed": test_tof_observed,
        "binary_cv_splits": binary_cv_splits,
        "binary_targets": binary_targets,
        "bfrb_cv_splits": bfrb_cv_splits,
        "bfrb_targets": bfrb_targets,
        "bfrb_train": bfrb_train,
        "imu_cols": imu_cols,
        "thm_cols": thm_cols,
        "tof_cols": tof_cols,
        "imu_scaler": imu_scaler,
        "thm_scaler": thm_scaler,
        "tof_scaler": tof_scaler,
    }

if __name__ == "__main__":
    data = preprocess()
    save_preprocessing_outputs(data)
    print("Preprocessing complete. Data dictionary keys:", list(data.keys()))