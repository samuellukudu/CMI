import polars as pl
import numpy as np
from sklearn.preprocessing import RobustScaler, StandardScaler, MinMaxScaler, MaxAbsScaler
from typing import Tuple, List, Dict
import os
from pathlib import Path

# Paths
TRAIN_PATH = "input/train.csv"
TEST_PATH = "input/test.csv"
RESULTS_DIR = "scaler_comparison"

def load_data(train_path: str, test_path: str) -> Tuple[pl.DataFrame, pl.DataFrame]:
    """Load the train and test data"""
    train = pl.read_csv(train_path)
    test = pl.read_csv(test_path)
    return train, test

def get_feature_columns(df: pl.DataFrame) -> Tuple[List[str], List[str], List[str]]:
    """Identify feature columns by type"""
    imu_cols = [col for col in df.columns if col.startswith("acc_") or col.startswith("rot_")]
    thm_cols = [col for col in df.columns if col.startswith("thm_")]
    tof_cols = [col for col in df.columns if col.startswith("tof_")]
    return imu_cols, thm_cols, tof_cols

def fill_missing_values(df: pl.DataFrame, feature_cols: List[str]) -> pl.DataFrame:
    """Fill missing values with median, handling both null and -1 values"""
    for col in feature_cols:
        if col.startswith('tof_'):
            # For TOF features, treat both null and -1 as missing
            valid_mask = (pl.col(col) != -1) & (~pl.col(col).is_null())
            median_val = df.filter(valid_mask)[col].median()
            
            # Fill both null and -1 values
            df = df.with_columns(
                pl.when(pl.col(col).is_null() | (pl.col(col) == -1))
                .then(median_val)
                .otherwise(pl.col(col))
                .alias(col)
            )
        else:
            # For other features, handle null values
            median_val = df[col].median()
            df = df.with_columns(pl.col(col).fill_null(median_val))
    return df

def calculate_reconstruction_error(original: np.ndarray, reconstructed: np.ndarray) -> Dict[str, float]:
    """Calculate various error metrics between original and reconstructed data"""
    # Handle potential NaN or infinite values
    valid_mask = ~np.isnan(original) & ~np.isnan(reconstructed) & \
                ~np.isinf(original) & ~np.isinf(reconstructed)
    
    if not np.any(valid_mask):
        return {
            'mse': np.nan,
            'rmse': np.nan,
            'mae': np.nan,
            'mape': np.nan,
            'r2': np.nan
        }
    
    original = original[valid_mask]
    reconstructed = reconstructed[valid_mask]
    
    # Mean Squared Error
    mse = np.mean((original - reconstructed) ** 2)
    
    # Root Mean Squared Error
    rmse = np.sqrt(mse)
    
    # Mean Absolute Error
    mae = np.mean(np.abs(original - reconstructed))
    
    # Mean Absolute Percentage Error (avoiding division by zero)
    # Only calculate MAPE for non-zero values to avoid division by zero
    nonzero_mask = original != 0
    if np.any(nonzero_mask):
        mape = np.mean(np.abs((original[nonzero_mask] - reconstructed[nonzero_mask]) / 
                             original[nonzero_mask])) * 100
    else:
        mape = np.nan
    
    # R-squared (coefficient of determination)
    if np.all(original == original[0]):  # Check if all values are the same
        r2 = 1.0 if np.allclose(original, reconstructed) else 0.0
    else:
        ss_res = np.sum((original - reconstructed) ** 2)
        ss_tot = np.sum((original - np.mean(original)) ** 2)
        r2 = 1 - (ss_res / ss_tot)
    
    return {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'mape': mape,
        'r2': r2
    }

def fill_missing_numpy(data: np.ndarray) -> np.ndarray:
    """Fill missing values in numpy array using column-wise median"""
    if not np.any(np.isnan(data)) and not np.any(np.isinf(data)):
        return data
    
    filled_data = data.copy()
    medians = np.nanmedian(filled_data, axis=0)
    
    # Handle columns where all values might be NaN
    medians = np.where(np.isnan(medians), 0, medians)
    
    nan_mask = np.isnan(filled_data)
    inf_mask = np.isinf(filled_data)
    filled_data[nan_mask | inf_mask] = np.take(medians, np.where(nan_mask | inf_mask)[1])
    
    return filled_data

def evaluate_scaler(
    data: np.ndarray,
    scaler_name: str,
    scaler_instance: object,
    missing_value_test: bool = False
) -> Dict[str, Dict[str, float]]:
    """Evaluate a scaler's performance on the given data"""
    try:
        # Keep original data intact for error calculation
        original_data = data.copy()
        test_data = data.copy()
        
        # If testing missing value handling, introduce artificial missing values
        if missing_value_test:
            # Set random seed for reproducibility
            np.random.seed(42)
            # Randomly mask 20% of the data
            mask = np.random.choice([True, False], size=test_data.shape, p=[0.2, 0.8])
            test_data[mask] = np.nan
            n_missing = np.sum(mask)
            print(f"\n{scaler_name} - Testing with {n_missing} artificial missing values ({(n_missing/test_data.size)*100:.1f}%)")
            
            # Fill missing values and store for comparison
            filled_data = fill_missing_numpy(test_data)
            n_filled = np.sum(mask)
            if n_filled > 0:
                print(f"Info: {scaler_name} - Filled {n_filled} missing values with column-wise median")
            
        # Transform data using clean data
        if missing_value_test:
            scaled_data = scaler_instance.fit_transform(filled_data)
        else:
            scaled_data = scaler_instance.fit_transform(test_data)
            
        # Calculate distribution statistics on clean data
        clean_data = filled_data if missing_value_test else test_data
        orig_stats = {
            'mean': np.mean(clean_data, axis=0),
            'std': np.std(clean_data, axis=0),
            'min': np.min(clean_data, axis=0),
            'max': np.max(clean_data, axis=0),
            'q1': np.percentile(clean_data, 25, axis=0),
            'q3': np.percentile(clean_data, 75, axis=0)
        }
        
        scaled_stats = {
            'mean': np.mean(scaled_data, axis=0),
            'std': np.std(scaled_data, axis=0),
            'min': np.min(scaled_data, axis=0),
            'max': np.max(scaled_data, axis=0),
            'q1': np.percentile(scaled_data, 25, axis=0),
            'q3': np.percentile(scaled_data, 75, axis=0)
        }
        
        # Inverse transform
        reconstructed_data = scaler_instance.inverse_transform(scaled_data)
        
        # Calculate errors against original data (before artificial missing values)
        errors = calculate_reconstruction_error(original_data, reconstructed_data)
        
        # Add distribution statistics to results
        errors.update({
            'orig_mean': np.mean(orig_stats['mean']),
            'orig_std': np.mean(orig_stats['std']),
            'scaled_mean': np.mean(scaled_stats['mean']),
            'scaled_std': np.mean(scaled_stats['std']),
            'orig_iqr': np.mean(orig_stats['q3'] - orig_stats['q1']),
            'scaled_iqr': np.mean(scaled_stats['q3'] - scaled_stats['q1']),
            'orig_range': np.mean(orig_stats['max'] - orig_stats['min']),
            'scaled_range': np.mean(scaled_stats['max'] - scaled_stats['min'])
        })
        
        return {scaler_name: errors}
    except Exception as e:
        print(f"Error in {scaler_name}: {str(e)}")
        return {scaler_name: {
            'mse': np.nan,
            'rmse': np.nan,
            'mae': np.nan,
            'mape': np.nan,
            'r2': np.nan
        }}

def compare_scalers(data: np.ndarray, test_missing_values: bool = False) -> Dict[str, Dict[str, float]]:
    """Compare different scalers on the same dataset"""
    scalers = {
        'RobustScaler': RobustScaler(),
        'StandardScaler': StandardScaler(),
        'MinMaxScaler': MinMaxScaler(),
        'MaxAbsScaler': MaxAbsScaler()
    }
    
    results = {}
    # Keep original data intact and fill any existing missing values
    clean_data = fill_missing_numpy(data.copy())
    
    for name, scaler in scalers.items():
        # Evaluate on clean data
        results.update(evaluate_scaler(clean_data.copy(), name, scaler))
        
        # If requested, test with artificial missing values using a fresh copy
        if test_missing_values:
            missing_results = evaluate_scaler(clean_data.copy(), f"{name}_missing", scaler, missing_value_test=True)
            results.update(missing_results)
    
    return results

def format_results(results: Dict[str, Dict[str, float]], feature_type: str) -> str:
    """Format results into a readable string"""
    output = f"\nResults for {feature_type} features:\n"
    output += "=" * 100 + "\n"
    
    # Reconstruction metrics
    output += "Reconstruction Metrics:\n"
    output += "-" * 80 + "\n"
    output += f"{'Scaler':<15} {'MSE':<12} {'RMSE':<12} {'MAE':<12} {'MAPE':<12} {'R2':<12}\n"
    output += "-" * 80 + "\n"
    
    for scaler_name, metrics in results.items():
        output += (f"{scaler_name:<15} "
                  f"{metrics['mse']:<12.6f} "
                  f"{metrics['rmse']:<12.6f} "
                  f"{metrics['mae']:<12.6f} "
                  f"{metrics['mape']:<12.6f} "
                  f"{metrics['r2']:<12.6f}\n")
    
    # Distribution statistics
    output += "\nDistribution Statistics:\n"
    output += "-" * 100 + "\n"
    output += f"{'Scaler':<15} {'Orig Mean':<12} {'Scaled Mean':<12} {'Orig Std':<12} {'Scaled Std':<12} "
    output += f"{'Orig IQR':<12} {'Scaled IQR':<12} {'Orig Range':<12} {'Scaled Range':<12}\n"
    output += "-" * 100 + "\n"
    
    for scaler_name, metrics in results.items():
        output += (f"{scaler_name:<15} "
                  f"{metrics['orig_mean']:<12.6f} "
                  f"{metrics['scaled_mean']:<12.6f} "
                  f"{metrics['orig_std']:<12.6f} "
                  f"{metrics['scaled_std']:<12.6f} "
                  f"{metrics['orig_iqr']:<12.6f} "
                  f"{metrics['scaled_iqr']:<12.6f} "
                  f"{metrics['orig_range']:<12.6f} "
                  f"{metrics['scaled_range']:<12.6f}\n")
    
    return output

def main():
    # Create results directory
    Path(RESULTS_DIR).mkdir(exist_ok=True)
    
    # Load data
    train, test = load_data(TRAIN_PATH, TEST_PATH)
    
    # Get feature columns
    imu_cols, thm_cols, tof_cols = get_feature_columns(train)
    
    # Fill missing values
    all_features = imu_cols + thm_cols + tof_cols
    train = fill_missing_values(train, all_features)
    
    # Add data validation info
    def print_data_stats(data: np.ndarray, feature_type: str):
        print(f"\nData statistics for {feature_type}:")
        print(f"Shape: {data.shape}")
        print(f"NaN values: {np.isnan(data).sum()}")
        print(f"Inf values: {np.isinf(data).sum()}")
        print(f"Min value: {np.nanmin(data)}")
        print(f"Max value: {np.nanmax(data)}")
        print(f"Mean value: {np.nanmean(data)}")
        print("-" * 40)
    
    # Prepare results file
    results_path = os.path.join(RESULTS_DIR, 'scaler_comparison_results.txt')
    
    with open(results_path, 'w') as f:
        # Compare scalers for each feature type
        for feature_type, cols in [
            ('IMU', imu_cols),
            ('Thermal', thm_cols),
            ('ToF', tof_cols)
        ]:
            # Get feature data
            data = train.select(cols).to_numpy()
            
            # Print data statistics before scaling
            print_data_stats(data, feature_type)
            
            # For Thermal and ToF, also test with missing values
            test_missing = feature_type in ['Thermal', 'ToF']
            
            # Compare scalers
            results = compare_scalers(data, test_missing_values=test_missing)
            
            # Format and save results
            formatted_results = format_results(results, feature_type)
            f.write(formatted_results)
            print(formatted_results)  # Also print to console
            
            if test_missing:
                f.write("\nMissing Value Analysis:\n")
                f.write("-" * 80 + "\n")
                f.write(f"Original shape: {data.shape}\n")
                f.write(f"Original non-null values: {np.sum(~np.isnan(data))}\n")
                f.write(f"Original null values: {np.sum(np.isnan(data))}\n")
                print("\nMissing Value Analysis:")
                print("-" * 80)
                print(f"Original shape: {data.shape}")
                print(f"Original non-null values: {np.sum(~np.isnan(data))}")
                print(f"Original null values: {np.sum(np.isnan(data))}")

if __name__ == "__main__":
    main()
    print(f"\nResults have been saved to {RESULTS_DIR}/scaler_comparison_results.txt")
