
# Data loading, cleaning, encoding, and splitting for the Work-Life Balance and Longevity dataset

import json
import os
from typing import Dict, Tuple

import numpy as np
import pandas as pd


# Paths and column names
RAW_CSV = os.path.join('data', 'raw', 'quality_of_life.csv')
PROCESSED_DIR = os.path.join('data', 'processed')

CONTINUOUS_COLS = [
    'avg_work_hours_per_day',
    'avg_rest_hours_per_day',
    'avg_sleep_hours_per_day',
    'avg_exercise_hours_per_day',
]
CATEGORICAL_COLS = ['gender', 'occupation_type']
TARGET_COL = 'age_at_death'


def load_raw(path: str = RAW_CSV) -> pd.DataFrame:
    df = pd.read_csv(path)
    if 'id' in df.columns:
        df = df.drop(columns=['id'])
    return df


def encode(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, list]:
    # Drop one dummy variable to avoid dummy-variable trap
    df_encoded = pd.get_dummies(
        df,
        columns=CATEGORICAL_COLS,
        drop_first=True,
        dtype=float)

    feature_cols = [c for c in df_encoded.columns if c != TARGET_COL]

    X = df_encoded[feature_cols].to_numpy(dtype=np.float32)
    y = df_encoded[TARGET_COL].to_numpy(dtype=np.float32)

    return X, y, feature_cols

# Shuffles and splits into train / val / test
def split(X: np.ndarray,
          y: np.ndarray,
          ratios: Tuple[float, float, float] = (0.80, 0.10, 0.10),
          seed: int = 42) -> Dict[str, np.ndarray]:

    assert abs(sum(ratios) - 1.0) < 1e-6, 'ratios must sum to 1'

    rng = np.random.default_rng(seed)
    N = X.shape[0]
    shuffled_indices = rng.permutation(N)

    train_split_idx = int(ratios[0] * N)
    val_split_idx = int((ratios[0] + ratios[1]) * N)

    train_indices = shuffled_indices[:train_split_idx]
    val_indices = shuffled_indices[train_split_idx:val_split_idx]
    test_indices = shuffled_indices[val_split_idx:]

    return {
        'X_train': X[train_indices], 'y_train': y[train_indices],
        'X_val':   X[val_indices],   'y_val':   y[val_indices],
        'X_test':  X[test_indices],  'y_test':  y[test_indices],
    }


def standardize(X_train: np.ndarray,
                X_val: np.ndarray,
                X_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
   
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)
    std = np.where(std < 1e-8, 1.0, std)

    X_train_std = (X_train - mean) / std
    X_val_std = (X_val - mean) / std
    X_test_std = (X_test - mean) / std

    return X_train_std, X_val_std, X_test_std, {'mean': mean, 'std': std}

# Splits ages into three classes (0=Short, 1=Medium, 2=Long) using the 33rd and 67th percentiles of the training set
def bin_age(y_train: np.ndarray,
            y_val: np.ndarray,
            y_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, float]]:
  
    q33 = float(np.quantile(y_train, 1.0 / 3.0))
    q67 = float(np.quantile(y_train, 2.0 / 3.0))

    def _bin(y):
        out = np.zeros_like(y, dtype=np.int64)
        out[(y >= q33) & (y < q67)] = 1
        out[y >= q67] = 2
        return out

    return _bin(y_train), _bin(y_val), _bin(y_test), {'q33': q33, 'q67': q67}


def to_course_shape(X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    X_T = X.T
    y_row = np.expand_dims(y, axis=0).astype(np.float32)
    return X_T, y_row


def one_hot(y: np.ndarray, n_classes: int) -> np.ndarray:
    N = y.shape[0]
    out = np.zeros((n_classes, N), dtype=np.float32)
    out[y.astype(int), np.arange(N)] = 1.0
    return out


def save_splits(splits: Dict[str, np.ndarray],
                feature_names: list,
                stats: Dict[str, np.ndarray],
                age_edges: Dict[str, float],
                out_dir: str = PROCESSED_DIR) -> None:

    os.makedirs(out_dir, exist_ok=True)
    np.savez(
        os.path.join(out_dir, 'splits.npz'),
        **splits,
        mean=stats['mean'],
        std=stats['std'])
    with open(os.path.join(out_dir, 'meta.json'), 'w') as f:
        json.dump({
            'feature_names': feature_names,
            'age_bin_edges': age_edges,
        }, f, indent=2)

#   Load -> encode -> split -> standardize -> bin ages
def prepare(seed: int = 42,
            ratios: Tuple[float, float, float] = (0.80, 0.10, 0.10)) -> Dict:
    
    df = load_raw()
    X, y, feature_names = encode(df)

    splits = split(X, y, ratios=ratios, seed=seed)
    X_train_std, X_val_std, X_test_std, stats = standardize(
        splits['X_train'], splits['X_val'], splits['X_test'])

    y_train_cls, y_val_cls, y_test_cls, age_edges = bin_age(
        splits['y_train'], splits['y_val'], splits['y_test'])

    return {
        'X_train': X_train_std, 'y_train': splits['y_train'], 'y_train_cls': y_train_cls,
        'X_val':   X_val_std,   'y_val':   splits['y_val'],   'y_val_cls':   y_val_cls,
        'X_test':  X_test_std,  'y_test':  splits['y_test'],  'y_test_cls':  y_test_cls,
        'feature_names': feature_names,
        'stats': stats,
        'age_edges': age_edges,
    }
