
# Metric helpers to score sklearn baseline and our models

from typing import Dict

import numpy as np
import sklearn.metrics as skmetrics


# Regression metrics: MSE and R^2
def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:

    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()

    mse = float(np.mean((y_true - y_pred) ** 2))

    y_mean = np.mean(y_true)
    ss_tot = np.sum((y_true - y_mean) ** 2)
    ss_res = np.sum((y_true - y_pred) ** 2)
    r2 = float(1.0 - ss_res / ss_tot)

    return {'mse': mse, 'r2': r2}


def print_regression(name: str, dataset: str, splits: Dict[str, Dict[str, float]]) -> None:

    print('***** Results of {} on {} dataset *****'.format(name, dataset))
    for split_name in ('train', 'val', 'test'):
        if split_name not in splits:
            continue
        metrics = splits[split_name]
        pretty = {'train': 'Training', 'val': 'Validation', 'test': 'Testing'}[split_name]
        print('{} set mean squared error: {:.4f}'.format(pretty, metrics['mse']))
        print('{} set r-squared scores: {:.4f}'.format(pretty, metrics['r2']))



# Classification metrics: accuracy, precision, recall, F1
def classification_metrics(y_true: np.ndarray,
                           y_pred: np.ndarray,
                           labels: list = None) -> Dict[str, float]:

    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()

    if labels is None:
        labels = sorted(np.unique(np.concatenate([y_true, y_pred])).tolist())

    accuracy = float(np.mean(y_true == y_pred))

    precision, recall, f1, _ = skmetrics.precision_recall_fscore_support(
        y_true, y_pred,
        labels=labels,
        average='macro',
        zero_division=0)

    per_class_p, per_class_r, per_class_f1, _ = skmetrics.precision_recall_fscore_support(
        y_true, y_pred,
        labels=labels,
        average=None,
        zero_division=0)

    return {
        'accuracy': accuracy,
        'precision_macro': float(precision),
        'recall_macro': float(recall),
        'f1_macro': float(f1),
        'precision_per_class': [float(v) for v in per_class_p],
        'recall_per_class':    [float(v) for v in per_class_r],
        'f1_per_class':        [float(v) for v in per_class_f1],
        'labels': labels,
    }


def print_classification(name: str, dataset: str, splits: Dict[str, Dict[str, float]]) -> None:
    print('***** Results of {} on {} dataset *****'.format(name, dataset))
    for split_name in ('train', 'val', 'test'):
        if split_name not in splits:
            continue
        metrics = splits[split_name]
        pretty = {'train': 'Training', 'val': 'Validation', 'test': 'Testing'}[split_name]
        print('{} set mean accuracy: {:.4f}'.format(pretty, metrics['accuracy']))
