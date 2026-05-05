# Training and evaluation script
# Running `python -m src.train` from the project root writes results/metrics.json plus the figures under results/figures/

import json
import os
import time
import warnings
from typing import Dict

import numpy as np
import sklearn.cluster as skcluster
import sklearn.linear_model as sklinear

from src import data as data_mod
from src import evaluate as evalu
from src import model as our
from src import plots

warnings.filterwarnings(action='ignore')
np.random.seed(42)


METRICS_PATH = os.path.join('results', 'metrics.json')
FIG_DIR = os.path.join('results', 'figures')


def _to_native(obj):
    if isinstance(obj, dict):
        return {k: _to_native(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_native(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    return obj


def m1_data() -> Dict:
    print('\n========== Data preparation ==========')
    bundle = data_mod.prepare()
    data_mod.save_splits(
        splits={
            'X_train': bundle['X_train'], 'y_train': bundle['y_train'],
            'X_val':   bundle['X_val'],   'y_val':   bundle['y_val'],
            'X_test':  bundle['X_test'],  'y_test':  bundle['y_test'],
        },
        feature_names=bundle['feature_names'],
        stats=bundle['stats'],
        age_edges=bundle['age_edges'])
    print('Train: {}  Val: {}  Test: {}'.format(
        bundle['X_train'].shape, bundle['X_val'].shape, bundle['X_test'].shape))
    print('Age bin edges (training quantiles): q33={:.2f}  q67={:.2f}'.format(
        bundle['age_edges']['q33'], bundle['age_edges']['q67']))
    return bundle


def baseline_regression(bundle: Dict, metrics: Dict) -> None:
    print('\n========== Scikit-learn linear regression baseline ==========')
    model = sklinear.LinearRegression(fit_intercept=False)

    X_train_b = np.hstack([np.ones((bundle['X_train'].shape[0], 1)), bundle['X_train']])
    X_val_b   = np.hstack([np.ones((bundle['X_val'].shape[0], 1)),   bundle['X_val']])
    X_test_b  = np.hstack([np.ones((bundle['X_test'].shape[0], 1)),  bundle['X_test']])

    model.fit(X_train_b, bundle['y_train'])

    splits = {}
    for name, X, y in [('train', X_train_b, bundle['y_train']),
                       ('val',   X_val_b,   bundle['y_val']),
                       ('test',  X_test_b,  bundle['y_test'])]:
        y_hat = model.predict(X)
        splits[name] = evalu.regression_metrics(y, y_hat)

    evalu.print_regression('scikit-learn linear regression model', 'Longevity', splits)
    metrics['regression']['baseline_sklearn'] = splits


def custom_regression(bundle: Dict, metrics: Dict) -> None:
    print('\n========== Custom linear regression ==========')
    X_train_dN, y_train_1N = data_mod.to_course_shape(bundle['X_train'], bundle['y_train'])
    X_val_dN,   y_val_1N   = data_mod.to_course_shape(bundle['X_val'],   bundle['y_val'])
    X_test_dN,  y_test_1N  = data_mod.to_course_shape(bundle['X_test'],  bundle['y_test'])

    metrics['regression']['custom'] = {}

    for solver in ('normal_equation', 'pseudoinverse'):
        model = our.LinearRegression()
        model.fit(X_train_dN, y_train_1N, solver=solver)
        splits = {}
        for name, X, y in [('train', X_train_dN, y_train_1N),
                           ('val',   X_val_dN,   y_val_1N),
                           ('test',  X_test_dN,  y_test_1N)]:
            splits[name] = {
                'mse': model.score(X, y, scoring_func='mean_squared_error'),
                'r2':  model.score(X, y, scoring_func='r_squared'),
            }
        evalu.print_regression('our linear regression model trained with {}'.format(solver),
                               'Longevity', splits)
        metrics['regression']['custom'][solver] = splits


def optimizer_sweep(bundle: Dict, metrics: Dict) -> None:
    print('\n========== Optimizer comparison for linear regression ==========')
    X_train_dN, y_train_1N = data_mod.to_course_shape(bundle['X_train'], bundle['y_train'])
    X_val_dN,   y_val_1N   = data_mod.to_course_shape(bundle['X_val'],   bundle['y_val'])
    X_test_dN,  y_test_1N  = data_mod.to_course_shape(bundle['X_test'],  bundle['y_test'])

    optimizer_configs = [
        # (label, optimizer_type, alpha, eta_decay_factor, beta, batch_size, T)
        ('GD',          'gradient_descent',                       1e-2, 0.0, 0.0, None, 2000),
        ('momentum GD', 'momentum_gradient_descent',              1e-2, 0.0, 0.9, None, 2000),
        ('SGD',         'stochastic_gradient_descent',            1e-2, 0.5, 0.0,  256, 5000),
        ('momentum SGD','momentum_stochastic_gradient_descent',   1e-2, 0.5, 0.9,  256, 5000),
    ]

    losses_by_optimizer = {}
    metrics['regression']['custom']['gradient_descent'] = {}

    for label, opt_type, alpha, eta_decay, beta, bs, T in optimizer_configs:
        np.random.seed(42)
        model = our.LinearRegression()
        t0 = time.time()
        loss_history = model.fit(
            X_train_dN, y_train_1N,
            solver='gradient_descent',
            optimizer_type=opt_type,
            T=T, alpha=alpha, eta_decay_factor=eta_decay, beta=beta,
            batch_size=bs, n_step_per_log=10, verbose=False)
        elapsed = time.time() - t0

        splits = {}
        for name, X, y in [('train', X_train_dN, y_train_1N),
                           ('val',   X_val_dN,   y_val_1N),
                           ('test',  X_test_dN,  y_test_1N)]:
            splits[name] = {
                'mse': model.score(X, y, scoring_func='mean_squared_error'),
                'r2':  model.score(X, y, scoring_func='r_squared'),
            }
        splits['time_seconds'] = elapsed
        splits['hyperparameters'] = {
            'alpha': alpha, 'eta_decay_factor': eta_decay,
            'beta': beta, 'batch_size': bs, 'T': T,
        }

        evalu.print_regression('our linear regression model with {}'.format(label),
                               'Longevity', splits)
        print('  time elapsed: {:.2f}s'.format(elapsed))
        metrics['regression']['custom']['gradient_descent'][label] = splits
        losses_by_optimizer[label] = loss_history

    plots.loss_curves(
        losses_by_optimizer,
        title='Linear regression: training MSE per logged step',
        out_path=os.path.join(FIG_DIR, 'regression_loss_curves.png'))


def classification(bundle: Dict, metrics: Dict) -> None:
    print('\n========== Classification (short / medium / long lifespan) ==========')
    LABELS = [0, 1, 2]
    DISPLAY = ['Short', 'Medium', 'Long']

    # ---- sklearn baseline ----
    sk_model = sklinear.LogisticRegression(penalty=None, fit_intercept=False, max_iter=2000)
    X_train_b = np.hstack([np.ones((bundle['X_train'].shape[0], 1)), bundle['X_train']])
    X_val_b   = np.hstack([np.ones((bundle['X_val'].shape[0], 1)),   bundle['X_val']])
    X_test_b  = np.hstack([np.ones((bundle['X_test'].shape[0], 1)),  bundle['X_test']])

    sk_model.fit(X_train_b, bundle['y_train_cls'])

    sk_splits = {}
    for name, X, y in [('train', X_train_b, bundle['y_train_cls']),
                       ('val',   X_val_b,   bundle['y_val_cls']),
                       ('test',  X_test_b,  bundle['y_test_cls'])]:
        y_hat = sk_model.predict(X)
        sk_splits[name] = evalu.classification_metrics(y, y_hat, labels=LABELS)

    evalu.print_classification('scikit-learn logistic regression model', 'Longevity', sk_splits)
    metrics['classification']['baseline_sklearn'] = sk_splits

    sk_y_test_hat = sk_model.predict(X_test_b)
    plots.confusion_matrix(
        y_true=bundle['y_test_cls'],
        y_pred=sk_y_test_hat,
        labels=LABELS,
        display_labels=DISPLAY,
        title='sklearn logistic regression (test) — confusion matrix',
        out_path=os.path.join(FIG_DIR, 'classification_confusion_matrix_sklearn.png'))

    X_train_dN, _ = data_mod.to_course_shape(bundle['X_train'], bundle['y_train'])
    X_val_dN,   _ = data_mod.to_course_shape(bundle['X_val'],   bundle['y_val'])
    X_test_dN,  _ = data_mod.to_course_shape(bundle['X_test'],  bundle['y_test'])

    Y_train_oh = data_mod.one_hot(bundle['y_train_cls'], n_classes=3)

    np.random.seed(42)
    model = our.LogisticRegression()
    loss_history = model.fit(
        X_train_dN, Y_train_oh,
        T=5000, alpha=1e-2, eta_decay_factor=0.5, beta=0.9,
        batch_size=256,
        optimizer_type='momentum_stochastic_gradient_descent',
        n_step_per_log=10, verbose=False)

    cu_splits = {}
    for name, X_dN, y_int in [('train', X_train_dN, bundle['y_train_cls']),
                              ('val',   X_val_dN,   bundle['y_val_cls']),
                              ('test',  X_test_dN,  bundle['y_test_cls'])]:
        y_hat = model.predict(X_dN).ravel()
        cu_splits[name] = evalu.classification_metrics(y_int, y_hat, labels=LABELS)

    evalu.print_classification('our logistic regression model', 'Longevity', cu_splits)
    metrics['classification']['custom'] = cu_splits

    y_test_hat = model.predict(X_test_dN).ravel()
    plots.confusion_matrix(
        y_true=bundle['y_test_cls'],
        y_pred=y_test_hat,
        labels=LABELS,
        display_labels=DISPLAY,
        title='Logistic regression (test) — confusion matrix',
        out_path=os.path.join(FIG_DIR, 'classification_confusion_matrix.png'))

    plots.loss_curves(
        {'momentum SGD': loss_history},
        title='Logistic regression: training cross-entropy per logged step',
        out_path=os.path.join(FIG_DIR, 'classification_loss_curve.png'))


def clustering(bundle: Dict, metrics: Dict) -> None:
    print('\n========== K-Means clustering ==========')
    feature_names = bundle['feature_names']
    cont_idx = [i for i, n in enumerate(feature_names) if n in data_mod.CONTINUOUS_COLS]
    X_train_cont = bundle['X_train'][:, cont_idx]

    ks = list(range(2, 8))
    inertias = []
    for k in ks:
        km = skcluster.KMeans(n_clusters=k, n_init=10, random_state=42).fit(X_train_cont)
        inertias.append(float(km.inertia_))

    plots.kmeans_elbow(ks, inertias, os.path.join(FIG_DIR, 'kmeans_elbow.png'))

    chosen_k = 4
    km = skcluster.KMeans(n_clusters=chosen_k, n_init=10, random_state=42).fit(X_train_cont)
    labels = km.labels_

    profiles = []
    cluster_summary = []
    for c in range(chosen_k):
        mask = labels == c
        cont_means = X_train_cont[mask].mean(axis=0)
        mean_age = float(bundle['y_train'][mask].mean())
        size = int(mask.sum())
        profiles.append(cont_means)
        cluster_summary.append({
            'cluster': c,
            'size': size,
            'mean_age_at_death': mean_age,
            'mean_lifestyle_standardized': cont_means.tolist(),
        })
        print('Cluster {}  n={:5d}  mean age at death={:.2f}'.format(c, size, mean_age))

    plots.cluster_lifestyle_profile(
        profile_means=np.array(profiles),
        feature_names=data_mod.CONTINUOUS_COLS,
        out_path=os.path.join(FIG_DIR, 'kmeans_cluster_profiles.png'))

    plots.cluster_mean_age(
        cluster_ids=[c['cluster'] for c in cluster_summary],
        mean_ages=[c['mean_age_at_death'] for c in cluster_summary],
        sizes=[c['size'] for c in cluster_summary],
        out_path=os.path.join(FIG_DIR, 'kmeans_cluster_age.png'))

    metrics['clustering'] = {
        'inertias_by_k': dict(zip(ks, inertias)),
        'chosen_k': chosen_k,
        'clusters': cluster_summary,
    }


def main() -> None:
    os.makedirs(FIG_DIR, exist_ok=True)

    metrics = {
        'regression': {},
        'classification': {},
        'clustering': {},
    }

    bundle = m1_data()
    baseline_regression(bundle, metrics)
    custom_regression(bundle, metrics)
    optimizer_sweep(bundle, metrics)
    classification(bundle, metrics)
    clustering(bundle, metrics)

    os.makedirs(os.path.dirname(METRICS_PATH), exist_ok=True)
    with open(METRICS_PATH, 'w') as f:
        json.dump(_to_native(metrics), f, indent=2)
    print('\nWrote metrics to {}'.format(METRICS_PATH))
    print('Wrote figures to {}'.format(FIG_DIR))


if __name__ == '__main__':
    main()
