import sys
import time
import csv
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import cdist
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split

# ==================== CONFIGURATION ====================
DATASET_SIZE = "100k"           # e.g., "100k", "500k", "1M"
CSV_FILENAME = "fuzzyc_results.csv"

# Show confusion matrix plots? (True/False)
SHOW_PLOTS = False

# Whether to use grid search (False = single fit with fixed hyperparameters)
USE_GRID_SEARCH = True

# Hyperparameters for single fit (best from earlier experiments)
SINGLE_PARAMS = {
    'm': 2.2,
    'error': 1e-5,
    'max_iter': 100,
    'random_state': 42
}

# Grid search parameters (only used if USE_GRID_SEARCH = True)
GRID_PARAMS = {
    'm': {'start': 1.0, 'stop': 2.0, 'step': 0.1},
    'error': {'start': -8, 'stop': -5, 'num': 3},
    'max_iter': [100],
    'random_state': [42]
}
CV_FOLDS = 3

TEST_SIZE = 0.3   # fraction of data held out for testing
RANDOM_STATE = 42

# Feature groups
GROUPS = {
    'flow_timing': {'cols': ['duration'], 'type': 'numeric'},
    'size': {'cols': ['orig_bytes', 'resp_bytes', 'missed_bytes', 'orig_ip_bytes', 'resp_ip_bytes'], 'type': 'numeric'},
    'packet_count': {'cols': ['orig_pkts', 'resp_pkts'], 'type': 'numeric'},
    'ports': {'cols': ['id.orig_p', 'id.resp_p'], 'type': 'numeric'},
    'protocol': {'cols': ['proto'], 'type': 'categorical'},
    'conn_state': {'cols': ['conn_state'], 'type': 'categorical'},
    'history': {'cols': ['history'], 'type': 'categorical'},
    'service': {'cols': ['service'], 'type': 'categorical'},
    'locality': {'cols': ['local_orig', 'local_resp'], 'type': 'binary'}
}

# Progressive feature sets (nested)
PROGRESSIVE_SETS = [
    ['flow_timing', 'size', 'packet_count'],
    ['flow_timing', 'size', 'packet_count', 'ports'],
    ['flow_timing', 'size', 'packet_count', 'ports', 'protocol', 'conn_state'],
    ['flow_timing', 'size', 'packet_count', 'ports', 'protocol', 'conn_state', 'history'],
    ['flow_timing', 'size', 'packet_count', 'ports', 'protocol', 'conn_state', 'history', 'service'],
    ['flow_timing', 'size', 'packet_count', 'ports', 'protocol', 'conn_state', 'history', 'service', 'locality'],
    list(GROUPS.keys())
]

SET_NAMES = [
    "basic_flow",
    "basic+ports",
    "basic+ports+proto_state",
    "basic+ports+proto_state+history",
    "basic+ports+proto_state+history+service",
    "basic+ports+proto_state+history+service+locality",
    "all_features"
]
# =======================================================


def expand_grid_param(value):
    """Convert a range specification into a list of values."""
    if isinstance(value, list):
        return value
    elif isinstance(value, dict):
        if 'step' in value:
            start = value['start']
            stop = value['stop']
            step = value['step']
            return np.arange(start, stop + step/2, step).tolist()
        elif 'num' in value:
            start = value['start']
            stop = value['stop']
            num = value['num']
            return np.logspace(start, stop, num).tolist()
        else:
            raise ValueError("Range dict must contain either 'step' or 'num'")
    else:
        return [value]


# --- FCM implementation ---
class FCM:
    def __init__(self, n_clusters=2, max_iter=150, m=2, error=1e-5, random_state=42):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.m = m
        self.error = error
        self.random_state = random_state
        self.u = None
        self.centers = None

    def fit(self, X):
        N, C = X.shape[0], self.n_clusters
        r = np.random.RandomState(self.random_state)
        u = r.rand(N, C)
        u = u / np.tile(u.sum(axis=1)[np.newaxis].T, C)

        iteration = 0
        while iteration < self.max_iter:
            u2 = u.copy()
            centers = self._next_centers(X, u)
            u = self._next_u(X, centers)
            iteration += 1
            if np.linalg.norm(u - u2) < self.error:
                break

        self.u = u
        self.centers = centers
        return self

    def _next_centers(self, X, u):
        um = u ** self.m
        return (X.T @ um / np.sum(um, axis=0)).T

    def _next_u(self, X, centers):
        power = float(2 / (self.m - 1))
        temp = cdist(X, centers) ** power
        denominator_ = temp.reshape((X.shape[0], 1, -1)).repeat(temp.shape[-1], axis=1)
        denominator_ = temp[:, :, np.newaxis] / denominator_
        return 1 / denominator_.sum(2)

    def predict(self, X):
        if len(X.shape) == 1:
            X = np.expand_dims(X, axis=0)
        power = float(2 / (self.m - 1))
        temp = cdist(X, self.centers) ** power
        denominator_ = temp.reshape((X.shape[0], 1, -1)).repeat(temp.shape[-1], axis=1)
        denominator_ = temp[:, :, np.newaxis] / denominator_
        u = 1 / denominator_.sum(2)
        return np.argmax(u, axis=-1)


class FuzzyCMeansClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, n_clusters=2, max_iter=150, m=2, error=1e-5, random_state=42):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.m = m
        self.error = error
        self.random_state = random_state
        self.fcm_ = None
        self.cluster_to_label_ = None

    def fit(self, X, y):
        self.fcm_ = FCM(
            n_clusters=self.n_clusters,
            max_iter=self.max_iter,
            m=self.m,
            error=self.error,
            random_state=self.random_state
        )
        self.fcm_.fit(X)
        clusters = self.fcm_.predict(X)
        self.cluster_to_label_ = {}
        for c in np.unique(clusters):
            mask = clusters == c
            self.cluster_to_label_[c] = np.bincount(y[mask]).argmax()
        return self

    def predict(self, X):
        clusters = self.fcm_.predict(X)
        return np.array([self.cluster_to_label_[c] for c in clusters])


# --- Feature preprocessing (fit on train, transform train+test) ---
def preprocess_numeric(df, cols, scaler=None):
    X = df[cols].copy().replace('-', np.nan).fillna(0).astype(float)
    if scaler is None:
        scaler = StandardScaler()
        return scaler.fit_transform(X), scaler
    return scaler.transform(X), scaler

def preprocess_binary(df, cols):
    return df[cols].copy().replace('-', 0).fillna(0).astype(int).values

def preprocess_categorical(df, cols, encoder=None):
    X = df[cols].copy().fillna('missing').astype(str).replace('-', 'missing')
    if encoder is None:
        encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        return encoder.fit_transform(X), encoder
    return encoder.transform(X), encoder

def build_group_features(df, mode='train', encoders=None, scalers=None):
    """Preprocess all feature groups. Fits encoders/scalers when mode='train'."""
    group_data = {}
    if mode == 'train':
        encoders, scalers = {}, {}
    for name, info in GROUPS.items():
        cols = info['cols']
        typ = info['type']
        if typ == 'numeric':
            if mode == 'train':
                X, scaler = preprocess_numeric(df, cols)
                scalers[name] = scaler
            else:
                X, _ = preprocess_numeric(df, cols, scalers[name])
            group_data[name] = X
        elif typ == 'categorical':
            if mode == 'train':
                X, encoder = preprocess_categorical(df, cols)
                encoders[name] = encoder
            else:
                X, _ = preprocess_categorical(df, cols, encoders[name])
            group_data[name] = X
        elif typ == 'binary':
            group_data[name] = preprocess_binary(df, cols)
    if mode == 'train':
        return group_data, encoders, scalers
    return group_data


def combine_groups(group_data, group_names):
    """Combine multiple group arrays horizontally."""
    parts = [group_data[name] for name in group_names]
    return np.hstack(parts)


def plot_confusion_matrix(cm, labels, title="Confusion Matrix"):
    """Plot confusion matrix using seaborn heatmap."""
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(title)
    plt.tight_layout()
    plt.show()


def main(csv_path):
    # Read CSV
    try:
        df = pd.read_csv(csv_path, encoding='utf-16')
    except (UnicodeDecodeError, UnicodeError):
        df = pd.read_csv(csv_path, encoding='utf-8')

    # Split into train/test before any preprocessing
    df_train, df_test = train_test_split(df, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    df_train = df_train.reset_index(drop=True)
    df_test = df_test.reset_index(drop=True)
    print(f"Split: {len(df_train)} train, {len(df_test)} test")

    # Fit encoders/scalers on train only, then apply to both splits
    group_train, encoders, scalers = build_group_features(df_train, mode='train')
    group_test = build_group_features(df_test, mode='test', encoders=encoders, scalers=scalers)

    y_train = (df_train['label'] == 'Malicious').astype(int).values
    y_test = (df_test['label'] == 'Malicious').astype(int).values

    file_exists = os.path.isfile(CSV_FILENAME)

    with open(CSV_FILENAME, 'a', newline='') as f:
        writer = csv.writer(f)

        if not file_exists:
            header = [
                'dataset_size', 'feature_set', 'accuracy', 'precision',
                'recall', 'f1', 'fpr', 'training_time_seconds',
                'n_features', 'confusion_matrix', 'm', 'error', 'max_iter'
            ]
            writer.writerow(header)
            print("CSV header written.")

        for set_idx, group_names in enumerate(PROGRESSIVE_SETS):
            set_name = SET_NAMES[set_idx]
            print(f"\n=== Feature set: {set_name} ===")
            X_train = combine_groups(group_train, group_names)
            X_test = combine_groups(group_test, group_names)
            n_features = X_train.shape[1]

            if USE_GRID_SEARCH:
                expanded_grid = {}
                for param, value in GRID_PARAMS.items():
                    expanded_grid[param] = expand_grid_param(value)
                print("Expanded parameter grid:")
                for k, v in expanded_grid.items():
                    print(f"  {k}: {v}")
                total_candidates = np.prod([len(v) for v in expanded_grid.values()])
                print(f"Total candidates: {total_candidates}")

                from sklearn.model_selection import GridSearchCV
                base_clf = FuzzyCMeansClassifier(n_clusters=2)
                grid = GridSearchCV(
                    base_clf,
                    expanded_grid,
                    scoring='f1',
                    cv=CV_FOLDS,
                    n_jobs=-1,
                    verbose=1
                )
                start = time.time()
                grid.fit(X_train, y_train)
                fit_time = time.time() - start

                print(f"Grid search completed in {fit_time:.2f} seconds")
                print("Best parameters:", grid.best_params_)
                print("Best CV F1 score: {:.4f}".format(grid.best_score_))

                best_model = grid.best_estimator_
                y_pred = best_model.predict(X_test)
                m_val = grid.best_params_['m']
                error_val = grid.best_params_['error']
                max_iter_val = grid.best_params_['max_iter']
            else:
                clf = FuzzyCMeansClassifier(
                    n_clusters=2,
                    max_iter=SINGLE_PARAMS['max_iter'],
                    m=SINGLE_PARAMS['m'],
                    error=SINGLE_PARAMS['error'],
                    random_state=SINGLE_PARAMS['random_state']
                )
                start = time.time()
                clf.fit(X_train, y_train)
                fit_time = time.time() - start
                y_pred = clf.predict(X_test)
                m_val = SINGLE_PARAMS['m']
                error_val = SINGLE_PARAMS['error']
                max_iter_val = SINGLE_PARAMS['max_iter']

            # Metrics evaluated on held-out test set
            cm = confusion_matrix(y_test, y_pred)
            tn, fp, fn, tp = cm.ravel()
            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred)
            rec = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

            cm_str = f"[[{tn},{fp}],[{fn},{tp}]]"

            print(f"Training time: {fit_time:.2f} seconds")
            print("Confusion matrix:")
            print(cm)
            print(f"Accuracy:  {acc:.4f}")
            print(f"Precision: {prec:.4f}")
            print(f"Recall:    {rec:.4f}")
            print(f"F1-score:  {f1:.4f}")
            print(f"FPR:       {fpr:.4f}")

            if SHOW_PLOTS:
                plot_confusion_matrix(cm, labels=['Benign', 'Malicious'], title=f"Confusion Matrix - {set_name}")

            row = [
                DATASET_SIZE, set_name, acc, prec, rec, f1, fpr, fit_time,
                n_features, cm_str, m_val, error_val, max_iter_val
            ]
            writer.writerow(row)
            print(f"Results written to {CSV_FILENAME}")

    print("\nAll feature sets processed. Results appended to", CSV_FILENAME)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python fuzzyc.py <csv_file>")
        sys.exit(1)
    main(sys.argv[1])