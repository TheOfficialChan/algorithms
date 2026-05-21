import sys
import time
import csv
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import IsolationForest
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import GridSearchCV, train_test_split
from joblib import parallel_backend

# ==================== CONFIGURATION ====================
DATASET_SIZE = "100k"            # e.g., "100k", "500k", "1M"
CSV_FILENAME = "isolationforest_results.csv"

# Show confusion matrix plots? (True/False)
SHOW_PLOTS = False

# Whether to use grid search (False = single fit with fixed hyperparameters)
USE_GRID_SEARCH = False

# Hyperparameters for single fit (best from earlier experiments)
SINGLE_PARAMS = {
    'contamination': 'auto',
    'n_estimators': 100,
    'max_samples': 'auto',
    'random_state': 42
}

# Grid search parameters (only used if USE_GRID_SEARCH = True)
# Fixed: contamination must be a list so it is iterated over during grid search
GRID_PARAMS = {
    'contamination': [0.15,0.20,0.30, 0.40, 0.50],
    'n_estimators': [50, 100, 150, 200],
    'max_samples': ['auto', 0.5, 0.8],
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


# --- Isolation Forest wrapper ---
class IsolationForestClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, contamination='auto', n_estimators=100, max_samples='auto', random_state=None):
        self.contamination = contamination
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.random_state = random_state
        self.iforest_ = None
        self.cluster_to_label_ = None
        self.classes_ = None

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        self.iforest_ = IsolationForest(
            contamination=self.contamination,
            n_estimators=self.n_estimators,
            max_samples=self.max_samples,
            random_state=self.random_state
        )
        self.iforest_.fit(X)   # unsupervised, y not used for training
        # Map -1/1 cluster outputs to majority label from training data
        cluster_assignments = self.iforest_.predict(X)
        self.cluster_to_label_ = {}
        for c in np.unique(cluster_assignments):
            mask = cluster_assignments == c
            self.cluster_to_label_[c] = np.bincount(y[mask]).argmax()
        return self

    def predict(self, X):
        cluster_assignments = self.iforest_.predict(X)
        return np.array([self.cluster_to_label_[c] for c in cluster_assignments])


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
    plt.close()


def read_csv_auto_encoding(filepath):
    """Try common encodings and fallback."""
    encodings = ['utf-16', 'utf-8']
    for enc in encodings:
        try:
            return pd.read_csv(filepath, encoding=enc)
        except (UnicodeDecodeError, UnicodeError):
            continue
    return pd.read_csv(filepath, encoding='utf-8', errors='replace')


def process_single_feature_set(set_idx, group_names, writer, group_train, group_test, y_train, y_test):
    set_name = SET_NAMES[set_idx]
    print(f"\n=== Feature set: {set_name} ===")
    X_train = combine_groups(group_train, group_names)
    X_test = combine_groups(group_test, group_names)

    if USE_GRID_SEARCH:
        expanded_grid = {}
        for param, value in GRID_PARAMS.items():
            expanded_grid[param] = expand_grid_param(value)
        print("Expanded parameter grid:")
        for k, v in expanded_grid.items():
            print(f"  {k}: {v}")
        total_candidates = np.prod([len(v) for v in expanded_grid.values()])
        print(f"Total candidates: {total_candidates}")

        base_clf = IsolationForestClassifier()
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
        contamination_val = grid.best_params_['contamination']
        n_estimators_val = grid.best_params_['n_estimators']
        max_samples_val = grid.best_params_['max_samples']
    else:
        clf = IsolationForestClassifier(
            contamination=SINGLE_PARAMS['contamination'],
            n_estimators=SINGLE_PARAMS['n_estimators'],
            max_samples=SINGLE_PARAMS['max_samples'],
            random_state=SINGLE_PARAMS['random_state']
        )
        start = time.time()
        clf.fit(X_train, y_train)
        fit_time = time.time() - start
        y_pred = clf.predict(X_test)
        contamination_val = SINGLE_PARAMS['contamination']
        n_estimators_val = SINGLE_PARAMS['n_estimators']
        max_samples_val = SINGLE_PARAMS['max_samples']

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
        cm_str, contamination_val, n_estimators_val, max_samples_val
    ]
    writer.writerow(row)
    print(f"Results written to {CSV_FILENAME}")


def main(csv_path):
    if not os.path.isfile(csv_path):
        print(f"Error: File '{csv_path}' not found.")
        sys.exit(1)

    df = read_csv_auto_encoding(csv_path)

    # Split before any preprocessing to prevent leakage
    df_train, df_test = train_test_split(df, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    df_train = df_train.reset_index(drop=True)
    df_test = df_test.reset_index(drop=True)
    print(f"Split: {len(df_train)} train, {len(df_test)} test")

    # Fit encoders/scalers on train only, apply to both
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
                'confusion_matrix', 'contamination', 'n_estimators', 'max_samples'
            ]
            writer.writerow(header)
            print("CSV header written.")

        if USE_GRID_SEARCH:
            with parallel_backend('loky', n_jobs=-1):
                for set_idx, group_names in enumerate(PROGRESSIVE_SETS):
                    process_single_feature_set(set_idx, group_names, writer, group_train, group_test, y_train, y_test)
        else:
            for set_idx, group_names in enumerate(PROGRESSIVE_SETS):
                process_single_feature_set(set_idx, group_names, writer, group_train, group_test, y_train, y_test)

    print("\nAll feature sets processed. Results appended to", CSV_FILENAME)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python isolationforest.py <csv_file>")
        sys.exit(1)
    main(sys.argv[1])
