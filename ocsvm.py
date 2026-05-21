import sys
import time
import csv
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.svm import OneClassSVM
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

DATASET_SIZE = "100k"
CSV_FILENAME = "ocsvm_results_25mal.csv"
OUTPUT_FN_CSV = True

# Set to True to run grid search, False to use SINGLE_PARAMS
USE_GRID_SEARCH = True

# Single fit parameters (used when USE_GRID_SEARCH = False)
SINGLE_PARAMS = {
    'nu': 0.5583,
    'gamma': 'auto',
    'kernel': 'rbf'
}

# Grid search parameter ranges (used when USE_GRID_SEARCH = True)
GRID_PARAMS = {
    'nu': [0.125, 0.15, 0.175, 0.2, 0.225, 0.25],
    'gamma': [0.1],
    'kernel': ['poly', 'sigmoid'],
    'coef0': [0.0, 0.1, 0.5, 0.9, 1.0]      # only used for poly/sigmoid
}

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

# ========== Preprocessing ==========
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
    return np.hstack([group_data[name] for name in group_names])

def read_csv_auto(filepath):
    for enc in ['utf-16', 'utf-8']:
        try:
            return pd.read_csv(filepath, encoding=enc)
        except:
            continue
    return pd.read_csv(filepath, encoding='utf-8', errors='replace')

# ========== Main ==========
def main(train_csv, val_csv, test_csv=None):
    if test_csv is None:
        print("Only two files provided. Splitting second file into validation (50%) and test (50%).")
        df_mixed = read_csv_auto(val_csv)
        split_idx = len(df_mixed) // 2
        df_val = df_mixed.iloc[:split_idx]
        df_test = df_mixed.iloc[split_idx:]
        val_csv = "temp_val.csv"
        test_csv = "temp_test.csv"
        df_val.to_csv(val_csv, index=False)
        df_test.to_csv(test_csv, index=False)
    else:
        df_val = read_csv_auto(val_csv)
        df_test = read_csv_auto(test_csv)

    df_train = read_csv_auto(train_csv)

    # Preprocess
    group_train, encoders, scalers = build_group_features(df_train, mode='train')
    group_val = build_group_features(df_val, mode='test', encoders=encoders, scalers=scalers)
    group_test = build_group_features(df_test, mode='test', encoders=encoders, scalers=scalers)

    y_val = (df_val['label'] == 'Malicious').astype(int).values
    y_test = (df_test['label'] == 'Malicious').astype(int).values

    file_exists = os.path.isfile(CSV_FILENAME)
    with open(CSV_FILENAME, 'a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow([
                'dataset_size', 'feature_set',
                'test_accuracy', 'test_precision', 'test_recall', 'test_f1', 'test_fpr',
                'training_time_seconds', 'confusion_matrix',
                'nu', 'kernel', 'gamma', 'coef0'
            ])

        for set_idx, group_names in enumerate(PROGRESSIVE_SETS):
            set_name = SET_NAMES[set_idx]
            print(f"\n=== Feature set: {set_name} ===")
            X_train = combine_groups(group_train, group_names)
            X_val = combine_groups(group_val, group_names)
            X_test = combine_groups(group_test, group_names)

            if USE_GRID_SEARCH:
                best_f1 = -1
                best_params = None
                total_combos = len(list(ParameterGrid(GRID_PARAMS)))
                print(f"Grid search over {total_combos} combinations...")
                start_grid = time.time()
                for params in ParameterGrid(GRID_PARAMS):
                    model = OneClassSVM(**params)
                    model.fit(X_train)
                    y_val_pred = model.predict(X_val)
                    y_val_pred = np.where(y_val_pred == 1, 0, 1)
                    f1 = f1_score(y_val, y_val_pred)
                    if f1 > best_f1:
                        best_f1 = f1
                        best_params = params
                grid_time = time.time() - start_grid
                print(f"Grid search completed in {grid_time:.2f}s. Best F1 on validation: {best_f1:.4f}")
                print(f"Best parameters: {best_params}")
                final_params = best_params
                val_f1 = best_f1
            else:
                print(f"Using single parameters: {SINGLE_PARAMS}")
                final_params = SINGLE_PARAMS
                model = OneClassSVM(**final_params)
                model.fit(X_train)
                y_val_pred = model.predict(X_val)
                y_val_pred = np.where(y_val_pred == 1, 0, 1)
                val_f1 = f1_score(y_val, y_val_pred)
                print(f"Validation F1: {val_f1:.4f}")

            # Train final model with best/single params on full training set
            final_model = OneClassSVM(**final_params)
            start_train = time.time()
            final_model.fit(X_train)
            train_time = time.time() - start_train

            # Evaluate on test set
            y_test_pred = final_model.predict(X_test)
            y_test_pred = np.where(y_test_pred == 1, 0, 1)
            cm = confusion_matrix(y_test, y_test_pred)
            tn, fp, fn, tp = cm.ravel()
            acc = accuracy_score(y_test, y_test_pred)
            prec = precision_score(y_test, y_test_pred)
            rec = recall_score(y_test, y_test_pred)
            f1 = f1_score(y_test, y_test_pred)
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

            print(f"Test results:")
            print(f"Confusion matrix:\n{cm}")
            print(f"Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}, FPR: {fpr:.4f}")

            if OUTPUT_FN_CSV and fn > 0:
                fn_indices = np.where((y_test == 1) & (y_test_pred == 0))[0]
                fn_df = df_test.iloc[fn_indices].copy()
                fn_df.to_csv(f"fn_{DATASET_SIZE}_{set_name}.csv", index=False)
                print(f"Saved {len(fn_indices)} false negatives to fn_{DATASET_SIZE}_{set_name}.csv")

            # Fixed: removed coef0 from row to match updated header
            writer.writerow([
                DATASET_SIZE, set_name,
                acc, prec, rec, f1, fpr, train_time,
                f"[[{tn},{fp}],[{fn},{tp}]]",
                final_params.get('nu'),
                final_params.get('kernel'),
                final_params.get('gamma'),
                final_params.get('coef0')
            ])

    print(f"\nAll feature sets processed. Results appended to {CSV_FILENAME}")
    if val_csv == "temp_val.csv":
        os.remove(val_csv)
        os.remove(test_csv)

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python ocsvm.py <train_benign.csv> <validation_mixed.csv> [test_mixed.csv]")
        print("  If test_mixed.csv is omitted, validation_mixed.csv is split 50/50 into validation and test.")
        sys.exit(1)
    train_file = sys.argv[1]
    val_file = sys.argv[2]
    test_file = sys.argv[3] if len(sys.argv) > 3 else None
    main(train_file, val_file, test_file)
