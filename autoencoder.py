import sys
import time
import csv
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

# Try to import tqdm for progress bar
try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

# ==================== CONFIGURATION ====================
DATASET_SIZE = "10k"
CSV_FILENAME = "autoencoder_results.csv"
OUTPUT_FN_CSV = True

GRID_PARAMS = {
    'encoding_dim': [8, 16, 32],
    'learning_rate': [0.001, 0.0001],
    'dropout': [0.0, 0.2],
    'threshold_percentile': [90, 95, 99]
}
EPOCHS = 50
BATCH_SIZE = 256
EARLY_STOPPING_PATIENCE = 5

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
# =======================================================

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

def build_autoencoder(input_dim, encoding_dim, dropout_rate):
    model = Sequential([
        Dense(encoding_dim * 2, activation='relu', input_shape=(input_dim,)),
        Dropout(dropout_rate),
        Dense(encoding_dim, activation='relu'),
        Dropout(dropout_rate),
        Dense(encoding_dim * 2, activation='relu'),
        Dense(input_dim, activation='linear')
    ])
    return model

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
            writer.writerow(['dataset_size', 'feature_set', 'encoding_dim', 'learning_rate', 'dropout',
                             'threshold_percentile', 'val_f1', 'test_accuracy', 'test_precision',
                             'test_recall', 'test_f1', 'test_fpr', 'training_time_seconds',
                             'confusion_matrix'])

        for set_idx, group_names in enumerate(PROGRESSIVE_SETS):
            set_name = SET_NAMES[set_idx]
            print(f"\n=== Feature set: {set_name} ===", flush=True)
            X_train = combine_groups(group_train, group_names)
            X_val = combine_groups(group_val, group_names)
            X_test = combine_groups(group_test, group_names)

            input_dim = X_train.shape[1]
            best_val_f1 = -1
            best_params = None
            best_weights = None   # store weights instead of the model object
            best_threshold = None

            from itertools import product
            keys = list(GRID_PARAMS.keys())
            values = [GRID_PARAMS[k] for k in keys]
            combos = list(product(*values))
            total_combos = len(combos)
            print(f"Grid search over {total_combos} combinations...", flush=True)

            start_grid = time.time()
            combo_iter = enumerate(combos, 1)
            if HAS_TQDM:
                combo_iter = tqdm(combo_iter, total=total_combos, desc="Grid search")
            for combo_idx, combo in combo_iter:
                params = dict(zip(keys, combo))
                encoding_dim = params['encoding_dim']
                lr = params['learning_rate']
                dropout_rate = params['dropout']
                thresh_perc = params['threshold_percentile']

                if not HAS_TQDM:
                    print(f"[{combo_idx}/{total_combos}] enc={encoding_dim}, lr={lr}, drop={dropout_rate}, thresh={thresh_perc}%", end='', flush=True)

                model = build_autoencoder(input_dim, encoding_dim, dropout_rate)
                model.compile(optimizer=Adam(learning_rate=lr), loss='mse')
                early_stop = EarlyStopping(patience=EARLY_STOPPING_PATIENCE, restore_best_weights=True)
                start_time = time.time()
                model.fit(X_train, X_train,
                          epochs=EPOCHS,
                          batch_size=BATCH_SIZE,
                          validation_split=0.1,
                          callbacks=[early_stop],
                          verbose=0)
                train_time_this = time.time() - start_time

                train_pred = model.predict(X_train, verbose=0)
                train_mse = np.mean(np.square(X_train - train_pred), axis=1)
                threshold = np.percentile(train_mse, thresh_perc)

                val_pred = model.predict(X_val, verbose=0)
                val_mse = np.mean(np.square(X_val - val_pred), axis=1)
                y_val_pred = (val_mse > threshold).astype(int)

                f1 = f1_score(y_val, y_val_pred)
                if not HAS_TQDM:
                    print(f" -> F1={f1:.4f} (time={train_time_this:.1f}s)", flush=True)

                if f1 > best_val_f1:
                    best_val_f1 = f1
                    best_params = params
                    best_threshold = threshold
                    # Save weights rather than the model reference so that
                    # clear_session() below cannot invalidate them
                    best_weights = model.get_weights()
                    best_encoding_dim = encoding_dim
                    best_dropout = dropout_rate

                # Safe to clear now — best_weights holds a plain numpy list
                tf.keras.backend.clear_session()

            grid_time = time.time() - start_grid
            print(f"Grid search completed in {grid_time:.2f}s. Best validation F1: {best_val_f1:.4f}", flush=True)
            print(f"Best parameters: {best_params}", flush=True)

            # Rebuild the best architecture and restore saved weights
            best_model = build_autoencoder(input_dim, best_encoding_dim, best_dropout)
            best_model.compile(optimizer=Adam(learning_rate=best_params['learning_rate']), loss='mse')
            best_model.set_weights(best_weights)

            test_pred = best_model.predict(X_test, verbose=0)
            test_mse = np.mean(np.square(X_test - test_pred), axis=1)
            y_test_pred = (test_mse > best_threshold).astype(int)

            cm = confusion_matrix(y_test, y_test_pred)
            tn, fp, fn, tp = cm.ravel()
            acc = accuracy_score(y_test, y_test_pred)
            prec = precision_score(y_test, y_test_pred)
            rec = recall_score(y_test, y_test_pred)
            f1 = f1_score(y_test, y_test_pred)
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

            print(f"Test results:")
            print(f"Confusion matrix:\n{cm}")
            print(f"Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}, FPR: {fpr:.4f}", flush=True)

            if OUTPUT_FN_CSV and fn > 0:
                fn_indices = np.where((y_test == 1) & (y_test_pred == 0))[0]
                fn_df = df_test.iloc[fn_indices].copy()
                fn_df.to_csv(f"fn_{DATASET_SIZE}_{set_name}.csv", index=False)
                print(f"Saved {len(fn_indices)} false negatives to fn_{DATASET_SIZE}_{set_name}.csv", flush=True)

            writer.writerow([
                DATASET_SIZE, set_name,
                best_params['encoding_dim'], best_params['learning_rate'], best_params['dropout'],
                best_params['threshold_percentile'],
                best_val_f1, acc, prec, rec, f1, fpr,
                grid_time,
                f"[[{tn},{fp}],[{fn},{tp}]]"
            ])

            # Clean up the rebuilt best model before the next feature set
            tf.keras.backend.clear_session()

    print(f"\nAll feature sets processed. Results appended to {CSV_FILENAME}", flush=True)
    if val_csv == "temp_val.csv":
        os.remove(val_csv)
        os.remove(test_csv)

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python autoencoder.py <train_benign.csv> <validation_mixed.csv> [test_mixed.csv]")
        sys.exit(1)
    train_file = sys.argv[1]
    val_file = sys.argv[2]
    test_file = sys.argv[3] if len(sys.argv) > 3 else None
    main(train_file, val_file, test_file)
