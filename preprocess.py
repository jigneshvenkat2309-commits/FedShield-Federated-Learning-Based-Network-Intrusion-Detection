"""
FedShield: Preprocessing Pipeline for NSL-KDD
Implements reproducible preprocessing with strict train/test separation,
safe categorical encoding, scaling fitted ONLY on training data,
and support for both binary and 5-class multiclass intrusion detection.
"""

import os
import sys
import json
import pickle
import argparse
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# Official NSL-KDD 43-column schema (41 traffic features + label + difficulty score)
COLUMN_NAMES = [
    "duration", "protocol_type", "service", "flag", "src_bytes",
    "dst_bytes", "land", "wrong_fragment", "urgent", "hot",
    "num_failed_logins", "logged_in", "num_compromised", "root_shell",
    "su_attempted", "num_root", "num_file_creations", "num_shells",
    "num_access_files", "num_outbound_cmds", "is_host_login",
    "is_guest_login", "count", "srv_count", "serror_rate",
    "srv_serror_rate", "rerror_rate", "srv_rerror_rate", "same_srv_rate",
    "diff_srv_rate", "srv_diff_host_rate", "dst_host_count",
    "dst_host_srv_count", "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate",
    "dst_host_serror_rate", "dst_host_srv_serror_rate", "dst_host_rerror_rate",
    "dst_host_srv_rerror_rate", "label", "difficulty_level"
]

CATEGORICAL_COLS = ["protocol_type", "service", "flag"]
NUMERICAL_COLS = [c for c in COLUMN_NAMES if c not in CATEGORICAL_COLS and c not in ["label", "difficulty_level"]]

# Taxonomy mapping: 5 conventional categories for NSL-KDD (Normal, DoS, Probe, R2L, U2R)
ATTACK_TAXONOMY_5CLASS = {
    # Normal
    "normal": "Normal",
    # Denial of Service (DoS)
    "back": "DoS", "land": "DoS", "neptune": "DoS", "pod": "DoS", "smurf": "DoS",
    "teardrop": "DoS", "apache2": "DoS", "mailbomb": "DoS", "processtable": "DoS",
    "udpstorm": "DoS",
    # Surveillance / Probe
    "ipsweep": "Probe", "nmap": "Probe", "portsweep": "Probe", "satan": "Probe",
    "mscan": "Probe", "saint": "Probe",
    # Remote to Local (R2L)
    "ftp_write": "R2L", "guess_passwd": "R2L", "imap": "R2L", "multihop": "R2L",
    "phf": "R2L", "spy": "R2L", "warezclient": "R2L", "warezmaster": "R2L",
    "httptunnel": "R2L", "named": "R2L", "sendmail": "R2L", "snmpgetattack": "R2L",
    "snmpguess": "R2L", "worm": "R2L", "xlock": "R2L", "xsnoop": "R2L",
    # User to Root (U2R)
    "buffer_overflow": "U2R", "loadmodule": "U2R", "perl": "U2R", "rootkit": "U2R",
    "ps": "U2R", "sqlattack": "U2R", "xterm": "U2R"
}

CLASS_TO_INT_5CLASS = {
    "Normal": 0,
    "DoS": 1,
    "Probe": 2,
    "R2L": 3,
    "U2R": 4
}


def load_raw_dataset(raw_dir):
    """Load raw KDDTrain+.txt and KDDTest+.txt with schema verification."""
    train_path = os.path.join(raw_dir, "KDDTrain+.txt")
    test_path = os.path.join(raw_dir, "KDDTest+.txt")
    
    if not os.path.exists(train_path) or not os.path.exists(test_path):
        raise FileNotFoundError(
            f"Dataset files not found in {raw_dir}. Run download_data.py first."
        )
        
    train_df = pd.read_csv(train_path, header=None, names=COLUMN_NAMES)
    test_df = pd.read_csv(test_path, header=None, names=COLUMN_NAMES)
    
    return train_df, test_df


def inspect_data_quality(train_df, test_df):
    """Audit missing values, infinite values, and record integrity."""
    report = {
        "train_shape": list(train_df.shape),
        "test_shape": list(test_df.shape),
        "train_nulls": int(train_df.isna().sum().sum()),
        "test_nulls": int(test_df.isna().sum().sum()),
    }
    
    # Audit numeric columns for non-finite values (NaN / Inf)
    train_num_vals = train_df[NUMERICAL_COLS].to_numpy(dtype=np.float64)
    test_num_vals = test_df[NUMERICAL_COLS].to_numpy(dtype=np.float64)
    
    report["train_non_finite"] = int(np.logical_not(np.isfinite(train_num_vals)).sum())
    report["test_non_finite"] = int(np.logical_not(np.isfinite(test_num_vals)).sum())
    
    # Audit internal duplicates
    report["train_duplicate_rows"] = int(train_df.duplicated().sum())
    report["test_duplicate_rows"] = int(test_df.duplicated().sum())
    
    print("\n--- [Data Quality Audit] ---")
    print(f"Train Records: {train_df.shape[0]}, Features + Labels + Difficulty: {train_df.shape[1]}")
    print(f"Test Records:  {test_df.shape[0]}, Features + Labels + Difficulty: {test_df.shape[1]}")
    print(f"Missing Values (Train / Test): {report['train_nulls']} / {report['test_nulls']}")
    print(f"Non-finite Values (Train / Test): {report['train_non_finite']} / {report['test_non_finite']}")
    print(f"Internal Duplicates (Train / Test): {report['train_duplicate_rows']} / {report['test_duplicate_rows']}")
    
    return report


def encode_labels(labels_series, mode="binary"):
    """
    Encode attack labels.
    - binary: normal -> 0, all attacks -> 1
    - multiclass: 5 categories (Normal: 0, DoS: 1, Probe: 2, R2L: 3, U2R: 4)
    """
    labels_clean = labels_series.astype(str).str.strip().str.lower()
    
    if mode == "binary":
        # 0 = normal, 1 = attack
        int_labels = (labels_clean != "normal").astype(np.int64).to_numpy()
        class_mapping = {"normal": 0, "attack": 1}
        return int_labels, class_mapping
    elif mode == "multiclass":
        unmapped = set(labels_clean.unique()) - set(ATTACK_TAXONOMY_5CLASS.keys())
        if unmapped:
            raise ValueError(f"Unmapped attack categories found: {unmapped}")
        categories = labels_clean.map(ATTACK_TAXONOMY_5CLASS)
        int_labels = categories.map(CLASS_TO_INT_5CLASS).astype(np.int64).to_numpy()
        return int_labels, CLASS_TO_INT_5CLASS
    else:
        raise ValueError(f"Unsupported mode: {mode}. Must be 'binary' or 'multiclass'.")


def fit_preprocessor(train_df):
    """
    Fit OneHotEncoder and StandardScaler strictly on training data.
    - Encoders handle unseen test categories safely via handle_unknown='ignore'.
    - Difficulty score and label are excluded from features.
    """
    # 1. Fit OneHotEncoder on categorical features
    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=np.float32)
    encoder.fit(train_df[CATEGORICAL_COLS])
    
    # 2. Fit StandardScaler on numerical features
    scaler = StandardScaler()
    scaler.fit(train_df[NUMERICAL_COLS].to_numpy(dtype=np.float64))
    
    # Derive feature column names
    encoded_cat_names = list(encoder.get_feature_names_out(CATEGORICAL_COLS))
    all_feature_names = NUMERICAL_COLS + encoded_cat_names
    
    print("\n--- [Fitted Preprocessing Transformers] ---")
    print(f"Strict fitting partition: TRAINING DATA ONLY ({len(train_df)} rows)")
    print(f"Numerical features scaled: {len(NUMERICAL_COLS)}")
    print(f"Categorical features encoded: {len(CATEGORICAL_COLS)} -> {len(encoded_cat_names)} one-hot columns")
    print(f"Total model input features: {len(all_feature_names)}")
    
    artifacts = {
        "scaler": scaler,
        "encoder": encoder,
        "numerical_cols": NUMERICAL_COLS,
        "categorical_cols": CATEGORICAL_COLS,
        "feature_names": all_feature_names,
        "num_features": len(all_feature_names)
    }
    return artifacts


def transform_data(df, artifacts):
    """Transform features using pre-fitted transformers without data leakage."""
    scaler = artifacts["scaler"]
    encoder = artifacts["encoder"]
    
    # Transform numerical features
    num_scaled = scaler.transform(df[NUMERICAL_COLS].to_numpy(dtype=np.float64)).astype(np.float32)
    
    # Transform categorical features (unseen test categories map to all zeros)
    cat_encoded = encoder.transform(df[CATEGORICAL_COLS]).astype(np.float32)
    
    # Concatenate [Numerical, Categorical]
    X = np.hstack([num_scaled, cat_encoded]).astype(np.float32)
    return X


def run_pipeline(raw_dir="data/raw", output_dir="data/processed", mode="binary", random_seed=42):
    """Execute complete reproducible preprocessing pipeline."""
    np.random.seed(random_seed)
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"==================================================")
    print(f" FedShield Data Pipeline (Mode: {mode}, Seed: {random_seed})")
    print(f"==================================================")
    
    # 1. Load raw data
    train_df, test_df = load_raw_dataset(raw_dir)
    
    # 2. Audit quality
    quality_report = inspect_data_quality(train_df, test_df)
    
    # 3. Fit transformers ONLY on training data
    artifacts = fit_preprocessor(train_df)
    
    # 4. Transform features (train and test)
    X_train = transform_data(train_df, artifacts)
    X_test = transform_data(test_df, artifacts)
    
    # 5. Encode labels
    # Primary requested mode
    y_train, class_mapping = encode_labels(train_df["label"], mode=mode)
    y_test, _ = encode_labels(test_df["label"], mode=mode)
    
    # Also encode both binary and multiclass for flexibility
    y_train_bin, _ = encode_labels(train_df["label"], mode="binary")
    y_test_bin, _ = encode_labels(test_df["label"], mode="binary")
    y_train_multi, _ = encode_labels(train_df["label"], mode="multiclass")
    y_test_multi, _ = encode_labels(test_df["label"], mode="multiclass")
    
    # 6. Verify assertions and absence of leakage
    assert X_train.shape[0] == y_train.shape[0] == 125973, "Train row count mismatch!"
    assert X_test.shape[0] == y_test.shape[0] == 22544, "Test row count mismatch!"
    assert X_train.shape[1] == X_test.shape[1] == 122, "Feature dimension mismatch!"
    assert np.all(np.isfinite(X_train)), "Non-finite values found in X_train!"
    assert np.all(np.isfinite(X_test)), "Non-finite values found in X_test!"
    
    # Confirm label and difficulty fields do not exist in model inputs
    for forbidden in ["label", "difficulty_level"]:
        assert forbidden not in artifacts["feature_names"], f"Data leakage: {forbidden} found in features!"
    
    # 7. Print class distribution summary
    print("\n--- [Class Distribution Summary] ---")
    if mode == "binary":
        train_normal = int(np.sum(y_train == 0))
        train_attack = int(np.sum(y_train == 1))
        test_normal = int(np.sum(y_test == 0))
        test_attack = int(np.sum(y_test == 1))
        print(f"Train - Normal (0): {train_normal:,} ({train_normal/len(y_train)*100:.2f}%), Attack (1): {train_attack:,} ({train_attack/len(y_train)*100:.2f}%)")
        print(f"Test  - Normal (0): {test_normal:,} ({test_normal/len(y_test)*100:.2f}%), Attack (1): {test_attack:,} ({test_attack/len(y_test)*100:.2f}%)")
    else:
        for cat_name, code in CLASS_TO_INT_5CLASS.items():
            tr_cnt = int(np.sum(y_train == code))
            te_cnt = int(np.sum(y_test == code))
            print(f"  Class {code} ({cat_name:6s}) -> Train: {tr_cnt:6d} ({tr_cnt/len(y_train)*100:5.2f}%), Test: {te_cnt:5d} ({te_cnt/len(y_test)*100:5.2f}%)")
            
    # 8. Save processed files
    # Standard active target files
    np.save(os.path.join(output_dir, "train_features.npy"), X_train)
    np.save(os.path.join(output_dir, "train_labels.npy"), y_train)
    np.save(os.path.join(output_dir, "test_features.npy"), X_test)
    np.save(os.path.join(output_dir, "test_labels.npy"), y_test)
    
    # Binary and multiclass explicit label sets
    np.save(os.path.join(output_dir, "train_labels_binary.npy"), y_train_bin)
    np.save(os.path.join(output_dir, "test_labels_binary.npy"), y_test_bin)
    np.save(os.path.join(output_dir, "train_labels_multiclass.npy"), y_train_multi)
    np.save(os.path.join(output_dir, "test_labels_multiclass.npy"), y_test_multi)
    
    # Save preprocessing metadata and fitted transformers
    artifacts_path = os.path.join(output_dir, "preprocessing_artifacts.pkl")
    with open(artifacts_path, "wb") as f:
        pickle.dump({
            "scaler": artifacts["scaler"],
            "encoder": artifacts["encoder"],
            "feature_names": artifacts["feature_names"],
            "numerical_cols": artifacts["numerical_cols"],
            "categorical_cols": artifacts["categorical_cols"],
            "class_mapping": class_mapping,
            "mode": mode,
            "random_seed": random_seed
        }, f)
        
    summary_path = os.path.join(output_dir, "pipeline_summary.json")
    summary = {
        "mode": mode,
        "random_seed": random_seed,
        "train_samples": int(X_train.shape[0]),
        "test_samples": int(X_test.shape[0]),
        "num_features": int(X_train.shape[1]),
        "numerical_features_count": len(NUMERICAL_COLS),
        "categorical_features_count": len(CATEGORICAL_COLS),
        "one_hot_columns_count": len(artifacts["feature_names"]) - len(NUMERICAL_COLS),
        "class_mapping": class_mapping,
        "quality_report": quality_report
    }
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
        
    print(f"\n--- [Pipeline Complete] ---")
    print(f"Processed arrays saved to: {os.path.abspath(output_dir)}")
    print(f"  train_features.npy: {X_train.shape}, dtype={X_train.dtype}")
    print(f"  train_labels.npy:   {y_train.shape}, dtype={y_train.dtype}")
    print(f"  test_features.npy:  {X_test.shape}, dtype={X_test.dtype}")
    print(f"  test_labels.npy:    {y_test.shape}, dtype={y_test.dtype}")
    print(f"  preprocessing_artifacts.pkl & pipeline_summary.json saved successfully.")
    
    return summary


def main():
    parser = argparse.ArgumentParser(description="FedShield Data Preprocessing Pipeline")
    parser.add_argument("--mode", choices=["binary", "multiclass"], default="binary",
                        help="Classification mode: binary (normal vs attack) or multiclass (5 classes)")
    parser.add_argument("--raw-dir", default="data/raw", help="Path to raw NSL-KDD dataset directory")
    parser.add_argument("--output-dir", default="data/processed", help="Path to output processed directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    
    args = parser.parse_args()
    run_pipeline(raw_dir=args.raw_dir, output_dir=args.output_dir, mode=args.mode, random_seed=args.seed)


if __name__ == "__main__":
    main()
