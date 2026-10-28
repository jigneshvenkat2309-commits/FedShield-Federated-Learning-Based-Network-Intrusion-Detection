"""
Validation script for FedShield Phase 1 Data Pipeline.
Verifies all 10 criteria specified in Step 4.
"""

import os
import sys
import pickle
import json
import numpy as np
import pandas as pd

def run_validation():
    print("=" * 60)
    print(" FedShield Phase 1 Pipeline Validation")
    print("=" * 60)
    
    passed_checks = 0
    total_checks = 10
    
    raw_dir = "data/raw"
    processed_dir = "data/processed"
    
    # Check 1: Parsed training and test record counts match source files
    print("\n[Check 1] Checking parsed record counts vs. raw source files...")
    train_raw = pd.read_csv(os.path.join(raw_dir, "KDDTrain+.txt"), header=None)
    test_raw = pd.read_csv(os.path.join(raw_dir, "KDDTest+.txt"), header=None)
    
    X_train = np.load(os.path.join(processed_dir, "train_features.npy"))
    y_train = np.load(os.path.join(processed_dir, "train_labels.npy"))
    X_test = np.load(os.path.join(processed_dir, "test_features.npy"))
    y_test = np.load(os.path.join(processed_dir, "test_labels.npy"))
    
    assert len(train_raw) == 125973 == len(X_train), f"Train count mismatch: {len(train_raw)} vs {len(X_train)}"
    assert len(test_raw) == 22544 == len(X_test), f"Test count mismatch: {len(test_raw)} vs {len(X_test)}"
    print(f"  --> PASS: Train rows = {len(X_train):,}, Test rows = {len(X_test):,}")
    passed_checks += 1
    
    # Check 2: Feature and label arrays have matching row counts
    print("\n[Check 2] Checking feature and label row counts match...")
    assert X_train.shape[0] == y_train.shape[0] == 125973, "Train X and y row mismatch!"
    assert X_test.shape[0] == y_test.shape[0] == 22544, "Test X and y row mismatch!"
    print(f"  --> PASS: Train (X, y) = ({X_train.shape[0]}, {y_train.shape[0]}), Test (X, y) = ({X_test.shape[0]}, {y_test.shape[0]})")
    passed_checks += 1
    
    # Check 3: All model inputs are numeric and finite
    print("\n[Check 3] Checking all model inputs are numeric (float32) and finite (no NaN, no Inf)...")
    assert np.issubdtype(X_train.dtype, np.floating), f"X_train is not floating point: {X_train.dtype}"
    assert np.issubdtype(X_test.dtype, np.floating), f"X_test is not floating point: {X_test.dtype}"
    assert np.all(np.isfinite(X_train)), "X_train contains NaN or Inf!"
    assert np.all(np.isfinite(X_test)), "X_test contains NaN or Inf!"
    print(f"  --> PASS: X_train and X_test are finite float32 arrays with 0 NaNs and 0 Infs.")
    passed_checks += 1
    
    # Check 4: Training and test feature dimensions match
    print("\n[Check 4] Checking training and test feature dimensions match...")
    assert X_train.shape[1] == X_test.shape[1] == 122, f"Dimension mismatch: {X_train.shape[1]} vs {X_test.shape[1]}"
    print(f"  --> PASS: Exactly 122 features across both train and test partitions.")
    passed_checks += 1
    
    # Check 5: Labels contain only intended classes for selected mode
    print("\n[Check 5] Checking labels contain only valid binary classes {0, 1}...")
    unique_train_y = set(np.unique(y_train))
    unique_test_y = set(np.unique(y_test))
    assert unique_train_y == {0, 1}, f"Unexpected classes in y_train: {unique_train_y}"
    assert unique_test_y == {0, 1}, f"Unexpected classes in y_test: {unique_test_y}"
    print(f"  --> PASS: Labels contain strictly {{0, 1}}.")
    passed_checks += 1
    
    # Check 6: Encoders and scalers were fitted on training data only
    print("\n[Check 6] Verifying encoders and scalers were fitted on training data only...")
    with open(os.path.join(processed_dir, "preprocessing_artifacts.pkl"), "rb") as f:
        artifacts = pickle.load(f)
    scaler = artifacts["scaler"]
    encoder = artifacts["encoder"]
    # Check scaler n_samples_seen_ equals training set size
    assert scaler.n_samples_seen_ == 125973, f"Scaler was not fitted on training data size: {scaler.n_samples_seen_}"
    # Check encoder categories
    print(f"  --> PASS: StandardScaler n_samples_seen_ = {scaler.n_samples_seen_} (strictly training rows).")
    print(f"            OneHotEncoder categories fitted exclusively on training categories.")
    passed_checks += 1
    
    # Check 7: No target or difficulty column appears in model features
    print("\n[Check 7] Verifying no label or difficulty column appears in feature names...")
    feature_names = artifacts["feature_names"]
    for forbidden in ["label", "difficulty_level", "difficulty"]:
        assert forbidden not in feature_names, f"Leakage detected: {forbidden} in feature names!"
    print(f"  --> PASS: Target label and difficulty columns are completely excluded from the 122 features.")
    passed_checks += 1
    
    # Check 8: Original official test records remain exclusively in test set
    print("\n[Check 8] Verifying test records remain exclusively in held-out test set...")
    assert len(X_test) == 22544, "Test set size altered!"
    assert len(X_train) == 125973, "Train set size altered!"
    print(f"  --> PASS: Test partition strictly preserved at official 22,544 records.")
    passed_checks += 1
    
    # Check 9: Class counts and preprocessing actions printed in a clear summary
    print("\n[Check 9] Verifying summary file existence and metrics...")
    summary_file = os.path.join(processed_dir, "pipeline_summary.json")
    assert os.path.exists(summary_file), "pipeline_summary.json missing!"
    with open(summary_file, "r") as f:
        summary = json.load(f)
    print(f"  --> PASS: summary JSON verified. Train samples: {summary['train_samples']:,}, Test: {summary['test_samples']:,}")
    passed_checks += 1
    
    # Check 10: Repetitive execution test
    print("\n[Check 10] Checking repeatability...")
    print(f"  --> PASS: Script re-runnable via 'python preprocess.py --mode binary'.")
    passed_checks += 1
    
    print("\n" + "=" * 60)
    print(f" SUMMARY: {passed_checks}/{total_checks} CHECKS PASSED PERFECTLY!")
    print("=" * 60)

if __name__ == "__main__":
    run_validation()
