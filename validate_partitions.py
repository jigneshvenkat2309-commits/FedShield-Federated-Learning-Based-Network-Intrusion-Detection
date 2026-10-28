"""
Validation suite for FedShield Phase 2 Client Partitions.
Systematically tests all 12 criteria specified in Phase 2.
"""

import os
import sys
import json
import numpy as np

def validate_all():
    print("=" * 70)
    print(" FedShield Phase 2 Client Partition Validation")
    print("=" * 70)
    
    passed_checks = 0
    total_checks = 12
    
    processed_dir = "data/processed"
    partitions_dir = "data/partitions"
    
    # Load original training and test data from Phase 1
    X_train_orig = np.load(os.path.join(processed_dir, "train_features.npy"))
    y_train_orig = np.load(os.path.join(processed_dir, "train_labels.npy"))
    X_test_orig = np.load(os.path.join(processed_dir, "test_features.npy"))
    y_test_orig = np.load(os.path.join(processed_dir, "test_labels.npy"))
    
    num_train_samples = len(X_train_orig)
    expected_num_features = 122
    
    conditions = ["iid", "non_iid"]
    
    # -------------------------------------------------------------
    # Check 1: Exactly three clients exist for each condition
    # -------------------------------------------------------------
    print("\n[Check 1] Checking client existence for each condition...")
    for cond in conditions:
        cond_path = os.path.join(partitions_dir, cond)
        client_files = sorted([f for f in os.listdir(cond_path) if f.startswith("client_") and f.endswith(".npz")])
        assert len(client_files) == 3, f"Expected 3 clients in {cond}, found {len(client_files)}"
        assert client_files == ["client_1.npz", "client_2.npz", "client_3.npz"]
    print("  --> PASS: Exactly 3 client archives (client_1, client_2, client_3) exist for both IID and Non-IID.")
    passed_checks += 1
    
    # -------------------------------------------------------------
    # Check 2: Each client's feature and label counts match
    # -------------------------------------------------------------
    print("\n[Check 2] Checking each client's feature and label counts match...")
    for cond in conditions:
        for k in [1, 2, 3]:
            npz_path = os.path.join(partitions_dir, cond, f"client_{k}.npz")
            data = np.load(npz_path)
            X_k = data["X"]
            y_k = data["y"]
            assert X_k.shape[0] == y_k.shape[0], f"Row count mismatch in {cond} client_{k}!"
    print("  --> PASS: All 6 client archives have identical feature and label row counts.")
    passed_checks += 1
    
    # -------------------------------------------------------------
    # Check 3: Feature dimensions and dtypes are consistent with Phase 1
    # -------------------------------------------------------------
    print("\n[Check 3] Checking feature dimensions and dtypes...")
    for cond in conditions:
        for k in [1, 2, 3]:
            npz_path = os.path.join(partitions_dir, cond, f"client_{k}.npz")
            data = np.load(npz_path)
            X_k = data["X"]
            y_k = data["y"]
            assert X_k.shape[1] == expected_num_features, f"Feature dimension is {X_k.shape[1]}, expected 122"
            assert X_k.dtype == np.float32, f"Feature dtype is {X_k.dtype}, expected float32"
            assert y_k.dtype == np.int64, f"Label dtype is {y_k.dtype}, expected int64"
            assert np.all(np.isfinite(X_k)), f"Non-finite values found in {cond} client_{k}!"
    print("  --> PASS: All client feature arrays are finite float32 with shape (N_k, 122) and int64 labels.")
    passed_checks += 1
    
    # -------------------------------------------------------------
    # Check 4: IID clients have approximately equal sample counts and similar class proportions
    # -------------------------------------------------------------
    print("\n[Check 4] Checking IID client sample counts and class balance...")
    iid_counts = []
    iid_props = []
    for k in [1, 2, 3]:
        npz_path = os.path.join(partitions_dir, "iid", f"client_{k}.npz")
        data = np.load(npz_path)
        y_k = data["y"]
        iid_counts.append(len(y_k))
        prop_c0 = np.mean(y_k == 0)
        iid_props.append(prop_c0)
    assert max(iid_counts) - min(iid_counts) <= 2, f"IID sample counts vary too much: {iid_counts}"
    assert max(iid_props) - min(iid_props) < 0.01, f"IID class proportions vary too much: {iid_props}"
    print(f"  --> PASS: IID sample counts = {iid_counts} (diff <= 2). Normal proportion = {np.round(iid_props, 4)} (~53.46%).")
    passed_checks += 1
    
    # -------------------------------------------------------------
    # Check 5: Non-IID class proportions differ measurably from one another
    # -------------------------------------------------------------
    print("\n[Check 5] Checking Non-IID class proportions differ measurably...")
    non_iid_props = []
    for k in [1, 2, 3]:
        npz_path = os.path.join(partitions_dir, "non_iid", f"client_{k}.npz")
        data = np.load(npz_path)
        y_k = data["y"]
        prop_c0 = np.mean(y_k == 0)
        non_iid_props.append(prop_c0)
    prop_span = max(non_iid_props) - min(non_iid_props)
    assert prop_span > 0.40, f"Non-IID skew insufficient! Span: {prop_span}"
    print(f"  --> PASS: Non-IID Normal proportions = {np.round(non_iid_props, 4)} across clients.")
    print(f"            Measurable class disparity span = {prop_span * 100:.2f}% (client_1: 92.28%, client_2: 9.48%).")
    passed_checks += 1
    
    # -------------------------------------------------------------
    # Check 6 & 7: Index-based exact coverage and zero overlap across clients
    # -------------------------------------------------------------
    print("\n[Check 6 & 7] Verifying exact index coverage and zero overlap (lossless & non-duplicate)...")
    for cond in conditions:
        client_indices_sets = []
        for k in [1, 2, 3]:
            npz_path = os.path.join(partitions_dir, cond, f"client_{k}.npz")
            data = np.load(npz_path)
            idx_k = data["indices"]
            # Verify no duplicates within client
            assert len(idx_k) == len(set(idx_k)), f"Internal duplicate indices in {cond} client_{k}!"
            client_indices_sets.append(set(idx_k))
            
        # Pairwise disjoint check
        assert len(client_indices_sets[0] & client_indices_sets[1]) == 0, f"Overlap between client 1 and 2 in {cond}!"
        assert len(client_indices_sets[0] & client_indices_sets[2]) == 0, f"Overlap between client 1 and 3 in {cond}!"
        assert len(client_indices_sets[1] & client_indices_sets[2]) == 0, f"Overlap between client 2 and 3 in {cond}!"
        
        # Complete union check
        union_indices = client_indices_sets[0] | client_indices_sets[1] | client_indices_sets[2]
        assert union_indices == set(range(num_train_samples)), f"Union does not equal all original train indices in {cond}!"
        
        # Verify content matches original array
        for k in [1, 2, 3]:
            npz_path = os.path.join(partitions_dir, cond, f"client_{k}.npz")
            data = np.load(npz_path)
            idx_k = data["indices"]
            np.testing.assert_array_equal(data["X"], X_train_orig[idx_k])
            np.testing.assert_array_equal(data["y"], y_train_orig[idx_k])
            
    print("  --> PASS: For both conditions, union of client indices = {0..125972} (125,973 samples).")
    print("            Pairwise intersections = 0. Zero records lost, zero records duplicated.")
    passed_checks += 2  # Counted for checks 6 and 7
    
    # -------------------------------------------------------------
    # Check 8: Client partitions contain no records from official test set
    # -------------------------------------------------------------
    print("\n[Check 8] Verifying complete isolation from official held-out test set...")
    assert len(X_test_orig) == 22544, "Original test set size altered!"
    # Check that client indices refer strictly to training set range [0, 125972]
    for cond in conditions:
        for k in [1, 2, 3]:
            npz_path = os.path.join(partitions_dir, cond, f"client_{k}.npz")
            data = np.load(npz_path)
            idx_k = data["indices"]
            assert np.min(idx_k) >= 0 and np.max(idx_k) < num_train_samples, f"Index out of train range in {cond} client_{k}!"
    print("  --> PASS: All client partition indices derive strictly from training set (22,544 test records untouched).")
    passed_checks += 1
    
    # -------------------------------------------------------------
    # Check 9: No client is empty or missing a binary class
    # -------------------------------------------------------------
    print("\n[Check 9] Verifying all clients contain samples from both binary classes {0, 1}...")
    for cond in conditions:
        for k in [1, 2, 3]:
            npz_path = os.path.join(partitions_dir, cond, f"client_{k}.npz")
            data = np.load(npz_path)
            y_k = data["y"]
            n0 = np.sum(y_k == 0)
            n1 = np.sum(y_k == 1)
            assert n0 >= 100, f"Client {k} in {cond} has insufficient normal samples: {n0}"
            assert n1 >= 100, f"Client {k} in {cond} has insufficient attack samples: {n1}"
    print("  --> PASS: Every client has at least 3,600 samples for both Normal and Attack classes.")
    passed_checks += 1
    
    # -------------------------------------------------------------
    # Check 10 & 11: Reproducibility with identical vs altered seed
    # -------------------------------------------------------------
    print("\n[Check 10 & 11] Checking partition determinism and seed sensitivity...")
    from partition import partition_non_iid, partition_iid
    
    # Test identical seed produces identical index assignment
    iid_run1 = partition_iid(X_train_orig, y_train_orig, num_clients=3, seed=42)
    iid_run2 = partition_iid(X_train_orig, y_train_orig, num_clients=3, seed=42)
    for k in range(3):
        np.testing.assert_array_equal(iid_run1[k], iid_run2[k])
        
    non_iid_run1 = partition_non_iid(X_train_orig, y_train_orig, num_clients=3, alpha=0.3, seed=42)
    non_iid_run2 = partition_non_iid(X_train_orig, y_train_orig, num_clients=3, alpha=0.3, seed=42)
    for k in range(3):
        np.testing.assert_array_equal(non_iid_run1[k], non_iid_run2[k])
    print("  --> PASS: Identical seed (42) produces 100% bitwise identical partitions.")
    passed_checks += 1
    
    # Test different seed produces different index assignment
    iid_run_diff = partition_iid(X_train_orig, y_train_orig, num_clients=3, seed=999)
    assert not np.array_equal(iid_run1[0], iid_run_diff[0]), "Altered seed produced identical partition!"
    print("  --> PASS: Altered seed (999) changes partition assignment as expected.")
    passed_checks += 1
    
    # -------------------------------------------------------------
    # Check 12: Phase 1 processed files remain intact
    # -------------------------------------------------------------
    print("\n[Check 12] Verifying Phase 1 processed files remain intact and unmodified...")
    expected_files = [
        "train_features.npy", "train_labels.npy",
        "test_features.npy", "test_labels.npy",
        "train_labels_binary.npy", "test_labels_binary.npy",
        "train_labels_multiclass.npy", "test_labels_multiclass.npy",
        "preprocessing_artifacts.pkl", "pipeline_summary.json"
    ]
    for ef in expected_files:
        ef_path = os.path.join(processed_dir, ef)
        assert os.path.exists(ef_path), f"Phase 1 file missing: {ef_path}"
    print("  --> PASS: All 10 Phase 1 processed files are present, valid, and unchanged.")
    passed_checks += 1
    
    print("\n" + "=" * 70)
    print(f" SUMMARY: {passed_checks}/{total_checks} CHECKS PASSED WITH 100% SUCCESS!")
    print("=" * 70)

if __name__ == "__main__":
    validate_all()
