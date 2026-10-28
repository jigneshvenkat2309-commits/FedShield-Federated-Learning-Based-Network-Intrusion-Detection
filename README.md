# FedShield: Federated Learning-Based Network Intrusion Detection

FedShield evaluates Federated Averaging (FedAvg) against centralized and local-only baselines for Network Intrusion Detection Systems (NIDS) across IID and Non-IID client data distributions.

---

## 1. Environment Setup

- **Python Version:** 3.10+ (Tested on Python 3.13.7 64-bit on Windows)
- **Dependencies:** Install minimal requirements:

```bash
pip install -r requirements.txt
```

Verified core libraries:
- `torch>=2.0.0` (Verified: `2.14.1+cpu`)
- `scikit-learn>=1.3.0` (Verified: `1.9.1`)
- `pandas>=2.0.0` (Verified: `3.0.3`)
- `numpy>=1.24.0` (Verified: `2.5.0`)
- `matplotlib>=3.7.0` (Verified: `3.11.1`)

---

## 2. Dataset Acquisition

Download and cryptographically verify the official benchmark NSL-KDD dataset:

```bash
python download_data.py
```

Downloads:
- `data/raw/KDDTrain+.txt` (125,973 records, ~18.22 MB)
- `data/raw/KDDTest+.txt` (22,544 records, ~3.28 MB)

---

## 3. Preprocessing Pipeline

Run the preprocessing pipeline:

```bash
# Binary mode (default: normal = 0, attack = 1)
python preprocess.py --mode binary

# Multiclass mode (5-class: Normal, DoS, Probe, R2L, U2R)
python preprocess.py --mode multiclass
```

### Preprocessing Architecture & Zero-Leakage Guarantees:
1. **Separation of Splits:** The official test set (`KDDTest+.txt`) is strictly held out and never used for fitting any scaler or encoder.
2. **Feature Schema:** 41 traffic features (3 categorical, 38 numerical).
3. **Target & Post-Outcome Exclusion:** The `label` column (index 41) and `difficulty_level` column (index 42) are completely excluded from predictive features.
4. **Categorical Encoding:** `protocol_type`, `service`, and `flag` are encoded using `OneHotEncoder(handle_unknown='ignore', sparse_output=False)` fitted solely on training data (yielding 84 binary indicator columns). Unseen test categories are mapped safely to all-zeros without record loss.
5. **Numerical Scaling:** 38 numerical features are scaled using `StandardScaler` fitted strictly on training data ($N=125,973$).
6. **Total Transformed Features:** Exactly 122 numerical features ($38 + 84$).
7. **Quality Audit:** Verified 0 missing values, 0 non-finite values, and 0 internal duplicate rows in both partitions.

---

## 4. Pipeline Validation

Execute the automated 10-point test suite:

```bash
python validate_pipeline.py
```

Validates record counts, feature dimensions, float32 finite representations, label sets, absence of data leakage, and repeatability.

---

## 5. Simulated Client Partitioning (IID vs Non-IID)

Generate reproducible client partitions across 3 simulated federated clients:

```bash
# Generate both IID and Non-IID partitions (default: alpha=0.3, seed=42)
python partition.py --condition both --alpha 0.3 --seed 42

# Generate only IID partition
python partition.py --condition iid --seed 42

# Generate only Non-IID partition with custom alpha concentration
python partition.py --condition non_iid --alpha 0.5 --seed 42
```

### Partitioning Methodology:
1. **IID Condition (Independent and Identically Distributed):**
   - Stratified random assignment across binary classes.
   - Each client receives an identical sample size (~41,991 records) and matching class proportions (~53.46% Normal, ~46.54% Attack).
2. **Non-IID Condition (Dirichlet Label Skew):**
   - Dirichlet distribution $\mathbf{p}_c \sim \text{Dirichlet}(\alpha, \alpha, \alpha)$ for each class $c \in \{0, 1\}$.
   - With $\alpha = 0.3$:
     - **Client 1:** Skewed heavily toward Normal traffic ($92.28\%$ Normal, $7.72\%$ Attack).
     - **Client 2:** Skewed heavily toward Attack traffic ($9.48\%$ Normal, $90.52\%$ Attack).
     - **Client 3:** Mixed traffic distribution ($69.83\%$ Normal, $30.17\%$ Attack).
   - A minimum threshold (`min_class_ratio=0.05`) guarantees that every client retains thousands of samples from both classes, preventing ill-conditioned loss calculation or non-convergent local training.
3. **Methodological Note on Binary Label Skew:**
   - In binary intrusion detection mode, clients are skewed in their exposure to *benign vs. malicious traffic*.
   - Binary label skew does not represent attack-family specialization (e.g. DoS vs Probe), because all attack types are collapsed to label 1. It simulates edge scenarios such as enterprise corporate workstations (mostly benign) versus honeypots/DMZ servers (mostly attacks).
4. **Partition Integrity & Zero Overlap:**
   - Every one of the 125,973 original training records is assigned to **exactly one client**.
   - Zero duplication, zero dropped records.
   - The 22,544 official test records are strictly isolated in `data/processed/` and never partitioned or accessed during client training.

---

## 6. Partition Validation

Execute the automated 12-point partition test suite:

```bash
python validate_partitions.py
```

Validates:
- Exactly 3 clients exist per condition.
- Feature and label counts match and are consistent with Phase 1 dimensions (`(N_k, 122)` float32).
- IID clients are balanced in sample volume and class ratio.
- Non-IID clients demonstrate significant label skew (disparity span $> 80\%$).
- Exact index union equals $\{0..125972\}$ with zero pairwise overlap.
- Zero test data leakage.
- Reproducibility: identical seeds produce bitwise-identical partitions; different seeds alter assignments.
- Phase 1 files remain intact.

---

## 7. Model Architecture & Baseline Training

FedShield implements a lightweight PyTorch Multi-Layer Perceptron (`IntrusionMLP` in `model.py`):
- **Input Layer:** 122 numerical features
- **Hidden Layers:** Linear(122, 64) → ReLU → Dropout(0.2) → Linear(64, 32) → ReLU → Dropout(0.2)
- **Output Layer:** Linear(32, 1) returning raw scalar logit (unnormalized)
- **Loss:** `BCEWithLogitsLoss`
- **Total Parameters:** 9,985 parameters (39.00 KB)

### Training Commands:

```bash
# 1. Centralized Baseline (All 125,973 train records, 10 epochs, batch size 1024)
python train_centralized.py --epochs 10 --batch-size 1024 --lr 0.001 --seed 42

# 2. Independent Local Baselines (6 models: 3 IID, 3 Non-IID)
python train_local.py --epochs 10 --batch-size 1024 --lr 0.001 --base-seed 100

# 3. Consolidate Baseline Metrics
python consolidate_baselines.py

# 4. Automated Baseline Validation (10 checks)
python validate_baselines.py
```

---

## 8. Consolidated Baseline Evaluation Results

Evaluated on the official held-out test partition ($N=22,544$ records, 9,711 Normal, 12,833 Attack, threshold = 0.5):

| Model | Train Samples | Train Time | Accuracy | Precision | Recall | Macro-F1 | FPR |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Centralized Baseline** | 125,973 | 17.50s | **78.71%** | **96.87%** | 64.68% | **78.65%** | **2.76%** |
| Local Model (IID - client_1) | 41,992 | 5.79s | 74.51% | 91.50% | 60.87% | 74.44% | 7.48% |
| Local Model (IID - client_2) | 41,991 | 5.48s | 75.38% | 91.66% | 62.42% | 75.33% | 7.51% |
| Local Model (IID - client_3) | 41,990 | 5.55s | 74.48% | 91.48% | 60.83% | 74.41% | 7.49% |
| Local Model (NON_IID - client_1) | 47,008 | 6.44s | 71.33% | 91.54% | 54.69% | 71.09% | 6.68% |
| Local Model (NON_IID - client_2) | 51,654 | 6.72s | 81.93% | 91.80% | **74.96%** | 81.91% | 8.85% |
| Local Model (NON_IID - client_3) | 27,311 | 3.81s | 73.19% | 91.54% | 58.30% | 73.07% | 7.12% |

*Key Findings:*
- Centralized baseline achieves the lowest False Positive Rate (2.76%) and highest Precision (96.87%).
- IID local models perform consistently around ~74.8% accuracy and ~61.4% recall, reflecting the reduced local training volume ($N \approx 41,991$).
- Non-IID local models reflect their training skew: Client 1 (92% benign) suffers low attack recall (54.69%), whereas Client 2 (90% attacks) achieves high attack recall (74.96%) but suffers the highest false-alarm rate (8.85%).

---

## 9. Federated Averaging (FedAvg) Simulation

FedShield implements genuine Federated Averaging (FedAvg) with parameter aggregation weighted by client training-set size:

$$w_{t+1} = \sum_{k=1}^K \frac{n_k}{N} w_{t+1}^{(k)}$$

### Execution Commands:

```bash
# 1. Run FedAvg for both IID and Non-IID conditions (10 rounds, 1 local epoch/round)
python federated.py --condition both --rounds 10 --local-epochs 1 --batch-size 1024 --lr 0.001

# 2. Run only IID FedAvg
python federated.py --condition iid --rounds 10 --local-epochs 1

# 3. Run only Non-IID FedAvg
python federated.py --condition non_iid --rounds 10 --local-epochs 1

# 4. Generate Consolidated Full Benchmark Comparison Table
python generate_comparison.py

# 5. Execute Automated FedAvg Validation (12 checks)
python validate_federated.py
```

### Communication Cost Accounting:
- **Model Parameters:** 9,985 `float32` parameters = 39,940 bytes (39.00 KB raw tensors).
- **Serialized Model Payload:** 42,773 bytes (41.77 KB in PyTorch ZIP container).
- **Per Round ($K=3$ clients):**
  - Downlink (Broadcast to 3 clients): $3 \times 41.77 \text{ KB} = 125.31 \text{ KB}$.
  - Uplink (Upload from 3 clients): $3 \times 41.77 \text{ KB} = 125.31 \text{ KB}$.
  - Total per round: $250.62 \text{ KB}$.
- **Cumulative Cost (10 rounds):** $2,566,380 \text{ bytes} = \mathbf{2.566 \text{ MB}}$ (decimal: bytes / $10^6$) $\approx \mathbf{2.447 \text{ MiB}}$ (binary: bytes / $1024^2$) serialized payload ($2.40 \text{ MB}$ / $2.285 \text{ MiB}$ raw parameter payload).
*(Note: Represents simulated model parameter transfers; transport-layer headers, TCP/IP framing, and TLS handshakes are excluded).*

---

## 10. Final Audited Benchmark Performance (Held-Out Test Set $N=22,544$)

All models evaluated strictly on the official held-out test partition ($N=22,544$ records: 9,711 Normal, 12,833 Attack, fixed decision threshold $\tau = 0.5$):

| Model / Architecture | Condition | Scope | Train Samples | Comm Payload (MB / MiB) | Accuracy | Precision | Attack Recall | Macro-F1 | FPR | $\Delta$ Acc vs Local Avg | $\Delta$ Rec vs Local Avg | $\Delta$ Acc vs Centralized | $\Delta$ Rec vs Centralized |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Centralized Baseline** | Centralized | Upper Bound (All Data) | 125,973 | 0.00 / 0.00 | **78.71%** | **96.87%** | **64.68%** | **78.65%** | **2.76%** | — | — | **0.00 pp** | **0.00 pp** |
| Local Model (Client 1) | IID | Uniform Partition | 41,992 | 0.00 / 0.00 | 74.51% | 91.50% | 60.87% | 74.44% | 7.48% | -0.28 pp | -0.50 pp | -4.20 pp | -3.81 pp |
| Local Model (Client 2) | IID | Uniform Partition | 41,991 | 0.00 / 0.00 | 75.38% | 91.66% | 62.42% | 75.33% | 7.51% | +0.59 pp | +1.05 pp | -3.33 pp | -2.26 pp |
| Local Model (Client 3) | IID | Uniform Partition | 41,990 | 0.00 / 0.00 | 74.48% | 91.48% | 60.83% | 74.41% | 7.49% | -0.31 pp | -0.55 pp | -4.23 pp | -3.86 pp |
| **IID Local-Only Average** | IID | Arithmetic Mean (3 Clients) | 41,991 | 0.00 / 0.00 | **74.79%** | **91.55%** | **61.38%** | **74.73%** | **7.49%** | **0.00 pp** | **0.00 pp** | -3.92 pp | -3.31 pp |
| **FedAvg Global (IID)** | IID | Federated (10 Rnds, 3 Clts)| 125,973 | **2.566 / 2.447** | **76.32%** | **91.98%** | **63.98%** | **76.29%** | **7.37%** | **+1.53 pp** | **+2.61 pp** | **-2.39 pp** | **-0.70 pp** |
| Local Model (Client 1) | Non-IID | Skewed (92.3% Benign) | 47,008 | 0.00 / 0.00 | 71.33% | 91.54% | 54.69% | 71.09% | 6.68% | -4.16 pp | -7.96 pp | -7.38 pp | -10.00 pp |
| Local Model (Client 2) | Non-IID | Skewed (90.5% Attack) | 51,654 | 0.00 / 0.00 | 81.93% | 91.80% | 74.96% | 81.91% | 8.85% | +6.45 pp | +12.31 pp | +3.22 pp | +10.27 pp |
| Local Model (Client 3) | Non-IID | Mixed (69.8% Benign) | 27,311 | 0.00 / 0.00 | 73.19% | 91.54% | 58.30% | 73.07% | 7.12% | -2.29 pp | -4.35 pp | -5.51 pp | -6.39 pp |
| **Non-IID Local-Only Avg**| Non-IID | Arithmetic Mean (3 Clients) | 41,991 | 0.00 / 0.00 | **75.48%** | **91.63%** | **62.65%** | **75.36%** | **7.55%** | **0.00 pp** | **0.00 pp** | -3.22 pp | -2.04 pp |
| **FedAvg Global (Non-IID)**| Non-IID | Federated (10 Rnds, 3 Clts)| 125,973 | **2.566 / 2.447** | **74.18%** | **91.38%** | **60.34%** | **74.11%** | **7.52%** | **-1.30 pp** | **-2.31 pp** | **-4.52 pp** | **-4.35 pp** |

*Important Clarification on Comparisons:*
- All differences are computed from full-precision saved metrics before rounding to 2 decimal places.
- **Local-Only Average** represents the arithmetic mean across the three isolated models to serve as an unweighted reference baseline; it does **not** represent an ensemble or voting classifier.
- **Communication Accounting:** **MB** refers strictly to decimal megabytes ($10^6$ bytes) and **MiB** refers strictly to binary mebibytes ($2^{20} = 1,048,576$ bytes). Cumulative serialized payload is $2,566,380\text{ bytes} = \mathbf{2.566\text{ MB}} = \mathbf{2.447\text{ MiB}}$. Raw parameter payload is $2,396,400\text{ bytes} = \mathbf{2.396\text{ MB}} = \mathbf{2.285\text{ MiB}}$. Excludes TCP/IP, TLS 1.3, and application framing overhead.

### Key Audited Insights:
1. **IID FedAvg Bridges the Gap to Centralized:**
   - Under IID partitions, FedAvg achieves **76.32% Accuracy** and **63.98% Recall**, outperforming every single local IID model (+1.53 pp accuracy, +2.61 pp recall over local average).
   - In raw attack detections, IID FedAvg detects **8,211 attacks** on the held-out test set:
     - **+335 additional attacks (+334.67)** relative to the arithmetic mean of isolated local models (7,876.33 attacks).
     - **+399 additional attacks** vs. Client 1 (7,812 attacks).
     - **+200 additional attacks** vs. Client 2 (8,011 attacks).
     - **+405 additional attacks** vs. Client 3 (7,806 attacks). *(Note: The previously cited 405-attack improvement was specifically relative to Client 3, rather than the arithmetic average).*
   - It captures 97.0% of the centralized baseline's accuracy (76.32% vs. 78.71%) and 98.9% of its attack recall (63.98% vs. 64.68%) while keeping all 125,973 training records local.
2. **Mitigation of Local Client Blind Spots in Non-IID:**
   - Client 1 in the Non-IID condition observed $92.3\%$ normal traffic. Its isolated local model suffered a severe blind spot with an attack recall of only **54.69%** (7,018 detections out of 12,833 attacks).
   - Participating in Non-IID FedAvg elevates attack recall to **60.34% (+5.65 pp)**, detecting **+725 additional attacks** (7,743 detections).
   - Client 2 observed $90.5\%$ attack traffic and over-predicted attacks, suffering an elevated False Positive Rate of **8.85%** (859 false alarms on benign test flows). Non-IID FedAvg reduces its FPR to **7.52%**, **saving 129 false alarms** (730 false alarms). Relative to the average local FPR (7.55%, 733.0 false alarms), FedAvg eliminates 3.0 false alarms on average.
3. **The Non-IID Client Drift Cost:**
   - Non-IID FedAvg incurs a $-1.30\text{ pp}$ accuracy penalty and a $-2.31\text{ pp}$ recall penalty relative to the arithmetic average of isolated local models, and is $-2.14\text{ pp}$ lower in accuracy than IID FedAvg (74.18% vs. 76.32%). This reflects client drift resulting from divergent local optimization trajectories under Dirichlet label skew ($\alpha = 0.3$).

---

## 11. Phase 5 Final Reporting & Visualizations

Execute the Phase 5 visualization and validation pipeline:

```bash
# 1. Generate audited final comparison table artifacts (CSV & JSON)
python generate_final_comparison.py

# 2. Render all 5 publication-quality figures (300 DPI PNG)
python plot_results.py

# 3. Execute Phase 5 automated validation suite (7 checks)
python validate_final.py
```

### Publication Figures Catalog (`results/figures/`):
1. **`1_baseline_comparison.png`**: Multi-metric grouped bar chart comparing Centralized vs. Local Averages vs. FedAvg across Accuracy, Precision, Attack Recall, Macro-F1, and False Positive Rate.
2. **`2_client_heterogeneity.png`**: Client data composition and local performance breakdown illustrating label distributions and resulting recall/accuracy divergence.
3. **`3_fedavg_training_trajectory.png`**: Convergence trajectory over 10 communication rounds comparing Training Loss, Test Accuracy, Attack Recall, and Macro-F1 under IID vs. Non-IID.
4. **`4_confusion_matrices.png`**: Side-by-side audited confusion matrices for Centralized, IID FedAvg, and Non-IID FedAvg displaying exact counts and error distribution.
5. **`5_communication_tradeoff.png`**: Communication-performance trade-off curves mapping cumulative payload transfer ($2.566\text{ MB} / 2.447\text{ MiB}$) to test accuracy and attack recall.

---

## 12. Assessment of the Research Claim

**Original Requirement:**
> “The work should demonstrate a measurable improvement over conventional networking or learning baselines under multiple controlled network conditions.”

### Empirical Assessment:
1. **Under IID Network Conditions: REQUIREMENT SATISFIED**
   - **Empirical Evidence:** IID FedAvg achieves **76.32% Accuracy** (+1.53 pp over local average of 74.79%), **63.98% Attack Recall** (+2.61 pp over local average of 61.38%), **76.29% Macro-F1** (+1.56 pp), and **7.37% FPR** (-0.12 pp).
   - **Significance:** Demonstrates a measurable, strictly positive improvement across all classification metrics compared to isolated local training while preserving data locality.
2. **Under Non-IID Network Conditions: NUANCED TRADE-OFF (NOT UNIFORM IMPROVEMENT)**
   - **Empirical Evidence:** Non-IID FedAvg achieves **74.18% Accuracy** (-1.30 pp vs. local arithmetic mean of 75.48%) and **60.34% Attack Recall** (-2.31 pp vs. local arithmetic mean of 62.65%).
   - **Network Benefit (Blind Spot Mitigation):** The global model dramatically rescues severely skewed clients. Client 1 (92.3% benign) had an attack recall of only **54.69%**; FedAvg boosts its detection capability to **60.34%** (+5.65 pp, +725 attacks). Similarly, Client 2 (90.5% attack) had an elevated false alarm rate of **8.85%**; FedAvg reduces this to **7.52%** (saving 129 false alarms on held-out test traffic).
   - **Systemic Cost (Client Drift):** Label divergence ($\alpha = 0.3$) induces local gradient drift, penalizing the simple arithmetic FedAvg global optimum relative to IID FedAvg (-2.14 pp accuracy).
3. **Comparison Against Centralized Baseline:**
   - Centralized training represents the theoretical upper bound (**78.71% Accuracy**, **2.76% FPR**). Federated learning does not outperform centralized training (IID FedAvg is -2.39 pp lower; Non-IID FedAvg is -4.52 pp lower). FedAvg's purpose is privacy-preserving collaborative learning where centralization is prohibited.

---

## 13. System Assumptions and Limitations

1. **Simulation on Single Machine:** All client partitions and server aggregations are simulated in memory/disk on a single CPU machine. Real-world network factors such as latency jitter, packet loss, socket disconnects, and straggler nodes are not modeled.
2. **Communication Overhead Distinction:** The measured $2.566\text{ MB}$ ($2.447\text{ MiB}$) represents serialized PyTorch state dictionary payloads ($2.396\text{ MB} / 2.285\text{ MiB}$ raw parameter bytes). Real-world deployments require TCP/IP framing, TLS 1.3 encryption handshakes, HTTP/2 or gRPC metadata, which introduce an estimated 10–25% transport overhead.
3. **Privacy Scope & Absence of Formal Privacy Guarantees:** Standard Federated Averaging (FedAvg) prevents raw traffic centralization by keeping training flows on local storage. However, **FedAvg does not guarantee formal cryptographic privacy**. Without Differential Privacy (DP-SGD) or Secure Multiparty Computation (SMPC), raw parameter gradients and weight updates remain vulnerable to model-inversion or membership-inference attacks.

---

## 14. Full End-to-End Reproduction Guide

To reproduce the complete FedShield benchmark from scratch:

```bash
# Step 1: Download & verify dataset
python download_data.py

# Step 2: Preprocess features & labels (zero data leakage)
python preprocess.py --mode binary

# Step 3: Generate IID and Non-IID client partitions
python partition.py --condition both --alpha 0.3 --seed 42

# Step 4: Train Centralized Baseline (10 epochs)
python train_centralized.py --epochs 10 --batch-size 1024 --lr 0.001 --seed 42

# Step 5: Train Independent Local Baselines (6 models)
python train_local.py --epochs 10 --batch-size 1024 --lr 0.001 --base-seed 100

# Step 6: Consolidate baseline metrics
python consolidate_baselines.py

# Step 7: Execute FedAvg Simulation (10 rounds, 3 clients, IID & Non-IID)
python federated.py --condition both --rounds 10 --local-epochs 1 --batch-size 1024 --lr 0.001

# Step 8: Consolidate Phase 4 comparison tables
python generate_comparison.py

# Step 9: Generate Phase 5 final comparison tables & render all figures
python generate_final_comparison.py
python plot_results.py

# Step 10: Run the complete 5-phase validation suite (51 tests total)
python -c "import subprocess; [subprocess.run(['python', s], check=True) for s in ['validate_pipeline.py', 'validate_partitions.py', 'validate_baselines.py', 'validate_federated.py', 'validate_final.py']]"
```

---

## 15. Workspace Directory Structure

```text
c:\MY_TECH\CN_project\
├── data\
│   ├── raw\
│   │   ├── KDDTrain+.txt               # 125,973 raw training flows
│   │   └── KDDTest+.txt                # 22,544 held-out test flows
│   ├── processed\
│   │   ├── train_features.npy          # (125973, 122) float32
│   │   ├── train_labels.npy            # (125973,) int64
│   │   ├── test_features.npy           # (22544, 122) float32
│   │   ├── test_labels.npy             # (22544,) int64
│   │   ├── train_labels_binary.npy     # Binary {0, 1}
│   │   ├── test_labels_binary.npy
│   │   ├── preprocessing_artifacts.pkl # Fitted scaler, encoder & metadata
│   │   └── pipeline_summary.json       # Preprocessing audit log
│   └── partitions\
│       ├── iid\
│       │   ├── client_1.npz            # (41992, 122) float32
│       │   ├── client_2.npz            # (41991, 122) float32
│       │   ├── client_3.npz            # (41990, 122) float32
│       │   └── metadata.json
│       ├── non_iid\
│       │   ├── client_1.npz            # (47008, 122) float32 (92.3% benign)
│       │   ├── client_2.npz            # (51654, 122) float32 (90.5% attack)
│       │   ├── client_3.npz            # (27311, 122) float32 (69.8% benign)
│       │   └── metadata.json
│       └── partition_summary.json
├── models\
│   ├── centralized.pt                  # Checkpoint of centralized baseline
│   ├── iid\
│   │   ├── client_1.pt
│   │   ├── client_2.pt
│   │   └── client_3.pt
│   ├── non_iid\
│   │   ├── client_1.pt
│   │   ├── client_2.pt
│   │   └── client_3.pt
│   └── fedavg\
│       ├── iid\
│       │   └── global_final.pt         # Checkpoint of IID FedAvg global model
│       └── non_iid\
│           └── global_final.pt         # Checkpoint of Non-IID FedAvg global model
├── results\
│   ├── figures\
│   │   ├── 1_baseline_comparison.png   # Grouped bar chart across models & metrics
│   │   ├── 2_client_heterogeneity.png   # Class distribution & performance disparity
│   │   ├── 3_fedavg_training_trajectory.png # Convergence across 10 rounds
│   │   ├── 4_confusion_matrices.png    # Audited confusion matrices
│   │   └── 5_communication_tradeoff.png # Communication cost vs. performance
│   ├── baseline_metrics.csv            # Phase 3 baselines table
│   ├── baseline_metrics.json
│   ├── confusion_matrices.json
│   ├── training_history.json
│   ├── fedavg_metrics.csv              # Phase 4 FedAvg metrics
│   ├── fedavg_metrics.json
│   ├── fedavg_round_history.json       # Round-by-round trajectory
│   ├── fedavg_communication.json       # Communication cost accounting
│   ├── full_comparison_table.csv       # Phase 4 comparison table
│   ├── full_comparison_table.json
│   ├── final_comparison.csv            # Phase 5 audited benchmark comparison
│   └── final_comparison.json
├── download_data.py                    # Dataset downloader with SHA-256 verification
├── preprocess.py                       # Zero-leakage feature preprocessor
├── validate_pipeline.py                # Phase 1 test suite (10 checks)
├── partition.py                        # Client partition generator (IID / Non-IID Dirichlet)
├── validate_partitions.py              # Phase 2 test suite (12 checks)
├── model.py                            # IntrusionMLP PyTorch architecture
├── evaluation.py                       # Standardized evaluation & metric computation
├── train_centralized.py                # Centralized training baseline
├── train_local.py                      # Independent local baselines
├── consolidate_baselines.py            # Baseline metric aggregator
├── validate_baselines.py               # Phase 3 test suite (10 checks)
├── federated.py                        # Genuine FedAvg simulation engine
├── generate_comparison.py              # Phase 4 comparison generator
├── validate_federated.py               # Phase 4 test suite (12 checks)
├── generate_final_comparison.py        # Phase 5 audited benchmark table generator
├── plot_results.py                     # Publication-quality plotting suite (5 figures)
├── validate_final.py                   # Phase 5 final validation suite (7 checks)
├── requirements.txt                    # Minimal environment dependencies
└── README.md                           # Comprehensive documentation & reproduction guide
```
