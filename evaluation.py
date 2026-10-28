"""
FedShield: Evaluation Module
Computes comprehensive intrusion detection metrics on the held-out test set:
- Accuracy, Precision, Recall, Macro-F1, Binary-F1, False Positive Rate (FPR)
- Confusion Matrix (TN, FP, FN, TP) with Attack as the positive class.
- Safe handling of edge cases and non-finite numbers.
"""

import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


def evaluate_model(model, X_test, y_test, device="cpu", threshold=0.5, batch_size=2048):
    """
    Evaluates a PyTorch binary classifier on the held-out test set.
    
    Parameters:
      model: PyTorch nn.Module returning raw logits
      X_test: np.ndarray or torch.Tensor of shape (N, 122)
      y_test: np.ndarray or torch.Tensor of shape (N,)
      device: torch device ('cpu')
      threshold: classification threshold for Attack probability (default: 0.5)
      batch_size: batch size for evaluation inference
      
    Returns:
      metrics: dict containing all scalar performance metrics and confusion matrix
    """
    model.eval()
    model.to(device)
    criterion = nn.BCEWithLogitsLoss()
    
    if isinstance(X_test, np.ndarray):
        X_tensor = torch.from_numpy(X_test).float()
    else:
        X_tensor = X_test.float()
        
    if isinstance(y_test, np.ndarray):
        y_true = y_test.astype(np.int64)
        y_tensor = torch.from_numpy(y_test).float().unsqueeze(1)
    else:
        y_true = y_test.cpu().numpy().astype(np.int64)
        y_tensor = y_test.float().unsqueeze(1)
        
    num_samples = len(X_tensor)
    total_loss = 0.0
    all_probs = []
    
    with torch.no_grad():
        for i in range(0, num_samples, batch_size):
            bx = X_tensor[i:i + batch_size].to(device)
            by = y_tensor[i:i + batch_size].to(device)
            
            logits = model(bx)
            loss = criterion(logits, by)
            total_loss += loss.item() * len(bx)
            
            probs = torch.sigmoid(logits)
            all_probs.append(probs.cpu().numpy())
            
    avg_loss = total_loss / num_samples
    all_probs = np.vstack(all_probs).flatten()
    
    # Positive class = 1 (Attack), Negative class = 0 (Normal)
    y_pred = (all_probs >= threshold).astype(np.int64)
    
    # Compute confusion matrix: [[TN, FP], [FN, TP]]
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
    rec = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
    f1_binary = f1_score(y_true, y_pred, pos_label=1, zero_division=0)
    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
    
    # False Positive Rate: FP / (FP + TN)
    # Fraction of benign connections incorrectly flagged as attacks
    fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
    
    metrics = {
        "test_loss": float(avg_loss),
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1_binary": float(f1_binary),
        "f1_macro": float(f1_macro),
        "false_positive_rate": float(fpr),
        "confusion_matrix": {
            "TN": int(tn),
            "FP": int(fp),
            "FN": int(fn),
            "TP": int(tp),
            "total": int(tn + fp + fn + tp)
        },
        "threshold": float(threshold),
        "total_test_samples": int(num_samples)
    }
    
    return metrics


def format_metrics_table(metrics_dict, title="Model Evaluation"):
    """Format single model metrics into a readable string."""
    cm = metrics_dict["confusion_matrix"]
    s = f"\n=== {title} ===\n"
    s += f"  Accuracy:      {metrics_dict['accuracy']*100:.2f}%\n"
    s += f"  Precision:     {metrics_dict['precision']*100:.2f}%\n"
    s += f"  Recall:        {metrics_dict['recall']*100:.2f}%\n"
    s += f"  Macro-F1:      {metrics_dict['f1_macro']*100:.2f}%\n"
    s += f"  FPR:           {metrics_dict['false_positive_rate']*100:.2f}%\n"
    s += f"  Test Loss:     {metrics_dict['test_loss']:.4f}\n"
    s += f"  Confusion Matrix: [TN={cm['TN']:,}, FP={cm['FP']:,}, FN={cm['FN']:,}, TP={cm['TP']:,}] (Total={cm['total']:,})\n"
    return s
