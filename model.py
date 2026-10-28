"""
FedShield: Lightweight Neural Network Classifier
Implements a multi-layer perceptron (MLP) for network intrusion detection.
Returns raw logits for numerical stability with BCEWithLogitsLoss.
"""

import torch
import torch.nn as nn


class IntrusionMLP(nn.Module):
    """
    Lightweight Feedforward Neural Network for Binary Intrusion Detection.
    Architecture:
      Input (122) -> Linear(122, 64) -> ReLU -> Dropout(0.2)
                  -> Linear(64, 32)  -> ReLU -> Dropout(0.2)
                  -> Linear(32, 1)   -> Raw Logit Output
    """
    def __init__(self, input_dim=122, hidden_dims=(64, 32), dropout_rate=0.2):
        super(IntrusionMLP, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dims = list(hidden_dims)
        self.dropout_rate = dropout_rate
        
        layers = []
        prev_dim = input_dim
        for h_dim in self.hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.ReLU())
            if dropout_rate > 0.0:
                layers.append(nn.Dropout(p=dropout_rate))
            prev_dim = h_dim
            
        # Final output layer producing raw scalar logit per sample
        layers.append(nn.Linear(prev_dim, 1))
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        """Returns raw unnormalized logit (shape: [batch_size, 1])."""
        return self.network(x)
        
    def predict_proba(self, x):
        """Returns probability of positive class (Attack) using Sigmoid."""
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            probs = torch.sigmoid(logits)
        return probs

    def count_parameters(self):
        """Return total trainable parameters and model size in bytes."""
        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        size_bytes = total_params * 4  # float32 = 4 bytes
        return total_params, size_bytes


def create_model(input_dim=122, hidden_dims=(64, 32), dropout_rate=0.2, seed=None):
    """Factory function with optional seed initialization."""
    if seed is not None:
        torch.manual_seed(seed)
    model = IntrusionMLP(input_dim=input_dim, hidden_dims=hidden_dims, dropout_rate=dropout_rate)
    return model


if __name__ == "__main__":
    test_model = create_model(seed=42)
    params, size_bytes = test_model.count_parameters()
    print(f"IntrusionMLP Architecture:\n{test_model}")
    print(f"Total Trainable Parameters: {params:,} ({size_bytes / 1024:.2f} KB)")
    
    # Test forward pass with dummy batch
    dummy_x = torch.randn(8, 122)
    dummy_out = test_model(dummy_x)
    dummy_prob = test_model.predict_proba(dummy_x)
    print(f"Forward output shape: {dummy_out.shape}, Probs shape: {dummy_prob.shape}")
    assert dummy_out.shape == (8, 1), "Output shape mismatch!"
    assert (dummy_prob >= 0.0).all() and (dummy_prob <= 1.0).all(), "Invalid probability range!"
    print("Self-test passed successfully.")
