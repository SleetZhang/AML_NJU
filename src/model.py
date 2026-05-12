import torch
import torch.nn as nn


class CR_MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, num_classes=2, dropout=0.1):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Linear(hidden_dim, num_classes)
        self.reconstructor = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        h = self.encoder(x)
        return self.classifier(h)


def corrupt(X, mask_ratio=0.3):
    """Replace mask_ratio fraction of each row with values from a random row in the same batch (same column)."""
    m = (torch.rand_like(X) < mask_ratio).float()
    idx = torch.randint(0, X.size(0), (X.size(0),), device=X.device)
    X_random = X[idx]
    X_tilde = X * (1 - m) + X_random * m
    return X_tilde, m
