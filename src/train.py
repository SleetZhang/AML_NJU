import copy
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import balanced_accuracy_score
from tqdm import tqdm

from model import corrupt


def train(model, train_loader, val_loader, device,
          lr=1e-3, weight_decay=1e-4, max_epochs=200,
          patience=20, lam=0.5, mask_ratio=0.3):
    """
    Source-domain training: CE loss + λ * reconstruction loss.
    Early stopping on val Balanced Accuracy (higher is better).
    Returns the model loaded with the best checkpoint.
    """
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_val_bal_acc = -1.0
    best_state = copy.deepcopy(model.state_dict())
    patience_counter = 0

    for epoch in range(1, max_epochs + 1):
        # --- Training ---
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch:3d} [train]", leave=False)
        for X, y in pbar:
            X, y = X.to(device), y.to(device)

            logits = model(X)
            ce_loss = F.cross_entropy(logits, y)

            X_tilde, m = corrupt(X, mask_ratio)
            h_tilde = model.encoder(X_tilde)
            X_hat = model.reconstructor(h_tilde)
            recon_loss = F.mse_loss(X_hat * m, X * m)

            loss = ce_loss + lam * recon_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        # --- Validation: Balanced Accuracy ---
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for X_v, y_v in val_loader:
                preds = model(X_v.to(device)).argmax(dim=1).cpu()
                all_preds.append(preds)
                all_labels.append(y_v)
        val_bal_acc = balanced_accuracy_score(
            torch.cat(all_labels).numpy(),
            torch.cat(all_preds).numpy()
        )

        print(f"Epoch {epoch:3d} | val_BalAcc={val_bal_acc:.4f}", flush=True)

        if val_bal_acc > best_val_bal_acc:
            best_val_bal_acc = val_bal_acc
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}. Best val_BalAcc={best_val_bal_acc:.4f}")
                break

    model.load_state_dict(best_state)
    return model
