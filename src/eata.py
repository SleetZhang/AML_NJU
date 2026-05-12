import math
import torch
import torch.nn.functional as F

from tent import configure_model_for_tta

# For binary classification, max entropy = ln(2) ≈ 0.693
ENTROPY_THRESHOLD = 0.4 * math.log(2)  # ≈ 0.277


def eata_adapt_and_predict(model, ood_loader, device, lr=1e-4, reg_coef=0.001):
    """
    EATA: TENT + high-entropy sample filtering + anti-forgetting L2 regularization.
    Returns predictions (numpy int array) for the full OOD set.
    y in ood_loader is never used here — only X is consumed.
    """
    configure_model_for_tta(model)

    # Save initial BN params as anchor for anti-forgetting regularization
    anchor_params = {
        name: param.data.clone()
        for name, param in model.named_parameters()
        if param.requires_grad
    }

    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad], lr=lr
    )

    all_preds = []
    for X, _ in ood_loader:
        X = X.to(device)

        # Compute per-sample entropy
        with torch.no_grad():
            logits = model(X)
        probs = F.softmax(logits, dim=1)
        sample_entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=1)

        # Filter: keep low-entropy (reliable) samples for the update
        mask = sample_entropy < ENTROPY_THRESHOLD
        if mask.sum() > 1:  # BN in train mode requires at least 2 samples
            logits_filtered = model(X[mask])
            probs_filtered = F.softmax(logits_filtered, dim=1)
            entropy_loss = -(probs_filtered * torch.log(probs_filtered + 1e-8)).sum(dim=1).mean()

            # Anti-forgetting: L2 penalty on BN params relative to source-trained values
            reg_loss = sum(
                ((param - anchor_params[name]) ** 2).sum()
                for name, param in model.named_parameters()
                if param.requires_grad
            )

            loss = entropy_loss + reg_coef * reg_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Predict after adaptation
        with torch.no_grad():
            preds = model(X).argmax(dim=1)
        all_preds.append(preds.cpu())

    return torch.cat(all_preds).numpy()
