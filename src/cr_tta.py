import torch
import torch.nn.functional as F
from tqdm import tqdm

from model import corrupt
from tent import configure_model_for_tta


def cr_tta_adapt_and_predict(model, ood_loader, device, lr=1e-4, mask_ratio=0.3):
    """
    CR-TTA: reconstruction loss TTA, updating BN γ/β only.
    No label signal used — reconstruction depends only on feature distribution P(X).
    Returns predictions (numpy int array) for the full OOD set.
    y in ood_loader is never used here — only X is consumed.
    """
    configure_model_for_tta(model)
    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad], lr=lr
    )

    all_preds = []
    for X, _ in tqdm(ood_loader, desc="CR-TTA", leave=False):
        X = X.to(device)

        X_tilde, m = corrupt(X, mask_ratio)
        h_tilde = model.encoder(X_tilde)
        X_hat = model.reconstructor(h_tilde)
        recon_loss = F.mse_loss(X_hat * m, X * m)
        optimizer.zero_grad()
        recon_loss.backward()
        optimizer.step()

        with torch.no_grad():
            preds = model(X).argmax(dim=1)
        all_preds.append(preds.cpu())

    return torch.cat(all_preds).numpy()
