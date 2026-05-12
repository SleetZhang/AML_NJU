import torch
import torch.nn.functional as F


def configure_model_for_tta(model):
    """
    Freeze all parameters, then unfreeze BN affine params (γ, β).
    Set BN layers to train mode so they use batch statistics (not running stats).
    All other layers stay in eval mode (Dropout off).
    """
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    for module in model.modules():
        if isinstance(module, torch.nn.BatchNorm1d):
            module.train()
            module.weight.requires_grad = True  # γ
            module.bias.requires_grad = True    # β
    return model


def tent_adapt_and_predict(model, ood_loader, device, lr=1e-4):
    """
    TENT: entropy minimization on OOD batches, updating BN γ/β only.
    Returns predictions (numpy int array) for the full OOD set.
    y in ood_loader is never used here — only X is consumed.
    """
    configure_model_for_tta(model)
    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad], lr=lr
    )

    all_preds = []
    for X, _ in ood_loader:
        X = X.to(device)

        # One gradient step per batch
        logits = model(X)
        probs = F.softmax(logits, dim=1)
        entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=1).mean()
        optimizer.zero_grad()
        entropy.backward()
        optimizer.step()

        # Predict after adaptation
        with torch.no_grad():
            preds = model(X).argmax(dim=1)
        all_preds.append(preds.cpu())

    return torch.cat(all_preds).numpy()
