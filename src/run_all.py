import argparse
import copy
import os
import numpy as np
import pandas as pd
import torch

from data_loader import load_dataset
from model import CR_MLP
from train import train
from evaluate import compute_metrics
from tent import tent_adapt_and_predict
from eata import eata_adapt_and_predict
from cr_tta import cr_tta_adapt_and_predict

DATA_ROOT = os.path.join(os.path.dirname(__file__), "..", "TableShift_Dataset")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")


def run_one_seed(dataset_name, seed, device):
    torch.manual_seed(seed)
    np.random.seed(seed)

    print(f"\n{'='*60}")
    print(f"Dataset={dataset_name}  Seed={seed}")
    print(f"{'='*60}")

    # Load data
    train_loader, val_loader, idtest_loader, ood_loader, _ = load_dataset(
        DATA_ROOT, dataset_name
    )
    input_dim = next(iter(train_loader))[0].shape[1]

    # Train
    model = CR_MLP(input_dim=input_dim).to(device)
    model = train(model, train_loader, val_loader, device)

    # Save checkpoint
    ckpt_path = os.path.join(RESULTS_DIR, f"cr_mlp_{dataset_name}_seed{seed}.pt")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    torch.save(model.state_dict(), ckpt_path)

    results = []

    # --- ERM (no adaptation) ---
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for X, y in idtest_loader:
            preds = model(X.to(device)).argmax(dim=1).cpu()
            all_preds.append(preds)
            all_labels.append(y)
    id_metrics = compute_metrics(
        torch.cat(all_labels).numpy(), torch.cat(all_preds).numpy()
    )

    all_preds, all_labels = [], []
    with torch.no_grad():
        for X, y in ood_loader:
            preds = model(X.to(device)).argmax(dim=1).cpu()
            all_preds.append(preds)
            all_labels.append(y)
    y_ood_true = torch.cat(all_labels).numpy()
    ood_preds_erm = torch.cat(all_preds).numpy()
    ood_metrics = compute_metrics(y_ood_true, ood_preds_erm)

    results.append({
        "dataset": dataset_name, "seed": seed, "method": "ERM",
        "id_acc": id_metrics["acc"],
        "id_bal_acc": id_metrics["bal_acc"],
        "id_f1": id_metrics["f1"],
        "ood_acc": ood_metrics["acc"],
        "ood_bal_acc": ood_metrics["bal_acc"],
        "ood_f1": ood_metrics["f1"],
    })
    print(f"ERM       | ID Acc={id_metrics['acc']:.4f} BalAcc={id_metrics['bal_acc']:.4f} F1={id_metrics['f1']:.4f}"
          f" | OOD Acc={ood_metrics['acc']:.4f} BalAcc={ood_metrics['bal_acc']:.4f} F1={ood_metrics['f1']:.4f}")

    # --- TTA methods: each reloads the checkpoint ---
    tta_methods = {
        "TENT":   tent_adapt_and_predict,
        "EATA":   eata_adapt_and_predict,
        "CR-TTA": cr_tta_adapt_and_predict,
    }
    for method_name, adapt_fn in tta_methods.items():
        # Fresh model from checkpoint
        tta_model = CR_MLP(input_dim=input_dim).to(device)
        tta_model.load_state_dict(torch.load(ckpt_path, map_location=device))

        y_pred = adapt_fn(tta_model, ood_loader, device)
        metrics = compute_metrics(y_ood_true, y_pred)

        results.append({
            "dataset": dataset_name, "seed": seed, "method": method_name,
            "id_acc": None,
            "ood_acc": metrics["acc"],
            "ood_bal_acc": metrics["bal_acc"],
            "ood_f1": metrics["f1"],
        })
        print(f"{method_name:<9} | OOD Acc={metrics['acc']:.4f} "
              f"BalAcc={metrics['bal_acc']:.4f} F1={metrics['f1']:.4f}")

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+",
                        default=["nhanes_lead"],
                        help="Dataset names to run")
    parser.add_argument("--seeds", nargs="+", type=int,
                        default=[0],
                        help="Random seeds")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    all_results = []
    for dataset in args.datasets:
        for seed in args.seeds:
            all_results.extend(run_one_seed(dataset, seed, device))

    df_new = pd.DataFrame(all_results)
    out_path = os.path.join(RESULTS_DIR, "results.csv")
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Merge with existing results so separate runs don't overwrite each other
    if os.path.exists(out_path):
        df_old = pd.read_csv(out_path)
        df = pd.concat([df_old, df_new], ignore_index=True)
        df = df.drop_duplicates(subset=["dataset", "seed", "method"], keep="last")
        df = df.sort_values(["dataset", "method", "seed"]).reset_index(drop=True)
    else:
        df = df_new

    df.to_csv(out_path, index=False)
    print(f"\nResults saved to {out_path}")

    # Print summary: mean±std across seeds
    print("\n--- Summary (mean ± std across seeds) ---")
    for (dataset, method), grp in df.groupby(["dataset", "method"]):
        ood_acc = grp["ood_acc"].dropna()
        ood_bal = grp["ood_bal_acc"].dropna()
        ood_f1  = grp["ood_f1"].dropna()

        if method == "ERM":
            id_acc = grp["id_acc"].dropna()
            id_bal = grp["id_bal_acc"].dropna()
            id_f1  = grp["id_f1"].dropna()
            gap_acc = id_acc.mean() - ood_acc.mean()
            gap_bal = id_bal.mean() - ood_bal.mean() if len(id_bal) > 0 else float("nan")
            gap_f1  = id_f1.mean()  - ood_f1.mean()  if len(id_f1)  > 0 else float("nan")
            print(f"{dataset} | {method:<9} | "
                  f"ID  Acc={id_acc.mean():.4f}±{id_acc.std():.4f}  "
                  f"BalAcc={id_bal.mean():.4f}±{id_bal.std():.4f}  "
                  f"F1={id_f1.mean():.4f}±{id_f1.std():.4f}")
            print(f"{dataset} | {method:<9} | "
                  f"OOD Acc={ood_acc.mean():.4f}±{ood_acc.std():.4f}  "
                  f"BalAcc={ood_bal.mean():.4f}±{ood_bal.std():.4f}  "
                  f"F1={ood_f1.mean():.4f}±{ood_f1.std():.4f}")
            print(f"{dataset} | {method:<9} | "
                  f"Gap Acc={gap_acc:+.4f}              "
                  f"BalAcc={gap_bal:+.4f}              "
                  f"F1={gap_f1:+.4f}")
        else:
            print(f"{dataset} | {method:<9} | "
                  f"OOD Acc={ood_acc.mean():.4f}±{ood_acc.std():.4f}  "
                  f"BalAcc={ood_bal.mean():.4f}±{ood_bal.std():.4f}  "
                  f"F1={ood_f1.mean():.4f}±{ood_f1.std():.4f}")


if __name__ == "__main__":
    main()
