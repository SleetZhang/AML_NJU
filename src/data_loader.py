import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler


def load_dataset(data_root, dataset_name, batch_size=512):
    """
    Returns train/val/idtest/ood DataLoaders and binary_feature_mask.
    Training loader uses WeightedRandomSampler to balance classes.

    binary_feature_mask: bool array of shape (input_dim,)
        True for columns whose training values are all in {0, 1}.
    """
    def read(split, kind):
        path = f"{data_root}/{dataset_name}/{dataset_name}_{kind}{split}.csv"
        return pd.read_csv(path).values.astype(np.float32)

    X_train = read("train", "X")
    y_train = read("train", "y").ravel().astype(np.int64)
    X_val   = read("val",   "X")
    y_val   = read("val",   "y").ravel().astype(np.int64)
    X_id    = read("idtest","X")
    y_id    = read("idtest","y").ravel().astype(np.int64)
    X_ood   = read("ood",   "X")
    y_ood   = read("ood",   "y").ravel().astype(np.int64)

    # Detect binary columns from training data only
    binary_feature_mask = np.all(np.isin(X_train, [0.0, 1.0]), axis=0)

    # WeightedRandomSampler: minority class drawn proportionally more often
    # Each positive sample gets weight = n_neg/n_pos, each negative gets weight = 1
    # Result: each batch is ~50% positive / 50% negative on average
    n_pos = y_train.sum()
    n_neg = len(y_train) - n_pos
    sample_weights = np.where(y_train == 1, n_neg / n_pos, 1.0)
    sampler = WeightedRandomSampler(
        weights=torch.from_numpy(sample_weights).float(),
        num_samples=len(y_train),
        replacement=True,
    )

    def make_loader(X, y, shuffle, sampler=None):
        ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                          sampler=sampler, drop_last=False)

    train_loader  = make_loader(X_train, y_train, shuffle=False, sampler=sampler)
    val_loader    = make_loader(X_val,   y_val,   shuffle=False)
    idtest_loader = make_loader(X_id,    y_id,    shuffle=False)
    ood_loader    = make_loader(X_ood,   y_ood,   shuffle=False)

    return train_loader, val_loader, idtest_loader, ood_loader, binary_feature_mask
