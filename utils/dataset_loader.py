"""
Dynamic Dataset Loader & Validator
Auto-discovers datasets in the 'Data sets' folder (CSVs, MNIST, image folders),
validates data integrity, and produces standardized, configurable train/test splits.
"""

import os
import glob
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from torchvision import datasets, transforms
import torch
from torch.utils.data import TensorDataset, DataLoader, ConcatDataset, Subset


def discover_datasets(datasets_dir="Data sets"):
    """
    Scans the datasets_dir and returns a dictionary of available datasets:
    { display_name: { "id": key, "type": "tabular" | "mnist" | "image_folder", "path": file_or_dir_path } }
    """
    discovered = {}

    if not os.path.exists(datasets_dir):
        os.makedirs(datasets_dir, exist_ok=True)
        return discovered

    # 1. Check for CSV files (Tabular datasets)
    csv_files = glob.glob(os.path.join(datasets_dir, "*.csv"))
    for csv_file in csv_files:
        base_name = os.path.splitext(os.path.basename(csv_file))[0]
        display_name = f"{base_name.capitalize()} (CSV Tabular)"
        discovered[display_name] = {
            "id": base_name.lower(),
            "type": "tabular",
            "path": csv_file
        }

    # 2. Check for Subdirectories
    for item in os.listdir(datasets_dir):
        subpath = os.path.join(datasets_dir, item)
        if os.path.isdir(subpath):
            if item.lower() == "mnist":
                discovered["MNIST (Handwritten Digits)"] = {
                    "id": "mnist",
                    "type": "mnist",
                    "path": subpath
                }
            else:
                sub_entries = [d for d in os.listdir(subpath) if os.path.isdir(os.path.join(subpath, d))]
                if len(sub_entries) >= 2:
                    discovered[f"{item.capitalize()} (Image Folder)"] = {
                        "id": item.lower(),
                        "type": "image_folder",
                        "path": subpath
                    }

    return discovered


def load_dataset(dataset_info, test_size=0.2, random_state=42, batch_size=32):
    """
    Loads and validates the specified dataset with custom test_size and random_state.
    Returns a standardized dictionary containing:
      - X_train, y_train, X_test, y_test (numpy arrays)
      - class_names (list of strings)
      - input_dim (int)
      - num_classes (int)
      - is_image (bool)
      - train_loader, test_loader (PyTorch DataLoader objects)
    """
    dtype = dataset_info["type"]
    path = dataset_info["path"]
    test_size = float(test_size)
    random_state = int(random_state)

    # Clamp test_size between 0.05 and 0.50
    test_size = max(0.05, min(0.50, test_size))
    train_pct = int((1.0 - test_size) * 100)
    test_pct = int(test_size * 100)

    print(f"[*] Configuration: Split = {train_pct}% Train / {test_pct}% Test | Seed = {random_state}", flush=True)

    # ---------------- TABULAR CSV DATASET ---------------- #
    if dtype == "tabular":
        print(f"[*] Reading tabular dataset from: {path}", flush=True)
        df = pd.read_csv(path)

        if len(df) < 15:
            raise ValueError(f"Dataset in '{path}' has only {len(df)} rows. Minimum 15 samples required.")

        # Drop ID-like columns
        cols_to_drop = [c for c in df.columns if c.lower() in ['id', 'unnamed: 0', 'index']]
        if cols_to_drop:
            df = df.drop(columns=cols_to_drop)

        # Determine Target Column
        target_candidates = ['species', 'target', 'label', 'class', 'category', 'outcome', 'diagnosis']
        target_col = None
        for cand in target_candidates:
            for c in df.columns:
                if c.lower() == cand:
                    target_col = c
                    break
            if target_col:
                break

        if not target_col:
            target_col = df.columns[-1]

        print(f"[*] Target column: '{target_col}'", flush=True)

        y_raw = df[target_col].values
        X_df = df.drop(columns=[target_col])

        # Impute missing values
        if X_df.isnull().values.any():
            print("[*] Imputing missing values with column medians/modes...", flush=True)
            for c in X_df.columns:
                if pd.api.types.is_numeric_dtype(X_df[c]):
                    X_df[c] = X_df[c].fillna(X_df[c].median())
                else:
                    X_df[c] = X_df[c].fillna(X_df[c].mode()[0] if not X_df[c].mode().empty else "missing")

        # Encode categorical feature columns
        for c in X_df.columns:
            if not pd.api.types.is_numeric_dtype(X_df[c]):
                X_df[c] = LabelEncoder().fit_transform(X_df[c].astype(str))

        X = X_df.values.astype(np.float32)

        # Classification vs Regression check
        unique_targets = np.unique(y_raw)
        if len(unique_targets) > 50 and pd.api.types.is_float_dtype(y_raw):
            raise ValueError(
                f"Target column '{target_col}' has {len(unique_targets)} continuous values. "
                "These models are built for classification. Please provide a discrete categorical target."
            )

        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(y_raw)
        class_names = [str(c) for c in label_encoder.classes_]

        num_classes = len(class_names)
        input_dim = X.shape[1]

        # Configurable Train/Test Split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        print(f"[*] Split summary: {len(X_train)} training samples, {len(X_test)} testing samples.", flush=True)

        # PyTorch DataLoaders
        train_ds = TensorDataset(torch.tensor(X_train_s, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))
        test_ds = TensorDataset(torch.tensor(X_test_s, dtype=torch.float32), torch.tensor(y_test, dtype=torch.long))

        actual_batch = min(batch_size, len(X_train))
        train_loader = DataLoader(train_ds, batch_size=actual_batch, shuffle=True)
        test_loader = DataLoader(test_ds, batch_size=min(64, len(X_test)), shuffle=False)

        return {
            "name": dataset_info["id"],
            "type": "tabular",
            "is_image": False,
            "X_train": X_train_s,
            "y_train": y_train,
            "X_test": X_test_s,
            "y_test": y_test,
            "class_names": class_names,
            "input_dim": input_dim,
            "num_classes": num_classes,
            "train_loader": train_loader,
            "test_loader": test_loader,
            "scaler": scaler,
            "label_encoder": label_encoder
        }

    # ---------------- MNIST DATASET ---------------- #
    elif dtype == "mnist":
        print(f"[*] Loading MNIST dataset from: {path}", flush=True)
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        # torchvision automatically expects root/MNIST/raw
        root_dir = os.path.dirname(path)
        train_dataset = datasets.MNIST(root=root_dir, train=True, download=False, transform=transform)
        test_dataset = datasets.MNIST(root=root_dir, train=False, download=False, transform=transform)

        # Total pool: 70,000 images
        all_ds = ConcatDataset([train_dataset, test_dataset])
        total_len = len(all_ds)
        n_test = int(total_len * test_size)
        n_train = total_len - n_test

        torch.manual_seed(random_state)
        train_sub_ds, test_sub_ds = torch.utils.data.random_split(all_ds, [n_train, n_test])

        print(f"[*] Split summary: {n_train} training samples ({train_pct}%), {n_test} testing samples ({test_pct}%).", flush=True)

        # Extract features for flat models (KNN, SVM)
        # We sample up to 10k train and 2k test to keep execution responsive
        flat_tr_samples = min(10000, n_train)
        flat_te_samples = min(2000, n_test)

        np.random.seed(random_state)
        # Pull flat numpy representations
        all_images = np.concatenate([train_dataset.data.numpy(), test_dataset.data.numpy()]).reshape(-1, 784) / 255.0
        all_targets = np.concatenate([train_dataset.targets.numpy(), test_dataset.targets.numpy()])

        tr_indices = np.random.choice(n_train, size=flat_tr_samples, replace=False)
        te_indices = np.random.choice(range(n_train, total_len), size=flat_te_samples, replace=False)

        X_train_sub = all_images[tr_indices]
        y_train_sub = all_targets[tr_indices]
        X_test_sub = all_images[te_indices]
        y_test_sub = all_targets[te_indices]

        actual_batch = max(32, batch_size)
        train_loader = DataLoader(train_sub_ds, batch_size=actual_batch, shuffle=True, pin_memory=True)
        test_loader = DataLoader(test_sub_ds, batch_size=256, shuffle=False)

        return {
            "name": "mnist",
            "type": "mnist",
            "is_image": True,
            "X_train": X_train_sub,
            "y_train": y_train_sub,
            "X_test": X_test_sub,
            "y_test": y_test_sub,
            "class_names": [str(i) for i in range(10)],
            "input_dim": 784,
            "num_classes": 10,
            "train_loader": train_loader,
            "test_loader": test_loader,
            "scaler": None,
            "label_encoder": None
        }

    # ---------------- IMAGE FOLDER DATASET ---------------- #
    elif dtype == "image_folder":
        print(f"[*] Loading ImageFolder dataset from: {path}", flush=True)
        img_transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        full_ds = datasets.ImageFolder(root=path, transform=img_transform)
        class_names = full_ds.classes

        n_test = int(len(full_ds) * test_size)
        n_train = len(full_ds) - n_test

        torch.manual_seed(random_state)
        train_ds, test_ds = torch.utils.data.random_split(full_ds, [n_train, n_test])

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

        flat_x_tr, flat_y_tr = [], []
        for x_b, y_b in train_loader:
            flat_x_tr.append(x_b.view(x_b.size(0), -1).numpy())
            flat_y_tr.append(y_b.numpy())
        X_train_sub = np.vstack(flat_x_tr)
        y_train_sub = np.concatenate(flat_y_tr)

        flat_x_te, flat_y_te = [], []
        for x_b, y_b in test_loader:
            flat_x_te.append(x_b.view(x_b.size(0), -1).numpy())
            flat_y_te.append(y_b.numpy())
        X_test_sub = np.vstack(flat_x_te)
        y_test_sub = np.concatenate(flat_y_te)

        return {
            "name": dataset_info["id"],
            "type": "image_folder",
            "is_image": True,
            "X_train": X_train_sub,
            "y_train": y_train_sub,
            "X_test": X_test_sub,
            "y_test": y_test_sub,
            "class_names": class_names,
            "input_dim": X_train_sub.shape[1],
            "num_classes": len(class_names),
            "train_loader": train_loader,
            "test_loader": test_loader,
            "scaler": None,
            "label_encoder": None
        }

    else:
        raise ValueError(f"Unsupported dataset type: {dtype}")
