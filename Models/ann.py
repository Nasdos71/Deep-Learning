"""
Artificial Neural Network (ANN) PyTorch Model Plugin
Configurable: epochs, learning rate, optimizer (AdamW, Adam, SGD).
Runs on NVIDIA GPU (RTX 5060) with automatic architecture scaling.
"""

import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from Models.base_model import BaseModel, save_evaluation_results


class DynamicMLP(nn.Module):
    """Deep Neural Network that dynamically adapts architecture to dataset complexity."""
    def __init__(self, input_dim: int, num_classes: int):
        super(DynamicMLP, self).__init__()

        if input_dim >= 100:
            h1, h2 = 256, 128
            self.network = nn.Sequential(
                nn.Linear(input_dim, h1),
                nn.BatchNorm1d(h1),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(h1, h2),
                nn.BatchNorm1d(h2),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(h2, num_classes)
            )
        else:
            h1 = max(32, input_dim * 4)
            h2 = max(16, input_dim * 2)
            self.network = nn.Sequential(
                nn.Linear(input_dim, h1),
                nn.ReLU(),
                nn.Linear(h1, h2),
                nn.ReLU(),
                nn.Linear(h2, num_classes)
            )

    def forward(self, x):
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        return self.network(x)


class ANNModel(BaseModel):
    MODEL_NAME = "ANN (Artificial Neural Network - GPU)"
    DESCRIPTION = "Multi-layer perceptron running on PyTorch CUDA GPU"

    def train(self, data: dict, output_dir: str, epochs: int = 50, lr: float = 0.001, optimizer_name: str = "adamw", **kwargs) -> float:
        input_dim = data["input_dim"]
        num_classes = data["num_classes"]
        dataset_name = data["name"]
        class_names = data["class_names"]
        train_loader = data["train_loader"]
        test_loader = data["test_loader"]

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dev_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
        opt_choice = optimizer_name.lower()

        print(f"[*] Hyperparameters: Epochs={epochs} | LR={lr} | Optimizer='{opt_choice.upper()}'", flush=True)
        print(f"[*] Hardware: {device} ({dev_name}) | Input Dim: {input_dim} -> Classes: {num_classes}", flush=True)

        model = DynamicMLP(input_dim=input_dim, num_classes=num_classes).to(device)
        criterion = nn.CrossEntropyLoss()

        if opt_choice == "sgd":
            optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
        elif opt_choice == "adam":
            optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
        else:
            optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

        t_start = time.time()
        for ep in range(1, epochs + 1):
            model.train()
            running_loss = 0.0
            correct = 0
            total = 0
            t_ep = time.time()

            for bx, by in train_loader:
                bx, by = bx.to(device), by.to(device)
                optimizer.zero_grad()
                out = model(bx)
                loss = criterion(out, by)
                loss.backward()
                optimizer.step()

                running_loss += loss.item() * bx.size(0)
                preds = out.argmax(dim=1)
                total += by.size(0)
                correct += (preds == by).sum().item()

            ep_loss = running_loss / total
            ep_acc = (correct / total) * 100
            ep_time = time.time() - t_ep

            if epochs <= 10 or ep % max(1, epochs // 10) == 0 or ep == epochs:
                print(f"   [Epoch {ep:2d}/{epochs:2d}] Loss: {ep_loss:.4f} | Train Acc: {ep_acc:.2f}% ({ep_time:.2f}s)", flush=True)

        print(f"[*] Training finished in {time.time() - t_start:.2f}s. Evaluating test dataset...", flush=True)

        model.eval()
        all_preds, all_targets = [], []
        with torch.no_grad():
            for bx, by in test_loader:
                bx = bx.to(device)
                preds = model(bx).argmax(dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(by.numpy())

        y_pred = np.array(all_preds)
        y_true = np.array(all_targets)

        acc = save_evaluation_results(
            output_dir=output_dir,
            dataset_name=dataset_name,
            model_name="ann",
            y_true=y_true,
            y_pred=y_pred,
            class_names=class_names,
            model_obj=model,
            save_filename="ann_model.pt"
        )
        return acc


model_instance = ANNModel()
