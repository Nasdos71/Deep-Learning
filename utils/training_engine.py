"""
Training Engine & Pipeline Coordinator
Connects a discovered dataset with a selected model plugin, parses flexible
hyperparameters (split %, random seed, algorithm settings), and executes training.
"""

import sys
import os
import argparse

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils.dataset_loader import discover_datasets, load_dataset
from utils.model_discovery import discover_models, load_model_instance


def run_pipeline(
    dataset_key: str,
    model_key: str,
    test_size: float = 0.2,
    random_state: int = 42,
    batch_size: int = 32,
    **kwargs
):
    print("=" * 65, flush=True)
    print(f"  TRAINING ENGINE -> [DATASET: {dataset_key.upper()}] | [MODEL: {model_key.upper()}]", flush=True)
    print("=" * 65, flush=True)

    # 1. Discover and Match Dataset
    datasets_dir = os.path.join(PROJECT_ROOT, "Data sets")
    available_datasets = discover_datasets(datasets_dir)

    selected_ds_info = None
    for disp_name, info in available_datasets.items():
        if info["id"] == dataset_key.lower() or dataset_key.lower() in disp_name.lower():
            selected_ds_info = info
            break

    if not selected_ds_info:
        print(f"[ERROR] Dataset '{dataset_key}' not found. Available datasets: {[v['id'] for v in available_datasets.values()]}", flush=True)
        sys.exit(1)

    # 2. Load and Validate Dataset with custom split & seed
    data = load_dataset(
        selected_ds_info,
        test_size=test_size,
        random_state=random_state,
        batch_size=batch_size
    )
    print(f"[*] Loaded dataset: {data['name']} ({len(data['class_names'])} classes, {data['input_dim']} features)", flush=True)

    # 3. Load Model Plugin from Models/
    models_dir = os.path.join(PROJECT_ROOT, "Models")
    try:
        model = load_model_instance(model_key.lower(), models_dir=models_dir)
        print(f"[*] Loaded model plugin: {getattr(model, 'MODEL_NAME', model_key)}", flush=True)
    except Exception as e:
        print(f"[ERROR] Could not load model '{model_key}': {e}", flush=True)
        sys.exit(1)

    # 4. Prepare Output Directory
    output_dir = os.path.join(PROJECT_ROOT, "output", data["name"], model_key.lower())
    os.makedirs(output_dir, exist_ok=True)
    print(f"[*] Target output directory: {output_dir}", flush=True)

    # 5. Execute Training with flexible kwargs
    acc = model.train(data=data, output_dir=output_dir, **kwargs)
    print(f"\n[SUMMARY] Successfully finished training {model_key.upper()} on {data['name'].upper()} with {acc*100:.2f}% accuracy.", flush=True)
    return acc


def main():
    parser = argparse.ArgumentParser(description="Deep Learning Studio Flexible Training Engine")

    # Dataset & Splitting parameters
    parser.add_argument('--dataset', type=str, required=True, help="Dataset ID (e.g. iris, mnist)")
    parser.add_argument('--model', type=str, required=True, help="Model ID (e.g. knn, svm, ann)")
    parser.add_argument('--test_size', type=float, default=0.20, help="Test set fraction (e.g. 0.20 for 20%)")
    parser.add_argument('--random_state', type=int, default=42, help="Random seed for reproducible splitting")
    parser.add_argument('--batch_size', type=int, default=32, help="Batch size for loaders")

    # KNN parameters
    parser.add_argument('--k', type=int, default=5, help="Number of neighbors for KNN")
    parser.add_argument('--knn_weights', type=str, default='distance', choices=['distance', 'uniform'])
    parser.add_argument('--knn_metric', type=str, default='euclidean', choices=['euclidean', 'manhattan', 'minkowski'])

    # SVM parameters
    parser.add_argument('--svm_kernel', type=str, default='rbf', choices=['rbf', 'linear', 'poly'])
    parser.add_argument('--svm_c', type=float, default=1.0, help="Regularization parameter C")

    # ANN parameters
    parser.add_argument('--epochs', type=int, default=None, help="Epoch count for neural models")
    parser.add_argument('--lr', type=float, default=0.001, help="Learning rate for neural models")
    parser.add_argument('--optimizer', type=str, default='adamw', choices=['adamw', 'adam', 'sgd'])

    args = parser.parse_args()

    # Model specific kwargs
    model_kwargs = {
        "k": args.k,
        "weights": args.knn_weights,
        "metric": args.knn_metric,
        "kernel": args.svm_kernel,
        "C": args.svm_c,
        "lr": args.lr,
        "optimizer_name": args.optimizer
    }
    if args.epochs is not None:
        model_kwargs["epochs"] = args.epochs

    run_pipeline(
        dataset_key=args.dataset,
        model_key=args.model,
        test_size=args.test_size,
        random_state=args.random_state,
        batch_size=args.batch_size,
        **model_kwargs
    )


if __name__ == '__main__':
    main()
