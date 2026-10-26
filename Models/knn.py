"""
K-Nearest Neighbors (KNN) Model Plugin
Configurable: k_neighbors, weights (uniform/distance), metric (euclidean/manhattan)
"""

import time
from sklearn.neighbors import KNeighborsClassifier
from Models.base_model import BaseModel, save_evaluation_results


class KNNModel(BaseModel):
    MODEL_NAME = "KNN (K-Nearest Neighbors)"
    DESCRIPTION = "Non-parametric instance-based classifier"

    def train(self, data: dict, output_dir: str, k: int = 5, weights: str = "distance", metric: str = "euclidean", **kwargs) -> float:
        X_train, y_train = data["X_train"], data["y_train"]
        X_test, y_test = data["X_test"], data["y_test"]
        dataset_name = data["name"]
        class_names = data["class_names"]

        k_valid = max(1, min(k, len(X_train) - 1))
        weights_valid = weights.lower() if weights.lower() in ["uniform", "distance"] else "distance"
        metric_valid = metric.lower() if metric.lower() in ["euclidean", "manhattan", "minkowski"] else "euclidean"

        print(f"[*] Hyperparameters: k={k_valid} | weights='{weights_valid}' | metric='{metric_valid}'", flush=True)

        model = KNeighborsClassifier(n_neighbors=k_valid, weights=weights_valid, metric=metric_valid, n_jobs=-1)
        t0 = time.time()
        model.fit(X_train, y_train)
        print(f"[*] Training finished in {time.time() - t0:.2f}s. Evaluating test predictions...", flush=True)

        y_pred = model.predict(X_test)

        acc = save_evaluation_results(
            output_dir=output_dir,
            dataset_name=dataset_name,
            model_name="knn",
            y_true=y_test,
            y_pred=y_pred,
            class_names=class_names,
            model_obj=model,
            save_filename="knn_model.joblib"
        )
        return acc


model_instance = KNNModel()
