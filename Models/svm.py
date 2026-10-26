"""
Support Vector Machine (SVM) Model Plugin
Configurable: kernel (rbf/linear/poly), C (regularization parameter)
"""

import time
from sklearn.svm import SVC, LinearSVC
from Models.base_model import BaseModel, save_evaluation_results


class SVMModel(BaseModel):
    MODEL_NAME = "SVM (Support Vector Machine)"
    DESCRIPTION = "Maximum-margin hyperplane classifier"

    def train(self, data: dict, output_dir: str, kernel: str = "rbf", C: float = 1.0, **kwargs) -> float:
        X_train, y_train = data["X_train"], data["y_train"]
        X_test, y_test = data["X_test"], data["y_test"]
        dataset_name = data["name"]
        class_names = data["class_names"]

        kernel_choice = kernel.lower() if kernel.lower() in ["rbf", "linear", "poly", "sigmoid"] else "rbf"
        C_val = max(0.001, float(C))

        is_huge = (len(X_train) > 8000 or data.get("is_image", False)) and kernel_choice == "linear"

        if is_huge:
            print(f"[*] Hyperparameters: LinearSVC | C={C_val}", flush=True)
            model = LinearSVC(C=C_val, random_state=42, max_iter=2500)
        else:
            print(f"[*] Hyperparameters: SVC | kernel='{kernel_choice}' | C={C_val}", flush=True)
            model = SVC(kernel=kernel_choice, C=C_val, random_state=42)

        t0 = time.time()
        model.fit(X_train, y_train)
        print(f"[*] Training finished in {time.time() - t0:.2f}s. Evaluating test predictions...", flush=True)

        y_pred = model.predict(X_test)

        acc = save_evaluation_results(
            output_dir=output_dir,
            dataset_name=dataset_name,
            model_name="svm",
            y_true=y_test,
            y_pred=y_pred,
            class_names=class_names,
            model_obj=model,
            save_filename="svm_model.joblib"
        )
        return acc


model_instance = SVMModel()
