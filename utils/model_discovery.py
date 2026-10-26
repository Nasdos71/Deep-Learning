"""
Model Plugin Discovery Utility
Scans the Models/ directory and auto-discovers all model implementation files.
"""

import os
import glob
import importlib.util

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
DEFAULT_MODELS_DIR = os.path.join(PROJECT_ROOT, "Models")

IGNORED_FILES = {"__init__.py", "base_model.py"}


def discover_models(models_dir=None):
    """
    Scans models_dir for Python files and returns a dictionary of available models:
    { display_name: { "id": model_id, "file": filename, "filepath": path } }
    """
    if models_dir is None:
        models_dir = DEFAULT_MODELS_DIR

    models = {}
    py_files = glob.glob(os.path.join(models_dir, "*.py"))

    for filepath in py_files:
        filename = os.path.basename(filepath)
        if filename in IGNORED_FILES:
            continue

        model_id = os.path.splitext(filename)[0]
        display_name = f"{model_id.upper()} Model"

        try:
            spec = importlib.util.spec_from_file_location(model_id, filepath)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            if hasattr(module, "model_instance") and hasattr(module.model_instance, "MODEL_NAME"):
                display_name = module.model_instance.MODEL_NAME
            elif hasattr(module, "MODEL_NAME"):
                display_name = module.MODEL_NAME
        except Exception:
            display_name = f"{model_id.upper()} (Plugin)"

        models[display_name] = {
            "id": model_id,
            "filename": filename,
            "filepath": filepath
        }

    return models


def load_model_instance(model_id, models_dir=None):
    """Loads and returns the model_instance from Models/<model_id>.py."""
    if models_dir is None:
        models_dir = DEFAULT_MODELS_DIR

    filepath = os.path.join(models_dir, f"{model_id}.py")
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Model file not found: {filepath}")

    spec = importlib.util.spec_from_file_location(model_id, filepath)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    if hasattr(module, "model_instance"):
        return module.model_instance
    elif hasattr(module, "train"):
        class FunctionModelWrapper:
            def train(self, data, output_dir, **kwargs):
                return module.train(data, output_dir, **kwargs)
        return FunctionModelWrapper()
    else:
        raise AttributeError(f"Module '{model_id}.py' must define 'model_instance' or a 'train' function.")
