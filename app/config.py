"""
Configuration module for managing dataset selection (ASL/ISL) and model paths.
"""
import os

# Available datasets
DATASETS = ["ASL", "ISL"]

# Current active dataset (default to ASL)
_current_dataset = "ASL"

# Base paths
BASE_DATASET_DIR = "dataset"
BASE_PKL_DIR = "pkl"  # Legacy fallback


def get_dataset_dir(dataset: str = None) -> str:
    """Get the dataset directory for the given dataset."""
    if dataset is None:
        dataset = _current_dataset
    return os.path.join(BASE_DATASET_DIR, dataset)


def get_pkl_dir(dataset: str = None) -> str:
    """Get the pkl directory for the given dataset."""
    return get_dataset_dir(dataset)


def get_model_paths(dataset: str = None):
    """Get the model file paths for the given dataset."""
    pkl_dir = get_pkl_dir(dataset)
    paths = {
        "sign_dataset": os.path.join(pkl_dir, "sign_dataset.pkl"),
        "sign_model": os.path.join(pkl_dir, "sign_language_model.pkl"),
        "motion_dataset": os.path.join(pkl_dir, "motion_dataset.pkl"),
        "motion_model": os.path.join(pkl_dir, "motion_model.pkl"),
    }
    return paths


def get_current_dataset() -> str:
    """Get the currently active dataset."""
    return _current_dataset


def set_current_dataset(dataset: str):
    """Set the currently active dataset."""
    global _current_dataset
    if dataset in DATASETS:
        _current_dataset = dataset
        # Ensure the dataset directory exists
        os.makedirs(get_pkl_dir(dataset), exist_ok=True)
    else:
        raise ValueError(f"Unknown dataset: {dataset}. Must be one of {DATASETS}")


def ensure_dataset_dirs():
    """Ensure all dataset directories exist."""
    for dataset in DATASETS:
        os.makedirs(get_pkl_dir(dataset), exist_ok=True)