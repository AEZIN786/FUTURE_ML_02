import os
import joblib
import logging
from typing import Any

def setup_logging() -> None:
    """Configures the logging format and level for the application."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

def get_project_root() -> str:
    """Returns the absolute path to the project root directory."""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def get_data_path() -> str:
    """Returns the absolute path to the dataset CSV file."""
    return os.path.join(get_project_root(), 'dataset', 'support_tickets.csv')

def get_models_dir() -> str:
    """Returns the absolute path to the models directory and creates it if it doesn't exist."""
    models_dir = os.path.join(get_project_root(), 'models')
    os.makedirs(models_dir, exist_ok=True)
    return models_dir

def get_screenshots_dir() -> str:
    """Returns the absolute path to the screenshots directory and creates it if it doesn't exist."""
    screenshots_dir = os.path.join(get_project_root(), 'screenshots')
    os.makedirs(screenshots_dir, exist_ok=True)
    return screenshots_dir

def save_model(model: Any, filename: str) -> None:
    """
    Saves a model or vectorizer to the models directory using joblib.
    """
    filepath = os.path.join(get_models_dir(), filename)
    joblib.dump(model, filepath)
    logging.info(f"Model saved to {filepath}")

def load_model(filename: str) -> Any:
    """
    Loads a model or vectorizer from the models directory using joblib.
    """
    filepath = os.path.join(get_models_dir(), filename)
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Model file not found at {filepath}. Please train the models first.")
    model = joblib.load(filepath)
    logging.info(f"Model loaded from {filepath}")
    return model