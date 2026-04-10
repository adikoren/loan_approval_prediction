# src/__init__.py — Clean shortcut imports for external callers.
# No logic here — only re-exports so that callers can write:
#   from src import build_pipeline
# instead of:
#   from src.model import build_pipeline

from src.preprocessing import run_all_preprocessing
from src.features import run_all_feature_engineering, get_final_feature_columns
from src.model import build_pipeline, save_model, load_model
from src.train import train
from src.evaluate import evaluate
from src.predict import predict

__all__ = [
    "run_all_preprocessing",
    "run_all_feature_engineering",
    "get_final_feature_columns",
    "build_pipeline",
    "save_model",
    "load_model",
    "train",
    "evaluate",
    "predict",
]
