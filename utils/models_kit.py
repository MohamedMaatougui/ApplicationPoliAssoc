# utils/model_kit.py
import joblib
import pandas as pd
from pathlib import Path
import re

MODEL_ROOT = Path(__file__).resolve().parent.parent / "models"

# ------------------------------------------------------------------
# Build a dict  model_id -> {model: obj, features: obj, path: Path}
# ------------------------------------------------------------------
def _is_model_file(p: Path) -> bool:
    name = p.name.lower()
    return p.is_file() and name.endswith(('.pkl', '.joblib')) and (
        'model' in name or 'regressor' in name or 'classifier' in name
    )

def _find_partner_feature_obj(model_path: Path) -> Path | None:
    """
    Heuristic: look for a sibling file that contains 'feature' in the name.
    Example:  xgboost_model.pkl   ->  xgboost_features.pkl
              knn_model.pkl       ->  knn_features.pkl
    """
    parent = model_path.parent
    base = model_path.stem.split('_model')[0]  # strip suffix
    for cand in parent.glob(f'*feature*.pkl'):
        return cand
    # fallback: any feature pkl in same folder
    for cand in parent.glob('*feature*.pkl'):
        return cand
    return None

def build_model_registry() -> dict[str, dict]:
    registry = {}
    for model_path in MODEL_ROOT.rglob('*'):
        if not _is_model_file(model_path):
            continue
        model_id = re.sub(r'\s+', '_', model_path.stem)  # sanitize spaces
        feature_path = _find_partner_feature_obj(model_path)
        registry[model_id] = {
            'model': joblib.load(model_path),
            'features': joblib.load(feature_path) if feature_path else None,
            'path': model_path
        }
    return registry

MODELS = build_model_registry()   # imported by app.py