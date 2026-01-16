# core/utils.py
import numpy as np

def detect_problem_type(y):
    """Simple heuristic: if unique values less than threshold, treat as classification."""
    try:
        if y.dtype == object:
            return "classification"
        unique = np.unique(y)
        return "classification" if len(unique) < 20 else "regression"
    except Exception:
        return "regression"
