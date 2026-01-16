import os, json, shutil
import pandas as pd
import numpy as np
from typing import Optional, Tuple
from datetime import datetime

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, LabelEncoder
from sklearn.linear_model import SGDRegressor, SGDClassifier
from sklearn.metrics import mean_absolute_error, r2_score, accuracy_score, f1_score

from tpot import TPOTRegressor, TPOTClassifier
from ydata_profiling import ProfileReport
import joblib

from core.data_cleaner import DataCleaner
from core.visualizer import DataVisualizer
from core.utils import detect_problem_type

CHUNK_SIZE = 200_000

class AutoMLEngine:
    """
    AutoML Engine for Auto Data Bot

    Phase 1: TPOT discovers best pipeline on a sample
    Phase 2: SGD trains scalable model on full dataset incrementally
    """

    def __init__(self, memory, projects_dir: str = "projects"):
        self.memory = memory
        self.projects_dir = projects_dir
        os.makedirs(self.projects_dir, exist_ok=True)
        self.cleaner = DataCleaner()

   
    def run_pipeline(self, csv_path: str, target_column: Optional[str], project_name: str) -> Tuple[str, dict]:
        project_path = os.path.join(self.projects_dir, project_name)
        plots_dir = os.path.join(project_path, "plots")
        os.makedirs(plots_dir, exist_ok=True)

        # --------------------------------------------------------
        # SAMPLE FOR TPOT DISCOVERY
        # --------------------------------------------------------
        sample = self._sample_csv(csv_path, n=100_000)

        if target_column is None:
            target_column = sample.columns[-1]

        sample = sample.dropna(subset=[target_column])
        sample_clean, cat_cols = self.cleaner.clean(sample)

        # --------------------------------------------------------
        # PROFILING REPORT
        # --------------------------------------------------------
        profile = ProfileReport(sample_clean, title="Data Profiling Report", minimal=True, explorative=False)
        profile.to_file(os.path.join(project_path, "profiling_report.html"))

        # --------------------------------------------------------
        # DETECT PROBLEM TYPE
        # --------------------------------------------------------
        problem = detect_problem_type(sample_clean[target_column])

        # --------------------------------------------------------
        # PHASE 1 — TPOT DISCOVERY
        # --------------------------------------------------------
        best_strategy = self._discover_pipeline(sample_clean, target_column, problem)

        # --------------------------------------------------------
        # PHASE 2 — FULL DATA INCREMENTAL TRAINING
        # --------------------------------------------------------
        trained_model, trained_metrics = self._train_full_incremental(csv_path, target_column, problem, project_path)

        # --------------------------------------------------------
        # VISUALIZATION
        # --------------------------------------------------------
        viz = DataVisualizer(plots_dir)
        viz.correlation_heatmap(sample_clean)
        viz.numeric_histograms(sample_clean)

        # --------------------------------------------------------
        # SAVE ARTIFACTS
        # --------------------------------------------------------
        model_path = os.path.join(project_path, "best_model.joblib")
        joblib.dump(trained_model, model_path)
        sample_clean.to_csv(os.path.join(project_path, "cleaned_sample.csv"), index=False)

        summary = {
            "project": project_name,
            "dataset": os.path.basename(csv_path),
            "timestamp": datetime.utcnow().isoformat(),
            "problem_type": problem,
            "discovered_strategy": best_strategy,
            "final_model": "SGD (incremental)",
            "metrics": trained_metrics,
            "model_path": model_path,
            "profile_report": "profiling_report.html",
        }

        self.memory.add_run(summary)
        with open(os.path.join(project_path, "summary.json"), "w", encoding="utf8") as f:
            json.dump(summary, f, indent=2)

        return project_path, summary

    # ============================================================
    # CSV SAMPLER
    # ============================================================
    def _sample_csv(self, path: str, n: int) -> pd.DataFrame:
        chunks = []
        total = 0
        for chunk in pd.read_csv(path, chunksize=CHUNK_SIZE):
            chunks.append(chunk)
            total += len(chunk)
            if total >= n:
                break
        df = pd.concat(chunks, ignore_index=True)
        return df.sample(n=min(len(df), n), random_state=42)

    # ============================================================
    # PHASE 1 — TPOT DISCOVERY
    # ============================================================
    def _discover_pipeline(self, df: pd.DataFrame, target: str, problem: str) -> str:
        X = df.drop(columns=[target])
        y = df[target]

        cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
        if cat_cols:
            enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
            X[cat_cols] = enc.fit_transform(X[cat_cols])

        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)

        if problem == "regression":
            automl = TPOTRegressor(
                generations=5,
                population_size=30,
                n_jobs=1,  # single process for Flask safety
                random_state=42
            )
        else:
            automl = TPOTClassifier(
                generations=5,
                population_size=30,
                n_jobs=1,
                random_state=42
            )

        automl.fit(Xtr, ytr)
        return str(automl.fitted_pipeline_)

    
    def _train_full_incremental(self, path: str, target: str, problem: str, project_path: str):
        if problem == "regression":
            model = SGDRegressor(max_iter=1, tol=None)
        else:
            model = SGDClassifier(max_iter=1, tol=None)

        scaler = StandardScaler()
        label_encoder = None
        first_chunk = True

        classes = None
        if problem != "regression":
            full_target = pd.read_csv(path, usecols=[target])
            if full_target[target].dtype == object:
                label_encoder = LabelEncoder()
                y_full = label_encoder.fit_transform(full_target[target].astype(str))
            else:
                y_full = full_target[target].values
            classes = np.unique(y_full)

        for chunk in pd.read_csv(path, chunksize=CHUNK_SIZE):
            chunk = chunk.dropna(subset=[target])
            if chunk.empty:
                continue

            chunk, cat_cols = self.cleaner.clean(chunk)
            X = chunk.drop(columns=[target])
            y = chunk[target]

            if cat_cols:
                enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
                X[cat_cols] = enc.fit_transform(X[cat_cols])

            num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
            if first_chunk and num_cols:
                scaler.fit(X[num_cols])
            if num_cols:
                X[num_cols] = scaler.transform(X[num_cols])

            # Encode target if needed
            if problem != "regression" and y.dtype == object and label_encoder is not None:
                y = label_encoder.transform(y.astype(str))

            # Partial fit
            if problem == "regression":
                model.partial_fit(X, y)
            else:
                if first_chunk:
                    model.partial_fit(X, y, classes=classes)
                    first_chunk = False
                else:
                    model.partial_fit(X, y)

        # --------------------------------------------------------
        # VALIDATION SAMPLE
        # --------------------------------------------------------
        val = self._sample_csv(path, n=50_000)
        val, _ = self.cleaner.clean(val)
        Xv = val.drop(columns=[target])
        yv = val[target]

        if problem != "regression" and label_encoder is not None and yv.dtype == object:
            yv = label_encoder.transform(yv.astype(str))

        preds = model.predict(Xv)

        if problem == "regression":
            metrics = {
                "mae": float(mean_absolute_error(yv, preds)),
                "r2": float(r2_score(yv, preds)),
            }
        else:
            metrics = {
                "accuracy": float(accuracy_score(yv, preds)),
                "f1": float(f1_score(yv, preds, average="weighted")),
            }

        return model, metrics

    # ============================================================
    # ZIP PROJECT
    # ============================================================
    def package_project(self, project_dir: str) -> str:
        zip_path = os.path.join(self.projects_dir, os.path.basename(project_dir))
        shutil.make_archive(zip_path, "zip", project_dir)
        return zip_path + ".zip"
