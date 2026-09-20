import json
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.config import MODELS_DIR, ONNX_MODEL_PATH
from src.schemas import CustomerFeatures, PredictionResult

logger = logging.getLogger(__name__)


class ChurnPredictor:
    def __init__(self, models_dir: Path = MODELS_DIR):
        self.pipeline = joblib.load(models_dir / "pipeline.joblib")
        self.xgb_model = joblib.load(models_dir / "xgb_model.joblib")
        self.tree = joblib.load(models_dir / "decision_tree.joblib")
        self.explainer = joblib.load(models_dir / "shap_explainer.joblib")

        with open(models_dir / "threshold.json") as f:
            meta = json.load(f)
        self.threshold = meta["threshold"]
        self.feature_names = meta.get("feature_names", list(
            self.pipeline.named_steps["encode"].get_feature_names_out()
        ))

        self.onnx_session = None
        onnx_path = models_dir / "xgb_model.onnx"
        if onnx_path.exists():
            try:
                import onnxruntime as ort
                self.onnx_session = ort.InferenceSession(str(onnx_path))
                logger.info("ONNX runtime loaded for inference")
            except Exception:
                logger.warning("ONNX available but failed to load, using XGBoost")

    def predict(self, features: CustomerFeatures) -> PredictionResult:
        row_df = pd.DataFrame([features.to_dataframe_row()])
        X = self.pipeline.transform(row_df)

        if self.onnx_session is not None:
            X_float = X.astype(np.float32)
            input_name = self.onnx_session.get_inputs()[0].name
            probas = self.onnx_session.run(None, {input_name: X_float})[1]
            proba = float(probas[0][1])
        else:
            proba = float(self.xgb_model.predict_proba(X)[0, 1])

        prediction = proba >= self.threshold

        shap_values = self.explainer.shap_values(X)[0]
        top_indices = np.argsort(np.abs(shap_values))[::-1][:5]
        top_factors = [
            {
                "feature": self._clean_feature_name(self.feature_names[i]),
                "value": float(X[0, i]),
                "impact": float(shap_values[i]),
            }
            for i in top_indices
        ]

        tree_path = self._get_tree_path(X)

        return PredictionResult(
            churn_probability=round(proba, 4),
            churn_prediction=prediction,
            top_factors=top_factors,
            decision_tree_path=tree_path,
        )

    def _clean_feature_name(self, name: str) -> str:
        for prefix in ("num__", "bin__", "cat__"):
            if name.startswith(prefix):
                name = name[len(prefix):]
        return name.replace("_", " ").title()

    def _get_tree_path(self, X: np.ndarray) -> str:
        tree = self.tree.tree_
        node = 0
        path_parts = []

        while tree.children_left[node] != tree.children_right[node]:
            feature_idx = tree.feature[node]
            threshold = tree.threshold[node]
            feature_name = self._clean_feature_name(self.feature_names[feature_idx])
            value = X[0, feature_idx]

            if value <= threshold:
                path_parts.append(f"{feature_name} <= {threshold:.2f}")
                node = tree.children_left[node]
            else:
                path_parts.append(f"{feature_name} > {threshold:.2f}")
                node = tree.children_right[node]

        counts = tree.value[node][0]
        label = "Churn" if counts[1] > counts[0] else "No Churn"
        confidence = counts[int(counts[1] > counts[0])] / counts.sum()

        return f"{' -> '.join(path_parts)} -> {label} ({confidence:.0%} confidence)"
