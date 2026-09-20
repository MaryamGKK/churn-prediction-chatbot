import json

import joblib
import numpy as np
import optuna
import pandas as pd
import shap
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text
from xgboost import XGBClassifier

from src.config import (
    CV_FOLDS,
    DATASET_PATH,
    MIN_RECALL,
    MODELS_DIR,
    ONNX_MODEL_PATH,
    OPTUNA_TRIALS,
    PIPELINE_PATH,
    RANDOM_STATE,
    SHAP_EXPLAINER_PATH,
    TEST_SIZE,
    THRESHOLD_PATH,
    TREE_MODEL_PATH,
    XGB_MODEL_PATH,
)
from src.preprocessing import build_pipeline, encode_target


def find_recall_optimized_threshold(
    y_true: np.ndarray, y_proba: np.ndarray, min_recall: float = MIN_RECALL,
) -> float:
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_proba)
    valid = recalls[:-1] >= min_recall
    if not valid.any():
        f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
        return float(thresholds[np.argmax(f1_scores[:-1])])
    f1_scores = 2 * (precisions[:-1][valid] * recalls[:-1][valid]) / (
        precisions[:-1][valid] + recalls[:-1][valid] + 1e-8
    )
    best_idx = np.where(valid)[0][np.argmax(f1_scores)]
    return float(thresholds[best_idx])


def objective(trial: optuna.Trial, X_train: np.ndarray, y_train: np.ndarray) -> float:
    params = {
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "n_estimators": trial.suggest_int("n_estimators", 100, 1000, step=50),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "gamma": trial.suggest_float("gamma", 0.0, 5.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 10.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 10.0),
        "scale_pos_weight": 2.8,
        "eval_metric": "logloss",
        "random_state": RANDOM_STATE,
    }
    model = XGBClassifier(**params)
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="roc_auc")
    return scores.mean()


def export_onnx(model: XGBClassifier, n_features: int) -> None:
    try:
        import onnxmltools
        from onnxmltools.convert import convert_xgboost
        from onnxmltools.convert.common.data_types import FloatTensorType

        initial_type = [("input", FloatTensorType([None, n_features]))]
        onnx_model = convert_xgboost(model, initial_types=initial_type)
        onnxmltools.utils.save_model(onnx_model, str(ONNX_MODEL_PATH))
        print(f"ONNX model saved -> {ONNX_MODEL_PATH}")

        import onnxruntime as ort
        sess = ort.InferenceSession(str(ONNX_MODEL_PATH))
        test_input = np.zeros((1, n_features), dtype=np.float32)
        result = sess.run(None, {"input": test_input})
        print(f"ONNX verification OK - output shape: {result[1].shape}")
    except Exception as e:
        print(f"ONNX export failed (non-critical): {e}")


def train() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(DATASET_PATH)
    y = encode_target(df["Churn"])

    pipeline = build_pipeline()
    X = pipeline.fit_transform(df)
    joblib.dump(pipeline, PIPELINE_PATH)
    print(f"Pipeline saved -> {PIPELINE_PATH} | shape: {X.shape}")

    feature_names = list(pipeline.named_steps["encode"].get_feature_names_out())
    print(f"Features ({len(feature_names)}): {feature_names[:10]}...")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y,
    )

    print(f"\nOptuna tuning ({OPTUNA_TRIALS} trials, {CV_FOLDS}-fold CV)...")
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize")
    study.optimize(lambda t: objective(t, X_train, y_train), n_trials=OPTUNA_TRIALS)
    print(f"Best AUC (CV): {study.best_value:.4f}")
    print(f"Best params: {study.best_params}")

    best_params = {
        **study.best_params,
        "scale_pos_weight": 2.8,
        "eval_metric": "logloss",
        "random_state": RANDOM_STATE,
    }
    xgb = XGBClassifier(**best_params)
    xgb.fit(X_train, y_train)
    joblib.dump(xgb, XGB_MODEL_PATH)
    print(f"XGBoost saved -> {XGB_MODEL_PATH}")

    y_proba = xgb.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_proba)
    threshold = find_recall_optimized_threshold(y_test, y_proba, MIN_RECALL)
    y_pred = (y_proba >= threshold).astype(int)

    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print(f"\n--- XGBoost Evaluation (threshold={threshold:.3f}, min_recall={MIN_RECALL}) ---")
    print(f"AUC-ROC: {auc:.4f}")
    print(f"Recall: {recall:.4f} {'(TARGET MET)' if recall >= MIN_RECALL else '(BELOW TARGET)'}")
    print(f"F1 (churn): {f1:.4f}")
    print(classification_report(y_test, y_pred, target_names=["No Churn", "Churn"]))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    with open(THRESHOLD_PATH, "w") as f:
        json.dump({
            "threshold": threshold, "auc": auc,
            "recall": recall, "f1": f1,
            "feature_names": feature_names,
        }, f, indent=2)
    print(f"Threshold saved -> {THRESHOLD_PATH}")

    tree = DecisionTreeClassifier(max_depth=5, class_weight="balanced", random_state=RANDOM_STATE)
    tree.fit(X_train, y_train)
    joblib.dump(tree, TREE_MODEL_PATH)
    tree_auc = roc_auc_score(y_test, tree.predict_proba(X_test)[:, 1])
    print(f"\n--- Decision Tree (CART) ---")
    print(f"AUC-ROC: {tree_auc:.4f}")

    tree_rules = export_text(tree, feature_names=feature_names, max_depth=3)
    print("Top rules (depth 3):")
    print(tree_rules[:1000])

    explainer = shap.TreeExplainer(xgb)
    joblib.dump(explainer, SHAP_EXPLAINER_PATH)
    print(f"SHAP explainer saved -> {SHAP_EXPLAINER_PATH}")

    print("\n--- ONNX Export ---")
    export_onnx(xgb, X.shape[1])

    print("\n=== Training complete. All artifacts saved to models/ ===")


if __name__ == "__main__":
    train()
