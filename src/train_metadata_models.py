import os
import argparse
import joblib
import pandas as pd
import numpy as np
from scipy.stats import loguniform, randint

from sklearn.model_selection import train_test_split, cross_validate, cross_val_predict, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score

# Optional: Import XGBoost if installed
try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

# Import the new metadata-enhanced preprocessor
from preprocess_metadata import AITAMetadataPreprocessor


def get_model(model_type: str, class_weight: str = "balanced", random_state: int = 42, optimize: bool = False):
    """
    Model Factory: Instantiates and returns the specified machine learning model.
    If optimize=True, returns a RandomizedSearchCV object wrapped around the model.
    """
    model_type = model_type.lower()

    if model_type == "logreg":
        base_model = LogisticRegression(max_iter=1000, class_weight=class_weight, random_state=random_state)
        param_dist = {
            'C': loguniform(1e-3, 10.0),  # Search continuously between 0.001 and 10.0
            'solver': ['lbfgs', 'liblinear']
        }
    
    elif model_type == "svm":
        base_model = LinearSVC(max_iter=2000, class_weight=class_weight, random_state=random_state)
        param_dist = {
            'C': loguniform(1e-3, 1e2)
        }
    
    elif model_type == "rf":
        base_model = RandomForestClassifier(class_weight=class_weight, random_state=random_state, n_jobs=-1)
        param_dist = {
            'n_estimators': randint(100, 500),
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': randint(2, 10)
        }
    
    elif model_type == "xgboost":
        if not HAS_XGBOOST:
            raise ImportError("XGBoost is not installed. Run `pip install xgboost` first.")
            
        base_model = XGBClassifier(scale_pos_weight=2.5, random_state=random_state, n_jobs=-1)
        param_dist = {
            'n_estimators': randint(100, 500),
            'learning_rate': loguniform(0.01, 0.3),
            'max_depth': randint(3, 10),
            'subsample': [0.8, 0.9, 1.0]
        }
    else:
        raise ValueError(f"Unknown model_type '{model_type}'")

    if not optimize:
        return base_model

    return RandomizedSearchCV(
        estimator=base_model,
        param_distributions=param_dist,
        n_iter=10, 
        scoring='f1_macro', 
        cv=3, 
        n_jobs=-1,
        random_state=random_state,
        verbose=1
    )


def get_artifact_path(model_dir: str, model_type: str, custom_name: str = None) -> str:
    if custom_name:
        filename = custom_name if custom_name.endswith(".joblib") else f"{custom_name}.joblib"
    else:
        filename = f"aita_{model_type}_metadata_bundle.joblib"
    return os.path.join(model_dir, filename)


def find_best_threshold(y_true: np.ndarray, y_probs: np.ndarray) -> float:
    thresholds = np.linspace(0.1, 0.9, 81)
    best_t = 0.5
    best_f1 = 0.0
    
    for t in thresholds:
        y_pred = (y_probs >= t).astype(int)
        score = f1_score(y_true, y_pred, average="macro")
        if score > best_f1:
            best_f1 = score
            best_t = t
            
    return best_t


def parse_args():
    parser = argparse.ArgumentParser(description="Train Metadata-Enhanced Classification Models on AITA Dataset")
    parser.add_argument("--model", type=str, default="logreg", choices=["logreg", "svm", "rf", "xgboost"])
    parser.add_argument("--class_weight", type=str, default="balanced", choices=["balanced", "none"])
    parser.add_argument("--max_features", type=int, default=3000, help="Max features for TF-IDF vocabulary")
    parser.add_argument("--data_path", type=str, default="data/raw/aita_10k_sample.csv")
    parser.add_argument("--model_dir", type=str, default="models")
    parser.add_argument("--threshold", type=float, default=0.5, help="Manual decision threshold")
    parser.add_argument("--output_name", type=str, default=None)
    parser.add_argument("--optimize_threshold", action="store_true", help="Automatically find best threshold using CV")
    parser.add_argument("--optimize_hyperparams", action="store_true", help="Run RandomizedSearchCV for hyperparameters")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.model_dir, exist_ok=True)
    cw = None if args.class_weight == "none" else "balanced"

    print(f"=== Starting Metadata-Enhanced Pipeline [{args.model.upper()}] ===")
    
    print(f"Loading raw data from '{args.data_path}'...")
    raw_df = pd.read_csv(args.data_path)
    
    # Identify target column
    target_col = 'is_asshole' if 'is_asshole' in raw_df.columns else 'label'
    
    print("Splitting raw data into Train and Test sets (80/20)...")
    train_df, test_df = train_test_split(
        raw_df, test_size=0.2, random_state=42, 
        stratify=raw_df[target_col] if target_col in raw_df.columns else None
    )
    
    print(f"Fitting AITAMetadataPreprocessor (max_features={args.max_features})...")
    preprocessor = AITAMetadataPreprocessor(max_features=args.max_features)
    train_clean_df, X_train, y_train = preprocessor.fit_transform(train_df)
    
    print("Transforming Test Set via preprocessor...")
    test_clean_df, X_test, y_test = preprocessor.transform(test_df)
    
    print(f"Train Matrix X shape: {X_train.shape} | Test Matrix X shape: {X_test.shape}")
    
    # 1. Initialize Model
    clf = get_model(args.model, class_weight=cw, optimize=args.optimize_hyperparams)
    
    # 2. Hyperparameter Tuning (if enabled)
    if args.optimize_hyperparams:
        print("\nRunning Hyperparameter Optimization...")
        clf.fit(X_train, y_train)
        print(f"-> Best Hyperparameters found: {clf.best_params_}")
        clf = clf.best_estimator_ 
    else:
        print(f"\nInitialized Model: {clf.__class__.__name__}")
    
    supports_proba = hasattr(clf, "predict_proba")
    
    # 3. Standard CV Baseline Metrics
    print("\nRunning 5-Fold Cross Validation...")
    cv_results = cross_validate(
        clf, X_train, y_train, 
        cv=5, 
        scoring=['f1_macro', 'accuracy'], 
        return_train_score=True,
        n_jobs=-1
    )

    train_f1 = np.mean(cv_results['train_f1_macro'])
    val_f1 = np.mean(cv_results['test_f1_macro'])

    print(f"CV Train Macro F1: {train_f1:.3f}")
    print(f"CV Val   Macro F1: {val_f1:.3f}")
    print(f"Overfitting Gap:   {train_f1 - val_f1:.3f}")
    
    # 4. Threshold Optimization
    if args.optimize_threshold:
        if supports_proba:
            print("\nOptimizing decision threshold via Out-of-Fold predictions...")
            y_oof_probs = cross_val_predict(clf, X_train, y_train, cv=5, method="predict_proba", n_jobs=-1)[:, 1]
            optimal_t = find_best_threshold(y_train, y_oof_probs)
            print(f"-> Optimal Threshold found: {optimal_t:.2f}")
            args.threshold = optimal_t
        else:
            print(f"\n⚠️ Warning: {clf.__class__.__name__} does not support probabilities. Skipping threshold optimization.")

    # 5. Final Model Fit
    print(f"\nTraining final {args.model.upper()} model on full training set...")
    clf.fit(X_train, y_train)
    
    # 6. Evaluation
    print(f"Evaluating on Holdout Test Set (Threshold: {args.threshold:.2f})...")
    if args.threshold != 0.5 and supports_proba:
        y_probs = clf.predict_proba(X_test)[:, 1]
        y_pred = (y_probs >= args.threshold).astype(int)
    else:
        y_pred = clf.predict(X_test)
    
    print("\n=== Classification Report ===")
    print(classification_report(y_test, y_pred, target_names=["Not Asshole (0)", "Asshole (1)"]))
    
    # 7. Model Bundling
    dict_report = classification_report(y_test, y_pred, output_dict=True)
    class_1_key = "1" if "1" in dict_report else 1
    
    artifact_path = get_artifact_path(args.model_dir, args.model, args.output_name)
    print(f"\nSaving bundled model artifact to '{artifact_path}'...")
    
    model_bundle = {
        "model_name": clf.__class__.__name__,
        "model_key": args.model,
        "model": clf,
        "preprocessor": preprocessor,
        "metrics": {
            "macro_f1": dict_report["macro avg"]["f1-score"],
            "class_1_f1": dict_report[class_1_key]["f1-score"],
            "accuracy": dict_report["accuracy"]
        },
        "config": {
            "preprocessor": "AITAMetadataPreprocessor",
            "max_features": args.max_features,
            "threshold": args.threshold,
            "threshold_optimized": args.optimize_threshold,
            "hyperparams_optimized": args.optimize_hyperparams,
            "class_weight": args.class_weight
        }
    }
    
    joblib.dump(model_bundle, artifact_path)
    print(f"Training complete! Artifact successfully saved to '{artifact_path}'.\n")


if __name__ == "__main__":
    main()