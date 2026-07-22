import joblib
import torch
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification


def analyze_all_models(
    test_df_path: str,
    logreg_baseline_path: str,
    logreg_meta_path: str,
    roberta_ce_path: str,
    roberta_focal_path: str,
    batch_size: int = 32
) -> pd.DataFrame:
    """
    Runs evaluation and prediction across all 4 trained model artifacts
    (2 joblib Logistic Regression models and 2 saved RoBERTa folder checkpoints).
    """
    print(f"Loading test set from '{test_df_path}'...")
    test_df = pd.read_csv(test_df_path)
    
    # Standardize target column inside the analysis DataFrame
    target_col = 'is_asshole' if 'is_asshole' in test_df.columns else 'label'
    analysis_df = test_df.copy()
    if target_col in analysis_df.columns:
        analysis_df['is_asshole'] = analysis_df[target_col].astype(int)

    # =========================================================================
    # 1. LOGISTIC REGRESSION (Standard TF-IDF Baseline)
    # =========================================================================
    print("\n[1/4] Running inference on Logistic Regression (TF-IDF Baseline)...")
    base_bundle = joblib.load(logreg_baseline_path)
    base_preprocessor = base_bundle["preprocessor"]
    base_model = base_bundle["model"]
    
    _, X_test_base, _ = base_preprocessor.transform(test_df)
    base_thresh = base_bundle["config"].get("threshold", 0.5)
    
    base_probs = base_model.predict_proba(X_test_base)[:, 1]
    analysis_df['prob_logreg_base'] = base_probs
    analysis_df['pred_logreg_base'] = (base_probs >= base_thresh).astype(int)

    # =========================================================================
    # 2. LOGISTIC REGRESSION (Metadata Enhanced)
    # =========================================================================
    print("[2/4] Running inference on Logistic Regression (+ Metadata)...")
    meta_bundle = joblib.load(logreg_meta_path)
    meta_preprocessor = meta_bundle["preprocessor"]
    meta_model = meta_bundle["model"]
    
    _, X_test_meta, _ = meta_preprocessor.transform(test_df)
    meta_thresh = meta_bundle["config"].get("threshold", 0.5)
    
    meta_probs = meta_model.predict_proba(X_test_meta)[:, 1]
    analysis_df['prob_logreg_meta'] = meta_probs
    analysis_df['pred_logreg_meta'] = (meta_probs >= meta_thresh).astype(int)

    # =========================================================================
    # 3. ROBERTA BATCHED INFERENCE HELPER
    # =========================================================================
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing compute device for Transformers: {device.type.upper()}")

    def get_roberta_probs(model_dir: str) -> np.ndarray:
        """Loads a saved Hugging Face directory checkpoint and performs batched inference."""
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        model.to(device)
        model.eval()
        
        # Prepare text prompts
        title_series = test_df['title'].fillna('').astype(str)
        if 'body' in test_df.columns:
            body_series = test_df['body'].fillna('').astype(str)
        elif 'selftext' in test_df.columns:
            body_series = test_df['selftext'].fillna('').astype(str)
        else:
            body_series = pd.Series('', index=test_df.index)

        texts = [
            f"TITLE: {t.strip()}\nSTORY: {b.strip()}" if t and b else (f"TITLE: {t.strip()}" if t else f"STORY: {b.strip()}")
            for t, b in zip(title_series, body_series)
        ]

        all_probs = []
        
        # Mini-batch processing to prevent GPU/RAM OOM
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]
            inputs = tokenizer(
                batch_texts, 
                padding=True, 
                truncation=True, 
                max_length=512, 
                return_tensors="pt"
            ).to(device)
            
            with torch.no_grad():
                outputs = model(**inputs)
                probs = torch.softmax(outputs.logits, dim=-1)[:, 1].cpu().numpy()
                all_probs.extend(probs)
                
        return np.array(all_probs)

    # =========================================================================
    # 4. ROBERTA MODELS (Cross-Entropy & Focal Loss)
    # =========================================================================
    print("[3/4] Running inference on RoBERTa (Full Dataset - Cross Entropy)...")
    ce_probs = get_roberta_probs(roberta_ce_path)
    analysis_df['prob_roberta_ce'] = ce_probs
    analysis_df['pred_roberta_ce'] = (ce_probs >= 0.50).astype(int)

    print("[4/4] Running inference on RoBERTa (Full Dataset - Focal Loss)...")
    focal_probs = get_roberta_probs(roberta_focal_path)
    analysis_df['prob_roberta_focal'] = focal_probs
    analysis_df['pred_roberta_focal'] = (focal_probs >= 0.50).astype(int)

    print("\nInference complete across all 4 models!\n")
    return analysis_df