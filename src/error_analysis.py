import sys
import os

# Alias preprocessors for joblib unpickling
try:
    import src.preprocess as preprocess
    sys.modules['preprocess'] = preprocess
except ImportError:
    pass

try:
    import src.preprocess_metadata as preprocess_metadata
    sys.modules['preprocess_metadata'] = preprocess_metadata
except ImportError:
    pass

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
    Runs evaluation across all 4 trained models on identical test samples.
    """
    print(f"Loading test set from '{test_df_path}'...")
    raw_test_df = pd.read_csv(test_df_path)
    
    # Ensure text columns exist
    body_col = 'body' if 'body' in raw_test_df.columns else ('selftext' if 'selftext' in raw_test_df.columns else '')
    body_series = raw_test_df[body_col].fillna('').astype(str) if body_col else pd.Series('', index=raw_test_df.index)
    title_series = raw_test_df['title'].fillna('').astype(str) if 'title' in raw_test_df.columns else pd.Series('', index=raw_test_df.index)

    # Filter completely blank posts up front so row counts stay aligned
    is_empty = (body_series.str.strip() == '') & (title_series.str.strip() == '')
    is_deleted = body_series.str.contains(r'^\[deleted\]$|^\[removed\]$', case=False, regex=True)
    valid_mask = ~(is_empty | is_deleted)

    test_df = raw_test_df[valid_mask].reset_index(drop=True)
    print(f"Filtered out {len(raw_test_df) - len(test_df)} deleted/empty posts. Valid test set size: {len(test_df)} rows.")

    target_col = 'is_asshole' if 'is_asshole' in test_df.columns else 'label'
    analysis_df = test_df.copy()
    if target_col in analysis_df.columns:
        analysis_df['is_asshole'] = analysis_df[target_col].astype(int)

    # Strip target columns for preprocessor transformation so clean_text() doesn't silently drop rows
    df_for_prep = test_df.drop(columns=['is_asshole', 'label'], errors='ignore')

    # =========================================================================
    # 1. LOGISTIC REGRESSION (Standard TF-IDF Baseline)
    # =========================================================================
    print("\n[1/4] Running inference on Logistic Regression (TF-IDF Baseline)...")
    base_bundle = joblib.load(logreg_baseline_path)
    base_preprocessor = base_bundle["preprocessor"]
    base_model = base_bundle["model"]
    
    _, X_test_base, _ = base_preprocessor.transform(df_for_prep)
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
    
    _, X_test_meta, _ = meta_preprocessor.transform(df_for_prep)
    meta_thresh = meta_bundle["config"].get("threshold", 0.5)
    
    meta_probs = meta_model.predict_proba(X_test_meta)[:, 1]
    analysis_df['prob_logreg_meta'] = meta_probs
    analysis_df['pred_logreg_meta'] = (meta_probs >= meta_thresh).astype(int)

    # =========================================================================
    # 3. ROBERTA BATCHED INFERENCE HELPER
    # =========================================================================
    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    print(f"\nUsing compute device for Transformers: {str(device).upper()}")

    def get_roberta_probs(model_dir: str) -> np.ndarray:
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        model.to(device)
        model.eval()
        
        t_series = test_df['title'].fillna('').astype(str)
        b_series = test_df['body'].fillna('').astype(str) if 'body' in test_df.columns else pd.Series('', index=test_df.index)

        texts = [
            f"TITLE: {t.strip()}\nSTORY: {b.strip()}" if t and b else (f"TITLE: {t.strip()}" if t else f"STORY: {b.strip()}")
            for t, b in zip(t_series, b_series)
        ]

        all_probs = []
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