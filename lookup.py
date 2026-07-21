import os
import glob
import sys

sys.path.append("src")  # <-- Change to "." if preprocess.py is in your root folder
sys.path.append(".")


import joblib
import pandas as pd

def compare_all_models(model_dir="models"):
    """
    Finds all .joblib files in model_dir, extracts metrics,
    and returns a DataFrame sorted by Macro F1.
    """
    records = []
    file_pattern = os.path.join(model_dir, "*.joblib")
    
    for filepath in glob.glob(file_pattern):
        try:
            bundle = joblib.load(filepath)
            metrics = bundle.get("metrics", {})
            config = bundle.get("config", {})
            
            records.append({
                "Filename": os.path.basename(filepath),
                "Model Key": bundle.get("model_key", "N/A"),
                "Macro F1": metrics.get("macro_f1"),
                "Class 1 F1": metrics.get("class_1_f1"),
                "Accuracy": metrics.get("accuracy"),
                "Feature Type": config.get("feature_type"),
                "Threshold": config.get("threshold"),
                "Optimized Thresh": config.get("threshold_optimized")
            })
        except Exception as e:
            print(f"Could not load '{filepath}': {e}")
            
    if not records:
        print("No .joblib model files found.")
        return None
        
    df = pd.DataFrame(records)
    # Sort models by Macro F1 score (highest first)
    df = df.sort_values(by="Macro F1", ascending=False).reset_index(drop=True)
    return df

# Run comparison
results_df = compare_all_models("models")
if results_df is not None:
    print("\n=== Model Comparison Leaderboard ===")
    print(results_df.to_string(index=False))


