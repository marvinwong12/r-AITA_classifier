import json
from datetime import datetime

def log_experiment(model_name: str, params: dict, metrics: dict, log_path: str = "reports/experiment_log.csv"):
    """
    Logs model metadata, hyperparameters, and test metrics to a CSV experiment log.
    """
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    
    # Structure entry
    run_data = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model_name": model_name,
        "macro_f1": round(metrics["macro_f1"], 4),
        "class_1_f1": round(metrics["class_1_f1"], 4),
        "accuracy": round(metrics["accuracy"], 4),
        "hyperparameters": json.dumps(params)
    }
    
    df_log = pd.DataFrame([run_data])
    
    # Append if file exists, create with header if not
    if not os.path.exists(log_path):
        df_log.to_csv(log_path, index=False)
    else:
        df_log.to_csv(log_path, mode='a', header=False, index=False)
        
    print(f"Logged experiment results to {log_path}")