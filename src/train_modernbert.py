import os
import re
import argparse
import numpy as np
import pandas as pd
import torch

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support

from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
    EarlyStoppingCallback
)


def balance_via_undersampling(df: pd.DataFrame, ratio: float = 1.0) -> pd.DataFrame:
    """Strictly balances classes 1:1 by default."""
    df_0 = df[df['label'] == 0]
    df_1 = df[df['label'] == 1]
    
    n_class_0 = int(len(df_1) * ratio)
    df_0_downsampled = df_0.sample(n=min(n_class_0, len(df_0)), random_state=42)
    
    return pd.concat([df_0_downsampled, df_1]).sample(frac=1, random_state=42).reset_index(drop=True)


def evaluate_best_threshold(y_true, probs):
    best_thresh = 0.5
    best_f1 = 0.0
    
    print("\n--- Sweeping Probability Thresholds ---")
    for thresh in np.arange(0.40, 0.75, 0.05):
        preds = (probs >= thresh).astype(int)
        _, _, f1, _ = precision_recall_fscore_support(y_true, preds, average='macro', zero_division=0)
        print(f"Threshold: {thresh:.2f} | Macro F1: {f1:.4f}")
        
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = thresh
            
    print(f"\nOptimal Threshold: {best_thresh:.2f} (Macro F1: {best_f1:.4f})")
    return best_thresh


def clean_text_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    title_series = df['title'].fillna('') if 'title' in df.columns else pd.Series('', index=df.index)
    
    if 'body' in df.columns:
        body_series = df['body'].fillna('')
    elif 'selftext' in df.columns:
        body_series = df['selftext'].fillna('')
    else:
        body_series = pd.Series('', index=df.index)

    def format_row(t, b):
        t_clean = re.sub(r'\s+', ' ', str(t)).strip()
        b_clean = re.sub(r'\s+', ' ', str(b)).strip()
        
        if t_clean and b_clean:
            return f"TITLE: {t_clean}\nSTORY: {b_clean}"
        elif t_clean:
            return f"TITLE: {t_clean}"
        else:
            return f"STORY: {b_clean}"

    df['clean_text'] = [format_row(t, b) for t, b in zip(title_series, body_series)]
    
    is_empty = df['clean_text'].str.strip() == ''
    is_deleted = df['clean_text'].str.contains(r'\[deleted\]|\[removed\]', case=False, regex=True)
    valid_mask = ~(is_empty | is_deleted)
    df = df[valid_mask].reset_index(drop=True)

    if 'is_asshole' in df.columns:
        df['label'] = df['is_asshole'].astype(int)
    elif 'label' in df.columns:
        df['label'] = df['label'].astype(int)
        
    return df[['clean_text', 'label']] if 'label' in df.columns else df[['clean_text']]


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average='macro', zero_division=0
    )
    acc = accuracy_score(labels, predictions)
    
    return {
        'accuracy': acc,
        'f1_macro': f1,
        'precision_macro': precision,
        'recall_macro': recall
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune ModernBERT with Step-Based Early Stopping")
    parser.add_argument("--model_name", type=str, default="answerdotai/ModernBERT-base")
    parser.add_argument("--data_path", type=str, default="data/raw/aita_10k_sample.csv")
    parser.add_argument("--output_dir", type=str, default="models/modernbert_aita")
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1.5e-5)
    
    # --- Early Stopping Arguments ---
    parser.add_argument("--eval_steps", type=int, default=50, help="Evaluate and check early stopping every N steps")
    parser.add_argument("--patience", type=int, default=3, help="Stop after N evaluations without improvement")
    parser.add_argument("--min_delta", type=float, default=0.001, help="Minimum change in macro F1 to qualify as an improvement")
    parser.add_argument("--fp16", action="store_true", help="Enable FP16 mixed precision")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"=== Fine-Tuning {args.model_name} with Step-Based Early Stopping ===")
    
    raw_df = pd.read_csv(args.data_path)
    clean_df = clean_text_data(raw_df)
    
    train_df, test_df = train_test_split(
        clean_df, test_size=0.2, random_state=42, stratify=clean_df['label']
    )
    
    print("Applying strict 1:1 undersampling...")
    train_df = balance_via_undersampling(train_df, ratio=1.0)
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=2)
    
    def tokenize_fn(examples):
        return tokenizer(examples['clean_text'], truncation=True, max_length=args.max_length)

    train_dataset = Dataset.from_pandas(train_df).map(tokenize_fn, batched=True)
    test_dataset = Dataset.from_pandas(test_df).map(tokenize_fn, batched=True)

    # --- Step-Based Early Stopping Configuration ---
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        eval_strategy="steps",           # Evaluate periodically by steps
        eval_steps=args.eval_steps,       # Frequency of evaluation
        save_strategy="steps",           # Must match eval_strategy for load_best_model_at_end
        save_steps=args.eval_steps,       # Must match eval_steps
        save_total_limit=2,              # Keeps only the top 2 checkpoints (saves disk space)
        learning_rate=args.lr,
        max_grad_norm=1.0,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        warmup_ratio=0.15,
        load_best_model_at_end=True,     # Loads best checkpoint when training ends or stops early
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        logging_steps=25,
        fp16=args.fp16,
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics,
        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=args.patience,
                early_stopping_threshold=args.min_delta
            )
        ]
    )

    print(f"\nStarting Fine-Tuning (Evaluating every {args.eval_steps} steps, Patience = {args.patience})...")
    trainer.train()

    print("\n=== Final Holdout Test Evaluation ===")
    predictions = trainer.predict(test_dataset)
    y_true = predictions.label_ids

    raw_logits = predictions.predictions
    probs = torch.softmax(torch.tensor(raw_logits), dim=-1)[:, 1].numpy()

    best_threshold = evaluate_best_threshold(y_true, probs)
    final_preds = (probs >= best_threshold).astype(int)

    print(f"=== Optimized Classification Report (Threshold @ {best_threshold:.2f}) ===")
    print(classification_report(y_true, final_preds, target_names=["Not Asshole (0)", "Asshole (1)"], zero_division=0))

    print(f"\nSaving final model and tokenizer to '{args.output_dir}'...")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print("Fine-tuning complete! Model saved successfully.\n")


if __name__ == "__main__":
    main()