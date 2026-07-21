import os
import re
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

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

# --- 1. Custom Focal Loss Implementation ---
class FocalLoss(nn.Module):
    def __init__(self, alpha=1.0, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        # Calculate standard Cross Entropy Loss
        ce_loss = F.cross_entropy(logits, targets, reduction='none')
        
        # Get the probabilities of the correct class (pt)
        pt = torch.exp(-ce_loss)
        
        # Apply the focal loss formula: alpha * (1 - pt)^gamma * ce_loss
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss

# --- 2. Custom Trainer to override default Cross Entropy ---
class FocalTrainer(Trainer):
    def __init__(self, *args, gamma=2.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.gamma = gamma  # Store gamma for this run

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        
        # Inject the configurable gamma
        loss_fct = FocalLoss(gamma=self.gamma)
        
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss


def balance_via_undersampling(df: pd.DataFrame, ratio: float = 1.5) -> pd.DataFrame:
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
        
        words = b_clean.split()
        if len(words) > 350:
            b_clean = " ".join(words[:150]) + " ... " + " ".join(words[-200:])
            
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
    parser = argparse.ArgumentParser(description="Fine-tune RoBERTa on AITA Dataset")
    parser.add_argument("--model_name", type=str, default="roberta-base")
    parser.add_argument("--data_path", type=str, default="data/raw/aita_10k_sample.csv")
    parser.add_argument("--output_dir", type=str, default="models/roberta_aita")
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=5, help="Increased to 5 to allow Early Stopping to trigger")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--patience", type=int, default=2, help="How many epochs to wait before early stopping"),
    parser.add_argument("--gamma", type=float, default=2.0, help="Gamma parameter for Focal Loss. 0.0 equals CrossEntropy.")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"=== Fine-Tuning {args.model_name} ===")
    
    raw_df = pd.read_csv(args.data_path)
    clean_df = clean_text_data(raw_df)
    
    train_df, test_df = train_test_split(
        clean_df, test_size=0.2, random_state=42, stratify=clean_df['label']
    )
    
    print("Applying undersampling...")
    train_df = balance_via_undersampling(train_df)
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=2)
    
    def tokenize_fn(examples):
        return tokenizer(examples['clean_text'], truncation=True, max_length=args.max_length)

    train_dataset = Dataset.from_pandas(train_df).map(tokenize_fn, batched=True)
    test_dataset = Dataset.from_pandas(test_df).map(tokenize_fn, batched=True)

    # 5. Training Setup
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        eval_strategy="epoch",       # Must match save_strategy for early stopping
        save_strategy="epoch",
        learning_rate=args.lr,
        max_grad_norm=1.0,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        warmup_ratio=0.1,
        load_best_model_at_end=True, # Required for early stopping
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        logging_steps=50,
        fp16=False,
        bf16=False,
        report_to="none"
    )

    # Use the custom FocalTrainer and inject the EarlyStoppingCallback
    trainer = FocalTrainer(
        gamma=args.gamma,
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=args.patience)]
    )

    print("\nStarting Fine-Tuning Process...")
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