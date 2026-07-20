import os
import re
import argparse
import numpy as np
import pandas as pd
import torch

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score, accuracy_score, precision_recall_fscore_support

from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding
)


def clean_text_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans raw dataframe: filters out deleted/removed/empty body text.
    """
    df = df.copy()
    if 'body' not in df.columns:
        df['body'] = ''
    else:
        df['body'] = df['body'].fillna('')

    if 'is_asshole' in df.columns:
        has_no_body = df['body'].isna() | (df['body'].str.strip() == '')
        is_deleted_or_removed = df['body'].astype(str).str.contains(
            r'\[deleted\]|\[removed\]', case=False, regex=True
        )
        valid_mask = ~(has_no_body | is_deleted_or_removed)
        df = df[valid_mask].reset_index(drop=True)

    # Standardize whitespace
    df['clean_text'] = df['body'].apply(lambda x: re.sub(r'\s+', ' ', str(x)).strip())
    # Rename target label column to 'label' for Hugging Face Trainer compatibility
    if 'is_asshole' in df.columns:
        df['label'] = df['is_asshole'].astype(int)
        
    return df[['clean_text', 'label']] if 'label' in df.columns else df[['clean_text']]


def compute_metrics(eval_pred):
    """
    Computes evaluation metrics during training epochs.
    """
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='macro')
    acc = accuracy_score(labels, predictions)
    
    return {
        'accuracy': acc,
        'f1_macro': f1,
        'precision_macro': precision,
        'recall_macro': recall
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune DeBERTa on AITA Dataset")
    parser.add_argument("--model_name", type=str, default="microsoft/deberta-v3-small", help="Hugging Face model checkpoint")
    parser.add_argument("--data_path", type=str, default="data/raw/aita_10k_sample.csv")
    parser.add_argument("--output_dir", type=str, default="models/deberta_aita")
    parser.add_argument("--max_length", type=int, default=512, help="Max token sequence length")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate for fine-tuning")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"=== Fine-Tuning {args.model_name} ===")
    
    # 1. Load and Clean Data
    print(f"Loading raw data from '{args.data_path}'...")
    raw_df = pd.read_csv(args.data_path)
    clean_df = clean_text_data(raw_df)
    
    # 2. Train/Test Split (80/20)
    train_df, test_df = train_test_split(
        clean_df, test_size=0.2, random_state=42, stratify=clean_df['label']
    )
    
    print(f"Train set size: {len(train_df)} | Test set size: {len(test_df)}")
    
    # 3. Load Tokenizer & Model
    print(f"Loading tokenizer & model weights for '{args.model_name}'...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=2)
    
    # 4. Tokenization Function
    def tokenize_fn(examples):
        return tokenizer(
            examples['clean_text'], 
            truncation=True, 
            max_length=args.max_length
        )

    # Convert Pandas DataFrames to Hugging Face Datasets
    train_dataset = Dataset.from_pandas(train_df)
    test_dataset = Dataset.from_pandas(test_df)

    print("Tokenizing datasets...")
    train_dataset = train_dataset.map(tokenize_fn, batched=True)
    test_dataset = test_dataset.map(tokenize_fn, batched=True)

    # 5. Training Setup
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        logging_steps=50,
        fp16=torch.cuda.is_available(), # Auto-enable mixed precision if GPU is present
        report_to="none" # Disable wandb/mlflow logging by default
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics
    )

    # 6. Train
    print("\nStarting Fine-Tuning Process...")
    trainer.train()

    # 7. Evaluate on Holdout Test Set
    print("\n=== Final Holdout Test Evaluation ===")
    predictions = trainer.predict(test_dataset)
    y_preds = np.argmax(predictions.predictions, axis=-1)
    y_true = predictions.label_ids

    print("\n=== Classification Report ===")
    print(classification_report(y_true, y_preds, target_names=["Not Asshole (0)", "Asshole (1)"]))

    # 8. Save Final Artifacts locally
    print(f"\nSaving final model and tokenizer to '{args.output_dir}'...")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print("Fine-tuning complete! Model saved successfully.\n")


if __name__ == "__main__":
    main()