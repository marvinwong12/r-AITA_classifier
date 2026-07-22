import os
import re
import nltk
import numpy as np
import pandas as pd
from typing import Tuple
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import RobustScaler

# Download VADER lexicon on first run if missing
try:
    nltk.data.find('sentiment/vader_lexicon')
except LookupError:
    nltk.download('vader_lexicon', quiet=True)


class AITAMetadataPreprocessor:
    """
    Feature Preprocessor for AITA Posts that combines:
    1. Text Features: TF-IDF on Title + Body text
    2. Numerical Metadata: Score, Comment Count, Controversiality Ratio, Edit Status
    3. Temporal Metadata: Hour of Day, Day of Week, Weekend Status (derived from Timestamp)
    4. Sentiment & Stylometrics: VADER polarity scores, CAPS ratio, Pronoun counts
    """
    def __init__(self, max_features: int = 3000):
        self.max_features = max_features
        
        # 1. Text Vectorizer (TF-IDF on Title + Body)
        self.vectorizer = TfidfVectorizer(
            max_features=self.max_features,
            ngram_range=(1, 2),
            stop_words='english',
            sublinear_tf=True
        )
        
        # 2. Sentiment Analyzer & Feature Scaler
        self.sia = SentimentIntensityAnalyzer()
        self.scaler = RobustScaler()
        
        # Track feature counts for sanity logging
        self.num_text_features = 0
        self.num_dense_features = 0

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Standardizes missing values, parses timestamps, drops deleted posts during training.
        """
        df = df.copy()

        # Handle missing text fields
        df['title'] = df['title'].fillna('').astype(str)
        df['body'] = df['body'].fillna('').astype(str) if 'body' in df.columns else pd.Series('', index=df.index)

        # Combine Title and Body into unified prompt string
        def format_full_text(t, b):
            t_clean = re.sub(r'\s+', ' ', t).strip()
            b_clean = re.sub(r'\s+', ' ', b).strip()
            if t_clean and b_clean:
                return f"TITLE: {t_clean}\nSTORY: {b_clean}"
            return t_clean or b_clean

        df['clean_full_text'] = [format_full_text(t, b) for t, b in zip(df['title'], df['body'])]

        # Filter deleted or empty posts if target label exists (Training Phase)
        if 'is_asshole' in df.columns:
            is_empty = df['clean_full_text'].str.strip() == ''
            is_deleted = df['clean_full_text'].str.contains(r'\[deleted\]|\[removed\]', case=False, regex=True)
            valid_mask = ~(is_empty | is_deleted)
            df = df[valid_mask].reset_index(drop=True)

        return df

    def extract_metadata_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Engineers numerical, temporal, and metadata signal columns.
        """
        meta_df = pd.DataFrame(index=df.index)

        # 1. Post Engagement Metrics (Log Transformed to handle high skew)
        score = df['score'].fillna(0).astype(float) if 'score' in df.columns else pd.Series(0.0, index=df.index)
        num_comments = df['num_comments'].fillna(0).astype(float) if 'num_comments' in df.columns else pd.Series(0.0, index=df.index)
        
        meta_df['score_log'] = np.log1p(np.maximum(0, score))
        meta_df['comments_log'] = np.log1p(np.maximum(0, num_comments))
        
        # Comment-to-Upvote Ratio (High ratio = controversial/debated post)
        meta_df['comment_to_score_ratio'] = num_comments / (np.maximum(0, score) + 10.0)

        # 2. Edit Status (Binary indicator)
        if 'edited' in df.columns:
            meta_df['is_edited'] = df['edited'].apply(
                lambda x: 0 if pd.isna(x) or str(x).lower() in ['false', '0'] else 1
            )
        else:
            meta_df['is_edited'] = 0

        # 3. Temporal Signals (Parsed from Timestamp)
        if 'timestamp' in df.columns:
            # Flexible datetime parsing (handles Unix timestamps or ISO string dates)
            dt_series = pd.to_datetime(df['timestamp'], unit='s', errors='coerce')
            dt_series = dt_series.fillna(pd.to_datetime(df['timestamp'], errors='coerce'))
            
            meta_df['hour_of_day'] = dt_series.dt.hour.fillna(12)
            meta_df['day_of_week'] = dt_series.dt.dayofweek.fillna(0)
            meta_df['is_weekend'] = meta_df['day_of_week'].isin([5, 6]).astype(int)
        else:
            meta_df['hour_of_day'] = 12
            meta_df['day_of_week'] = 0
            meta_df['is_weekend'] = 0

        # 4. Text Length Metrics
        meta_df['title_char_len'] = df['title'].apply(len)
        meta_df['title_word_count'] = df['title'].apply(lambda x: len(x.split()))
        meta_df['body_char_len'] = df['body'].apply(len)
        meta_df['body_word_count'] = df['body'].apply(lambda x: len(x.split()))

        # 5. Emotional / Stylometric Signals
        full_text = df['clean_full_text']
        meta_df['caps_ratio'] = full_text.apply(lambda x: sum(1 for c in x if c.isupper()) / (len(x) + 1))
        meta_df['exclamation_count'] = full_text.apply(lambda x: x.count('!'))
        meta_df['question_count'] = full_text.apply(lambda x: x.count('?'))

        # Pronoun Density Signals
        meta_df['i_pronouns'] = full_text.apply(lambda x: len(re.findall(r'\b(i|me|my|mine|myself)\b', x, re.I)))
        meta_df['they_pronouns'] = full_text.apply(lambda x: len(re.findall(r'\b(he|him|his|she|her|they|them|their)\b', x, re.I)))

        # 6. Sentiment Polarity Scores (VADER)
        sentiments = full_text.apply(lambda x: self.sia.polarity_scores(x))
        sentiment_df = pd.DataFrame(sentiments.tolist(), index=df.index)
        sentiment_df.columns = [f"vader_{col}" for col in sentiment_df.columns]

        return pd.concat([meta_df, sentiment_df], axis=1)

    def fit_transform(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
        """
        FITS vectorizer & scaler on TRAIN set and extracts combined feature matrix.
        """
        clean_df = self.clean_data(df)

        # 1. Fit TF-IDF on full text
        text_matrix = self.vectorizer.fit_transform(clean_df['clean_full_text']).toarray()
        self.num_text_features = text_matrix.shape[1]

        # 2. Extract & Fit Scaler on Numerical Metadata
        metadata_df = self.extract_metadata_features(clean_df)
        scaled_metadata = self.scaler.fit_transform(metadata_df)
        self.num_dense_features = scaled_metadata.shape[1]

        # 3. Stack text TF-IDF + metadata features horizontally
        X = np.hstack((text_matrix, scaled_metadata))
        y = clean_df['is_asshole'].values.astype(int) if 'is_asshole' in clean_df.columns else None

        print(f"=== Metadata-Enhanced Training Pipeline Fitted ===")
        print(f"Total Samples    : {X.shape[0]}")
        print(f"TF-IDF Features  : {self.num_text_features}")
        print(f"Metadata Features: {self.num_dense_features}")
        print(f"Combined Shape X : {X.shape}\n")

        return clean_df, X, y

    def transform(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
        """
        TRANSFORMS Test/Val/Inference data using ALREADY FITTED vectorizer and scaler.
        """
        clean_df = self.clean_data(df)

        # 1. Transform text via existing vectorizer
        text_matrix = self.vectorizer.transform(clean_df['clean_full_text']).toarray()

        # 2. Extract & Transform metadata via existing scaler
        metadata_df = self.extract_metadata_features(clean_df)
        scaled_metadata = self.scaler.transform(metadata_df)

        # 3. Stack features
        X = np.hstack((text_matrix, scaled_metadata))
        y = clean_df['is_asshole'].values.astype(int) if 'is_asshole' in clean_df.columns else None

        return clean_df, X, y


if __name__ == "__main__":
    # Quick sanity test execution
    sample_data = {
        'id': ['101', '102'],
        'timestamp': [1672531199, 1672617599],
        'title': ['AITA for turning off the Wi-Fi?', 'AITA for eating my roomies cake?'],
        'body': ['I turned off the router because my roommate was screaming.', 'I ate the last slice without asking.'],
        'edited': [False, 1672618000],
        'verdict': ['NTA', 'YTA'], # Kept in df, but strictly excluded from X
        'score': [450, 12],
        'num_comments': [89, 154],
        'is_asshole': [0, 1]
    }
    
    df_sample = pd.DataFrame(sample_data)
    
    preprocessor = AITAMetadataPreprocessor(max_features=100)
    clean_df, X, y = preprocessor.fit_transform(df_sample)
    
    print("Sanity Check Output Matrix shape:", X.shape)
    print("Sanity Check Target Labels:", y)