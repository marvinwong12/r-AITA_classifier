import os
import re
import nltk
import numpy as np
import pandas as pd
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import RobustScaler

# Download VADER lexicon on first run
try:
    nltk.data.find('sentiment/vader_lexicon')
except LookupError:
    nltk.download('vader_lexicon', quiet=True)


class AITAPreprocessor:
    def __init__(self, feature_type="tfidf", max_features=3000):
        """
        :param feature_type: 'tfidf' or 'transformer'
        :param max_features: Vocab size if using TF-IDF
        """
        self.feature_type = feature_type.lower()
        self.max_features = max_features
        
        # 1. Text Representation Strategy
        if self.feature_type == "tfidf":
            self.vectorizer = TfidfVectorizer(
                max_features=max_features,
                ngram_range=(1, 2),
                stop_words='english',
                sublinear_tf=True
            )
        elif self.feature_type == "transformer":
            try:
                from sentence_transformers import SentenceTransformer
                print("Loading SentenceTransformer model ('all-MiniLM-L6-v2')...")
                self.encoder = SentenceTransformer("all-MiniLM-L6-v2")
            except ImportError:
                raise ImportError(
                    "sentence-transformers is not installed. Run `pip install sentence-transformers`."
                )
        else:
            raise ValueError("feature_type must be 'tfidf' or 'transformer'")

        self.sia = SentimentIntensityAnalyzer()
        self.scaler = RobustScaler()

    def clean_text(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Isolates and cleans the body text field. Drops deleted/empty posts during training.
        """
        df = df.copy()
        
        # Ensure body column exists
        if 'body' not in df.columns:
            df['body'] = ''
        else:
            df['body'] = df['body'].fillna('')

        # Filter out missing or deleted/removed body text for dataset training
        if 'is_asshole' in df.columns:
            has_no_body = df['body'].isna() | (df['body'].str.strip() == '')
            is_deleted_or_removed = df['body'].astype(str).str.contains(
                r'\[deleted\]|\[removed\]', case=False, regex=True
            )
            valid_mask = ~(has_no_body | is_deleted_or_removed)
            df = df[valid_mask].reset_index(drop=True)

        # Standardize whitespace on body text
        df['clean_body'] = df['body'].apply(
            lambda x: re.sub(r'\s+', ' ', str(x)).strip()
        )
        return df

    def extract_sentiment(self, text_series: pd.Series) -> pd.DataFrame:
        """
        Extracts VADER sentiment scores (neg, neu, pos, compound) directly from story body.
        """
        sentiments = text_series.apply(lambda x: self.sia.polarity_scores(x))
        return pd.DataFrame(sentiments.tolist(), index=text_series.index)

    def extract_stylometrics(self, text_series: pd.Series) -> pd.DataFrame:
        """
        Extracts structural and emotional writing signals directly from story body:
        - Character and word counts
        - Shouting ratio (CAPS count)
        - Punctuation density (! and ?)
        - Self-justification vs. Other focus (I/me/my vs. he/she/they)
        """
        features = pd.DataFrame(index=text_series.index)

        # Length Metrics
        features['char_length'] = text_series.apply(len)
        features['word_count'] = text_series.apply(lambda x: len(x.split()))

        # Emotional/Tone Signals
        features['caps_ratio'] = text_series.apply(
            lambda x: sum(1 for c in x if c.isupper()) / (len(x) + 1)
        )
        features['exclamation_count'] = text_series.apply(lambda x: x.count('!'))
        features['question_count'] = text_series.apply(lambda x: x.count('?'))

        # Pronoun Ratios
        features['i_pronouns'] = text_series.apply(
            lambda x: len(re.findall(r'\b(i|me|my|mine|myself)\b', x, re.I))
        )
        features['they_pronouns'] = text_series.apply(
            lambda x: len(re.findall(r'\b(he|him|his|she|her|they|them|their)\b', x, re.I))
        )

        return features

    def _extract_text_representation(self, text_series: pd.Series, is_train: bool) -> np.ndarray:
        """
        Extracts TF-IDF or Dense Transformer Embeddings from the body text.
        """
        if self.feature_type == "tfidf":
            if is_train:
                return self.vectorizer.fit_transform(text_series).toarray()
            return self.vectorizer.transform(text_series).toarray()
            
        elif self.feature_type == "transformer":
            return self.encoder.encode(
                text_series.tolist(), 
                show_progress_bar=False, 
                batch_size=64
            )

    def fit_transform(self, df: pd.DataFrame):
        """
        FITS vectorizer and scaler on TRAIN body text data.
        """
        clean_df = self.clean_text(df)
        
        # 1. Main Text Representation (TF-IDF or Embeddings)
        text_features = self._extract_text_representation(clean_df['clean_body'], is_train=True)
        
        # 2. Body Sentiment & Stylometrics
        sentiment_df = self.extract_sentiment(clean_df['clean_body'])
        stylometrics_df = self.extract_stylometrics(clean_df['clean_body'])
        
        # Combine non-TF-IDF numerical features and scale them
        numeric_features = pd.concat([sentiment_df, stylometrics_df], axis=1)
        scaled_numeric = self.scaler.fit_transform(numeric_features)
        
        # Stack into final matrix X
        X = np.hstack((text_features, scaled_numeric))
        y = clean_df['is_asshole'].values if 'is_asshole' in clean_df.columns else None
        
        print(f"Body-Derived Training Matrix Shape: {X.shape}")
        return clean_df, X, y

    def transform(self, df: pd.DataFrame):
        """
        TRANSFORMS Test/Val/Inference data using ALREADY FITTED components.
        """
        clean_df = self.clean_text(df)
        
        # 1. Main Text Representation
        text_features = self._extract_text_representation(clean_df['clean_body'], is_train=False)
        
        # 2. Body Sentiment & Stylometrics
        sentiment_df = self.extract_sentiment(clean_df['clean_body'])
        stylometrics_df = self.extract_stylometrics(clean_df['clean_body'])
        
        numeric_features = pd.concat([sentiment_df, stylometrics_df], axis=1)
        scaled_numeric = self.scaler.transform(numeric_features)
        
        X = np.hstack((text_features, scaled_numeric))
        y = clean_df['is_asshole'].values if 'is_asshole' in clean_df.columns else None
        
        return clean_df, X, y


if __name__ == "__main__":
    raw_data_path = "data/raw/aita_10k_sample.csv"
    if os.path.exists(raw_data_path):
        print("--- Testing Pure Body Pipeline ---")
        raw_df = pd.read_csv(raw_data_path)
        
        prep = AITAPreprocessor(feature_type="tfidf", max_features=3000)
        clean_df, X, y = prep.fit_transform(raw_df)
        print(f"Feature Matrix (X) shape: {X.shape}")
        print("Pipeline verified! Features generated strictly from the body text.")