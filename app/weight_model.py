import os
import joblib
import pandas as pd
import json
import unicodedata
from opencc import OpenCC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from typing import Dict, Optional

class WeightModel:
    """
    A class to handle the retraining of TF-IDF and SVD models for text vectorization
    with field weighting. Models are retrained based on a DataFrame and saved for reuse.
    """

    def __init__(self, model_dir: str = "models/", version: str = "v1", config_path: Optional[str] = None):
        """
        Initializes the WeightModel instance.
        Parameters
        ----------
        model_dir : str
            Directory where models will be saved.
        version : str
            Version string to append to the saved filenames.
        config_path : str, optional
            Path to the configuration JSON file. If provided, the config will be loaded.
        train_df : DataFrame
            The list of Legal Identity same format as DSDT.xlsx
        """
        self.model_dir = model_dir
        self.version = version
        self.config = None
        self.train_df = None
        # Load configuration if path is provided
        if config_path:
            self.load_config_from_json(config_path)

    def load_config_from_json(self, path: str) -> dict:
        """
        Loads configuration from a JSON file.
        """
        with open(path, "r", encoding="utf-8") as f:
            self.config = json.load(f)
        return self.config

    @staticmethod
    def remove_diacritics(text: str) -> str:
        """
        Removes accents from the text.
        """
        if not isinstance(text, str):
            return ""
        text = ''.join(c for c in unicodedata.normalize('NFKD', text) if not unicodedata.combining(c))
        return text.replace('đ', 'd').replace('Đ', 'D')

    def load_and_preprocess_data(self, train_df: pd.DataFrame, keep_original=False) -> pd.DataFrame:
        """
        Preprocess the dataset (train_df), normalize text fields, and clean the data.
        """
        self.train_df = train_df  # Store train_df as an attribute of the class

        opencc = OpenCC('t2s')

        # List of text columns that need preprocessing
        text_fields = ['TEN_CHINH', 'NGAY_SINH', 'NOI_SINH', 'DIA_CHI_THUONG_TRU', 'QUE_QUAN', 'GIOI_TINH', 'NGHE_NGHIEP']
        
        # Preprocess text columns
        for col in text_fields:
            self.train_df[col + '_norm'] = self.train_df[col].fillna('').astype(str)
            self.train_df[col + '_norm'] = self.train_df[col + '_norm'].apply(self.remove_diacritics)
            self.train_df[col + '_norm'] = self.train_df[col + '_norm'].str.replace(r"[^\u00C0-\u024F\u4e00-\u9fa5\u3130-\u318F\uAC00-\uD7AFa-zA-Z0-9\s,.!?@()]+", "", regex=True)
            self.train_df[col + '_norm'] = self.train_df[col + '_norm'].str.replace(r"\\s+", " ", regex=True)
            if col in ['NOI_SINH', 'DIA_CHI_THUONG_TRU', 'QUE_QUAN']:
                self.train_df[col + '_norm'] = self.train_df[col + '_norm'].str.replace(r'[-–]', ',', regex=True)
            if col == 'TEN_CHINH':
                self.train_df[col + '_norm'] = self.train_df.apply(lambda x: opencc.convert(x[col + '_norm']) if x.get("VIET_KIEU", 0) == 1 else x[col + '_norm'], axis=1)
            if col == 'NGAY_SINH':
                self.train_df[col + '_norm'] = pd.to_datetime(self.train_df[col], errors='coerce').dt.strftime('%Y%m%d')
            if col == 'GIOI_TINH':
                self.train_df[col + '_norm'] = self.train_df[col + '_norm'].apply(lambda x: "Male" if x.upper() == "M" else ("Female" if x.upper() == "F" else "Unknown"))
                
        # Numeric columns processing
        numeric_columns = ["ID", "QUOC_TICH_HN_ID", "MA_NOI_SINH", "VIET_KIEU"]
        for col in numeric_columns:
            self.train_df[col + '_norm'] = pd.to_numeric(self.train_df[col], errors='coerce')

        list_fields_norm = [col + '_norm' for col in numeric_columns + text_fields]
        
        # Return the DataFrame with original or processed columns
        if keep_original:
            return self.train_df[numeric_columns + text_fields + list_fields_norm]
        else:
            self.train_df = self.train_df[list_fields_norm]
            self.train_df.columns = numeric_columns + text_fields
            return self.train_df

    def apply_field_weighting(self, row: pd.Series, field_weights: dict) -> str:
        """
        Combine row fields into a single weighted text string based on field_weights.
        """
        parts = []
        if field_weights is None:
            field_weights = {
                "ID": 3,
                "TEN_CHINH": 3,
                "NGAY_SINH": 2,
                "DIA_CHI_THUONG_TRU": 2,
                "NOI_SINH": 1,
                "QUE_QUAN": 1,
                "GIOI_TINH": 0,
                "NGHE_NGHIEP": 0
            }

        for field, weight in field_weights.items():
            value = str(row.get(field, "")).strip()
            if value:
                if weight > 0:
                    parts.append((value + " ") * weight)
                else:
                    parts.append(value)
        return " ".join(parts).strip()

    def generate_tfidf_matrix(self, df: pd.DataFrame, text_col: str = "weighted_text"):
        """
        Generate the TF-IDF matrix from the provided DataFrame same params with noteboook
        """
        tfidf = TfidfVectorizer(
            strip_accents='unicode',
            lowercase=True,
            stop_words=None,
            token_pattern=r'(?u)\b\w\w+\b',
            ngram_range=(1, 1),
            min_df=1,
            max_df=1.0,
            max_features=None,
            sublinear_tf=False,
            norm='l2',
            use_idf=True,
            smooth_idf=True,
        )
        tfidf_matrix = tfidf.fit_transform(df[text_col])
        return tfidf, tfidf_matrix

    def retrain_tfidf_and_svd(self, field_weights: dict):
        """
        Retrains the TF-IDF vectorizer and SVD model with the provided dataset and field weights.
        Saves the updated models for future use.
        Parameters
        ----------
        field_weights : dict
            A dictionary containing the field weights for the training data.
        """
        # Apply field weighting to create the "weighted_text" column
        self.train_df["weighted_text"] = self.train_df.apply(lambda row: self.apply_field_weighting(row, field_weights), axis=1)

        # Generate the TF-IDF matrix
        vectorizer, X_tfidf = self.generate_tfidf_matrix(self.train_df, text_col="weighted_text")
        print(f"TF-IDF matrix shape: {X_tfidf.shape}")

        # Perform dimensionality reduction with TruncatedSVD
        svd = TruncatedSVD(n_components=200, random_state=0)
        X_tfidf_reduced = svd.fit_transform(X_tfidf)
        print(f"TF-IDF matrix dimension reduced shape: {X_tfidf_reduced.shape}")

        # Save the updated models
        os.makedirs(self.model_dir, exist_ok=True)
        joblib.dump(vectorizer, f"{self.model_dir}/LI-weighted-TFIDF-{self.version}.pkl")
        joblib.dump(svd, f"{self.model_dir}/LI-weighted-TFIDF-SVD-{self.version}.pkl")
        print(f"Models saved in {self.model_dir} with version {self.version}.")
        
    def load_tfidf_and_svd(self):
        """
        Loads the previously saved TF-IDF and SVD models from the specified model directory.

        Returns
        -------
        tuple
            The loaded TF-IDF vectorizer and SVD model.
        """
        vectorizer = joblib.load(f"{self.model_dir}/LI-weighted-TFIDF-{self.version}.pkl")
        svd = joblib.load(f"{self.model_dir}/LI-weighted-TFIDF-SVD-{self.version}.pkl")
        return vectorizer, svd
    
if __name__ == "__main__":
    # Initialize the sample data as train set
    train_df = pd.read_excel("dataset/DSDT.xlsx")  # Replace with actual dataset loading

    # Define field weights for retraining
    field_weights = {
        "ID": 5,
        "TEN_CHINH": 4,
        "NGAY_SINH": 3,
        "DIA_CHI_THUONG_TRU": 3,
        "NOI_SINH": 1,
        "QUE_QUAN": 1,
        "GIOI_TINH": 1,
        "NGHE_NGHIEP": 0
    }

    # Define model version as suffix of model name 
    # -> Should has rule to map 1-1 with field_weights to save all versions
    # -> OR can Train and Replace the model to v2 for convinience 

    model = WeightModel(config_path="config_li.json", model_dir="models", version="v3")
    model.load_and_preprocess_data(train_df)
    
    # Retrain Model
    model.retrain_tfidf_and_svd(field_weights)