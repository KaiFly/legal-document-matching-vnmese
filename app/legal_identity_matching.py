from legal_identity_extract import LegalIdentity

import os
from datetime import datetime
import pandas as pd
import numpy as np
import re
import unicodedata
import joblib
from typing import Dict, Optional, List
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
import json

class ExtendedLegalIdentity(LegalIdentity):
    def __init__(
        self,
        text,
        id_number=None,
        full_name=None,
        dob=None,
        gender=None,
        nationality=None,
        place_of_origin=None,
        place_of_residence=None,
        diaspora_id=None,
        hometown=None,
        occupation=None,
        similar_score=None,
        semantic_score=None,
    ):
        super().__init__(
            text,
            id_number,
            full_name,
            dob,
            gender,
            nationality,
            place_of_origin,
            place_of_residence
        )
        self.diaspora_id = diaspora_id or ""
        self.hometown = hometown or ""
        self.occupation = occupation or ""
        self.similar_score = similar_score or ""
        self.semantic_score = semantic_score or ""

    def __str__(self):
        base_info = super().__str__()
        return base_info + str(self.diaspora_id) + ";" + str(self.hometown) + ";" + str(self.occupation)+ ";" + str(self.similar_score)+ ";" + str(self.semantic_score)

   
    @staticmethod
    def load_config_from_json(path: str) -> dict:
        """
        Tải cấu hình từ file JSON (ví dụ: config cho vector hóa).
        """
        with open(path, "r", encoding="utf-8") as f:
            config = json.load(f)
        return config
    
    # ==============================
    # === 1. Preprocessing Utils ===
    # ==============================
    @staticmethod
    def remove_diacritics(text):
        if text is None or not isinstance(text, str):
            return ""
        text = ''.join(c for c in unicodedata.normalize('NFKD', text) if not unicodedata.combining(c))
        return text.replace('đ', 'd').replace('Đ', 'D')

    @classmethod
    def load_and_preprocess_data(cls, df: pd.DataFrame, keep_original=False) -> pd.DataFrame:
        from opencc import OpenCC
        opencc = OpenCC('t2s')

        text_fields = ['TEN_CHINH', 'NGAY_SINH', 'NOI_SINH', 'DIA_CHI_THUONG_TRU', 'QUE_QUAN', 'NGHE_NGHIEP']
        for col in text_fields:
            df[col + '_norm'] = df[col].fillna('').astype(str)
            df[col + '_norm'] = df[col + '_norm'].apply(cls.remove_diacritics)
            df[col + '_norm'] = df[col + '_norm'].str.replace(r"[^\u00C0-\u024F\u4e00-\u9fa5\u3130-\u318F\uAC00-\uD7AFa-zA-Z0-9\s,.!?@()]+", "", regex=True)
            df[col + '_norm'] = df[col + '_norm'].str.replace(r"\\s+", " ", regex=True)
            if col in ['NOI_SINH', 'DIA_CHI_THUONG_TRU', 'QUE_QUAN']:
                df[col + '_norm'] = df[col + '_norm'].str.replace(r'[-–]', ',', regex=True)
            if col == 'TEN_CHINH':
                df[col + '_norm'] = df.apply(lambda x: opencc.convert(x[col + '_norm']) if x.get("VIET_KIEU", 0) == 1 else x[col + '_norm'], axis=1)
            if col == 'NGAY_SINH':
                df[col + '_norm'] = pd.to_datetime(df[col], errors='coerce').dt.strftime('%Y%m%d')

        numeric_columns = ["ID", "QUOC_TICH_HN_ID", "MA_NOI_SINH", "VIET_KIEU"]
        for col in numeric_columns:
            df[col + '_norm'] = pd.to_numeric(df[col], errors='coerce')

        list_fields_norm = [col + '_norm' for col in numeric_columns + text_fields]
        if keep_original:
            return df[numeric_columns + text_fields + list_fields_norm]
        else:
            df = df[list_fields_norm]
            df.columns = numeric_columns + text_fields
            return df

    # =================================
    # === 2. Weighted Text Building ===
    # =================================
    @staticmethod
    def apply_field_weighting(row: pd.Series, field_weights: dict) -> str:
        parts = []
        for field, weight in field_weights.items():
            value = str(row.get(field, "")).strip()
            if not value:
                continue
            if weight > 0:
                parts.append((value + " ") * weight)
            else:
                parts.append(value)
        return " ".join(parts).strip()

    # ========================================
    # === 3. Load TF-IDF + SVD Model =========
    # ========================================
    @staticmethod
    def load_weighted_model(model_path: str = "models/", version: str = "v1"):
        tfidf = joblib.load(f"{model_path}/LI-weighted-TFIDF-{version}.pkl")
        svd = joblib.load(f"{model_path}/LI-weighted-TFIDF-SVD-{version}.pkl")
        return tfidf, svd

    # ==========================================
    # === 4. Load Sentence Transformer Model ===
    # ==========================================
    @staticmethod
    def load_sentence_transformer_model(model_name: str = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'):
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
        return tokenizer, model

    @staticmethod
    def mean_pooling(model_output, attention_mask):
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, dim=1) / \
               torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)

    @classmethod
    def compute_sentence_embeddings(cls, sentences: List[str], tokenizer, model, device: str = 'cpu') -> np.ndarray:
        model.to(device)
        encoded_input = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt').to(device)
        with torch.no_grad():
            model_output = model(**encoded_input)
        embeddings = cls.mean_pooling(model_output, encoded_input['attention_mask'])
        return embeddings.cpu().numpy()

    # ==================================
    # === 5. Ensemble TF-IDF + BERT ===
    # ==================================
    @staticmethod
    def ensemble_vectorization(X_tfidf_reduced: np.ndarray, X_transformer: np.ndarray, alpha: float = 0.5, beta: float = 1.0) -> np.ndarray:
        X_tfidf_norm = normalize(X_tfidf_reduced, norm='l2', axis=1)
        X_st_norm = normalize(X_transformer, norm='l2', axis=1)
        return np.hstack([alpha * X_tfidf_norm, beta * X_st_norm])

    # ================================
    # === MAIN: Vectorization API ===
    # ================================
    def vectorize(
        self,
        model_dir: str = "models",
        model_version: str = "v1",
        field_weights: Optional[Dict[str, int]] = None,
        alpha: float = 0.5,
        beta: float = 1.0,
        device: str = 'cpu',
        transformer_model_name: str = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
    ) -> np.ndarray:
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

        # Create record dictionary
        record_dict = {
            "ID": self.id_number,
            "TEN_CHINH": self.full_name,
            "NGAY_SINH": self.dob,
            "GIOI_TINH": self.gender,
            "QUOC_TICH_HN_ID": self.nationality,
            "MA_NOI_SINH": None,
            "VIET_KIEU": 0,
            "NOI_SINH": self.place_of_origin,
            "QUE_QUAN": self.hometown,
            "DIA_CHI_THUONG_TRU": self.place_of_residence,
            "NGHE_NGHIEP": self.occupation
        }

        # Pipeline: preprocess → weighted text → vectorize
        df = self.load_and_preprocess_data(pd.DataFrame([record_dict]), keep_original=False)
        df["weighted_text"] = df.apply(lambda row: self.apply_field_weighting(row, field_weights), axis=1)

        vectorizer, svd = self.load_weighted_model(model_path=model_dir, version=model_version)
        X_tfidf = vectorizer.transform(df["weighted_text"])
        X_tfidf_reduced = svd.transform(X_tfidf)

        tokenizer, model = self.load_sentence_transformer_model(model_name=transformer_model_name)
        X_st = self.compute_sentence_embeddings(df["weighted_text"].tolist(), tokenizer, model, device=device)

        return self.ensemble_vectorization(X_tfidf_reduced, X_st, alpha=alpha, beta=beta)[0]
    def compare(
        self,
        other: 'ExtendedLegalIdentity',
        model_dir: str = "models/",
        model_version: str = "v1",
        field_weights: Optional[Dict[str, int]] = None,
        alpha: float = 0.5,
        beta: float = 1.0,
        device: str = 'cpu',
        transformer_model_name: str = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
    ) -> float:
        """
        So sánh cosine similarity giữa 2 bản ghi định danh.
        """
        vec1 = getattr(self, "vector_cache", None) or self.vectorize(
            model_dir, model_version, field_weights, alpha, beta, device, transformer_model_name
        )
        vec2 = getattr(other, "vector_cache", None) or other.vectorize(
            model_dir, model_version, field_weights, alpha, beta, device, transformer_model_name
        )
        return cosine_similarity([vec1], [vec2])[0][0]

    def to_dict(self, include_vector=True) -> Dict:
        """
        Xuất dữ liệu LegalIdentity thành dict để lưu trữ hoặc xuất API.
        """
        result = {
            "id_number": self.id_number,
            "full_name": self.full_name,
            "dob": self.dob,
            "gender": self.gender,
            "nationality": self.nationality,
            "place_of_origin": self.place_of_origin,
            "place_of_residence": self.place_of_residence,
            "diaspora_id": self.diaspora_id,
            "hometown": self.hometown,
            "occupation": self.occupation,
            "similar_score": self.similar_score,
            "semantic_score": self.semantic_score,
        }

        if include_vector:
            vector = getattr(self, "vector_cache", None)
            if vector is None:
                vector = self.vectorize()
                self.vector_cache = vector
            result["embedding"] = vector.tolist()
        return result

if __name__ == "__main__":
    LegalIdentity_1 = ExtendedLegalIdentity(
        text="",
        id_number="6000214863",
        full_name="Nguyễn Văn Sơn",
        dob="1990-01-01",
        gender="M",
        nationality="260.0",
        place_of_origin="Xã Nam Lộc, huyện Nam Đàn, tỉnh Nghệ An",
        place_of_residence=" Xóm 3, xã Nam Lộc, huyện Nam Đàn, Nghệ An",
        hometown="403",
        occupation="Lao dong tu do"
    )

    LegalIdentity_2 = ExtendedLegalIdentity(
        text="",
        id_number="6000214868",
        full_name="Nguyễn Văn Son",
        dob="1990-01-01",
        gender="m",
        nationality="260",
        place_of_origin="Xã Nam Lộc, huyện Nam Đàn, tỉnh Nghệ An",
        place_of_residence=" Xóm 3, xã Nam Lộc, huyện Nam Đàn, Nghệ An",
        hometown="403",
        occupation="Lao dong tu do"
    )

    config_path = "config_li.json"
    config = ExtendedLegalIdentity.load_config_from_json(config_path)
    vector_1 = LegalIdentity_1.vectorize(**config)
    vector_2 = LegalIdentity_2.vectorize(**config)
    result = LegalIdentity_1.compare(LegalIdentity_2, model_dir='../models')
    print(result)