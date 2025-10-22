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

from legal_identity_matching import ExtendedLegalIdentity


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

def remove_diacritics(text):
    if text is None or not isinstance(text, str):
        return ""
    text = ''.join(c for c in unicodedata.normalize('NFKD', text) if not unicodedata.combining(c))
    return text.replace('đ', 'd').replace('Đ', 'D')

def load_and_preprocess_data(df: pd.DataFrame, keep_original=False) -> pd.DataFrame:
    from opencc import OpenCC
    opencc = OpenCC('t2s')

    text_fields = ['TEN_CHINH', 'NGAY_SINH', 'NOI_SINH', 'DIA_CHI_THUONG_TRU', 'QUE_QUAN', 'NGHE_NGHIEP']
    for col in text_fields:
        df[col + '_norm'] = df[col].fillna('').astype(str)
        # df[col + '_norm'] = df[col + '_norm'].apply(cls.remove_diacritics)
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

def vectorize(
    identity_info: ExtendedLegalIdentity,
    field_weights: Optional[Dict[str, int]] = None,
    alpha: float = 0.5,
    beta: float = 1.0,
    device: str = 'cpu',
    model_dir: str = 'models',
    model_version: str = 'v1',
    transformer_model_name: str = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2',
    vectorizer=None, svd=None, tokenizer=None, model=None
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
        "ID": identity_info.id_number,
        "TEN_CHINH": identity_info.full_name,
        "NGAY_SINH": identity_info.dob,
        "GIOI_TINH": identity_info.gender,
        "QUOC_TICH_HN_ID": identity_info.nationality,
        "MA_NOI_SINH": None,
        "VIET_KIEU": 0,
        "NOI_SINH": identity_info.place_of_origin,
        "QUE_QUAN": identity_info.hometown,
        "DIA_CHI_THUONG_TRU": identity_info.place_of_residence,
        "NGHE_NGHIEP": identity_info.occupation
    }

    # Pipeline: preprocess → weighted text → vectorize
    df = load_and_preprocess_data(pd.DataFrame([record_dict]), keep_original=False)
    df["weighted_text"] = df.apply(lambda row: apply_field_weighting(row, field_weights), axis=1)

    X_tfidf = vectorizer.transform(df["weighted_text"])
    X_tfidf_reduced = svd.transform(X_tfidf)

    X_st = compute_sentence_embeddings(df["weighted_text"].tolist(), tokenizer, model, device=device)

    return ensemble_vectorization(X_tfidf_reduced, X_st, alpha=alpha, beta=beta)[0]

def load_weighted_model(model_path: str = "models/", version: str = "v1"):
    tfidf = joblib.load(f"{model_path}/LI-weighted-TFIDF-{version}.pkl")
    svd = joblib.load(f"{model_path}/LI-weighted-TFIDF-SVD-{version}.pkl")
    return tfidf, svd

def load_sentence_transformer_model(model_name: str = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    return tokenizer, model

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, dim=1) / \
           torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)

def compute_sentence_embeddings(sentences: List[str], tokenizer, model, device: str = 'cpu') -> np.ndarray:
    model.to(device)
    encoded_input = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt').to(device)
    with torch.no_grad():
        model_output = model(**encoded_input)
    embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
    return embeddings.cpu().numpy()

def ensemble_vectorization(X_tfidf_reduced: np.ndarray, X_transformer: np.ndarray, alpha: float = 0.5, beta: float = 1.0) -> np.ndarray:
    X_tfidf_norm = normalize(X_tfidf_reduced, norm='l2', axis=1)
    X_st_norm = normalize(X_transformer, norm='l2', axis=1)
    return np.hstack([alpha * X_tfidf_norm, beta * X_st_norm])

def compare(
    first_identity: ExtendedLegalIdentity,
    second_identity: ExtendedLegalIdentity,
    field_weights: Optional[Dict[str, int]] = None,
    alpha: float = 0.5,
    beta: float = 1.0,
    device: str = 'cpu',
    vectorizer=None, svd=None, tokenizer=None, model=None
) -> float:
    """
    So sánh cosine similarity giữa 2 bản ghi định danh.
    """
    vec1 = getattr(first_identity, "vector_cache", None) or vectorize(
        first_identity, field_weights, alpha, beta, device, vectorizer=vectorizer, svd=svd, tokenizer=tokenizer, model=model
    )
    vec2 = getattr(second_identity, "vector_cache", None) or vectorize(
        second_identity, field_weights, alpha, beta, device, vectorizer=vectorizer, svd=svd, tokenizer=tokenizer, model=model
    )
    return cosine_similarity([vec1], [vec2])[0][0]

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

    vectorizer, svd = load_weighted_model(model_path='models', version='v1')
    tokenizer, model = load_sentence_transformer_model(model_name='sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')

    config_path = "config_li.json"
    config = ExtendedLegalIdentity.load_config_from_json(config_path)

    vector_1 = vectorize(identity_info=LegalIdentity_1, vectorizer=vectorizer, svd=svd, tokenizer=tokenizer, model=model, **config)
    vector_2 = vectorize(identity_info=LegalIdentity_2, vectorizer=vectorizer, svd=svd, tokenizer=tokenizer, model=model, **config)
    result = compare(LegalIdentity_1, LegalIdentity_2, vectorizer=vectorizer, svd=svd, tokenizer=tokenizer, model=model)
    print(result)