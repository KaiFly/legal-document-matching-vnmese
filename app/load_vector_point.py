from datetime import datetime

import openpyxl
import pandas as pd
import json
from qdrant_client import models, QdrantClient
from sentence_transformers import SentenceTransformer

from sentence_transformer import vectorize, load_weighted_model, load_sentence_transformer_model
from legal_identity_extract import LegalIdentity
from legal_identity_matching import ExtendedLegalIdentity


def read_data_from_file(filename):
    legal_array = []
    df = pd.read_excel(filename, engine='openpyxl')
    for row in df.itertuples(index=False):
        id = str(row.ID)
        name = row.TEN_CHINH
        try:
            dob_date = datetime.strptime(str(row.NGAY_SINH), '%Y-%m-%d %H:%M:%S')
            dob = str(dob_date.date())
        except:
            dob = row.NGAY_SINH
            print(row.NGAY_SINH)
        gender = str(row.GIOI_TINH)
        nationality = str(row.QUOC_TICH_GOC_ID) if row.QUOC_TICH_GOC_ID is not None and str(row.QUOC_TICH_GOC_ID) != 'nan' else ''
        place_of_origin = str(row.NOI_SINH) if row.NOI_SINH is not None and str(row.NOI_SINH) != 'nan' else ''
        place_of_residence = str(row.DIA_CHI_THUONG_TRU) if row.DIA_CHI_THUONG_TRU is not None and str(row.DIA_CHI_THUONG_TRU) != 'nan' else ''
        hometown = str(row.QUE_QUAN) if row.QUE_QUAN is not None and str(row.QUE_QUAN) != 'nan' else ''
        occupation = str(row.NGHE_NGHIEP) if row.NGHE_NGHIEP is not None and str(row.NGHE_NGHIEP) != 'nan' else ''

        info = ExtendedLegalIdentity(
            text="",
            id_number=id,
            full_name=name,
            dob=dob,
            gender=gender,
            nationality=nationality,
            place_of_origin=place_of_origin,
            place_of_residence=place_of_residence,
            hometown=hometown,
            occupation=occupation
        )
        legal_array.append(info)

    return legal_array


def load_to_vector_points_v2(legal_array: [], collection_name):
    config_path = "config_li.json"
    config = ExtendedLegalIdentity.load_config_from_json(config_path)

    client = QdrantClient(url="http://localhost:6333")
    # client.set_model("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    # encoder = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

    vectorizer, svd = load_weighted_model(model_path='models', version='v1')
    tokenizer, model = load_sentence_transformer_model(model_name='sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')

    points = []
    for idx, doc in enumerate(legal_array):
        vector = vectorize(identity_info=doc, vectorizer=vectorizer, svd=svd, tokenizer=tokenizer, model=model, **config)
        payload = doc.to_dict(include_vector=False)
        point_struct = models.PointStruct(
            id=idx,
            vector=vector,
            payload=payload
        )
        points.append(point_struct)
        print('idx: ' + str(idx) + ', payload: ' + str(payload))

    client.upload_points(
        collection_name=collection_name,
        points=points,
    )

def load_to_vector_points(legal_array: []):
    metadata = []
    documents = []

    for legal in legal_array:
        documents.append(legal.pop("id") + ";" + legal.pop("name") + ";" + legal.pop("place_of_origin") + ";" + legal.pop("place_of_residence"))
        metadata.append(legal)

    client = QdrantClient(url="http://localhost:6333")
    client.set_model("sentence-transformers/all-MiniLM-L6-v2")
    client.set_sparse_model("prithivida/Splade_PP_en_v1")

    client.add(
        collection_name="legal_collection",
        documents=documents,
        metadata=metadata,
        parallel=0,  # Use all available CPU cores to encode data.
        # Requires wrapping code into if __name__ == '__main__' block
    )


def search(self, text: str):
    search_result = self.qdrant_client.query(
        collection_name=self.collection_name,
        query_text=text,
        query_filter=None,  # If you don't want any filters for now
        limit=5,  # 5 the closest results
    )
    # `search_result` contains found vector ids with similarity scores
    # along with the stored payload

    # Select and return metadata
    metadata = [hit.metadata for hit in search_result]
    return metadata

class NeuralSearcher:
    def __init__(self, collection_name):
        self.collection_name = collection_name
        # initialize Qdrant client
        self.qdrant_client = QdrantClient("http://localhost:6333")
        self.config = ExtendedLegalIdentity.load_config_from_json("config_li.json")

    def seach_by_legal_identity(self, legal_info: ExtendedLegalIdentity, vectorizer=None, svd=None, tokenizer=None, model=None):
        vector = vectorize(identity_info=legal_info, vectorizer=vectorizer, svd=svd, tokenizer=tokenizer, model=model, **self.config)
        search_result = self.qdrant_client.query_points(
            collection_name=self.collection_name,
            query=vector,
            query_filter=None,  # If you don't want any filters for now
            limit=5,  # 5 the most closest results is enough
        ).points
        payloads = [(hit.payload, hit.score) for hit in search_result]
        return payloads

    def search(self, text: str):
        # Convert text query into vector
        vector = self.model.encode(text).tolist()

        # Use `vector` for search for closest vectors in the collection
        search_result = self.qdrant_client.query_points(
            collection_name=self.collection_name,
            query=vector,
            query_filter=None,  # If you don't want any filters for now
            limit=5,  # 5 the most closest results is enough
        ).points
        # `search_result` contains found vector ids with similarity scores along with the stored payload
        # In this function you are interested in payload only
        payloads = [(hit.payload, hit.score) for hit in search_result]
        return payloads

if __name__ == '__main__':

    # # #test load data
    legal_dataset = read_data_from_file('/Users/thuyenhx/Downloads/DSDT.xlsx.xlsx')
    # # legal_dataset = read_data_from_file('/Users/thuyenhx/Documents/data_sample_matching.xlsx')
    load_to_vector_points_v2(legal_dataset, "legal_collection_test3")
    # load_to_vector_points_v2(legal_dataset[0:10], "legal_collection_test3")

    # legal_info = ExtendedLegalIdentity(
    #     text="",
    #     id_number="6000214868",
    #     full_name="Nguyễn Văn Son",
    #     dob="1990-01-01",
    #     gender="m",
    #     nationality="260",
    #     place_of_origin="Xã Nam Lộc, huyện Nam Đàn, tỉnh Nghệ An",
    #     place_of_residence=" Xóm 3, xã Nam Lộc, huyện Nam Đàn, Nghệ An",
    #     hometown="403",
    #     occupation="Lao dong tu do"
    # )
    # neural_searcher = NeuralSearcher(collection_name="legal_collection_test3")
    # result = neural_searcher.seach_by_legal_identity(legal_info)
    # for info, score in result:
    #     similar_info = ExtendedLegalIdentity(
    #         text="",
    #         id_number=info['id_number'],
    #         full_name=info['full_name'],
    #         dob=info['dob'],
    #         gender=info['gender'],
    #         nationality=info['nationality'],
    #         place_of_origin=info['place_of_origin'],
    #         place_of_residence=info['place_of_residence'],
    #         hometown=info['hometown'],
    #         occupation=info['occupation'],
    #         similar_score=score,
    #     )
    #     semantic_score = legal_info.compare(similar_info)
    #     print('semantic_score: ' + str(semantic_score) + ', similar_score: ' + str(score) + ', similar_info: ' + str(similar_info))

    # client = QdrantClient(url="http://localhost:6333")
    # # client.set_model("sentence-transformers/all-MiniLM-L6-v2")
    # # client.set_sparse_model("prithivida/Splade_PP_en_v1")
    # client.set_model("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    # encoder = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    # client.create_collection(
    #     collection_name="legal_collection_test2",
    #     vectors_config=models.VectorParams(
    #         size=encoder.get_sentence_embedding_dimension(),  # Vector size is defined by used model
    #         distance=models.Distance.COSINE,
    #     ),
    # )
