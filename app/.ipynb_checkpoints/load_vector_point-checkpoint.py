import openpyxl
import pandas as pd
import json
from qdrant_client import models, QdrantClient
from sentence_transformers import SentenceTransformer

def read_data_from_file(filename):
    legal_array = []
    df = pd.read_excel(filename, engine='openpyxl')
    for row in df.itertuples(index=False):
        id = str(row.ID)
        name = row.TEN_CHINH
        dob = str(row.NGAY_SINH)
        gender = str(row.GIOI_TINH)
        nationality = str(row.QUOC_TICH_GOC_ID)
        place_of_origin = str(row.NOI_SINH)
        place_of_residence = str(row.DIA_CHI_THUONG_TRU)
        info = {
            'id': id,
            'name': name,
            'dob': dob,
            'gender': gender,
            'nationality': nationality,
            'place_of_origin': place_of_origin,
            'place_of_residence': place_of_residence
        }
        legal_array.append(info)

    return legal_array


def load_to_vector_points_v2(legal_array: []):
    # metadata = []
    # documents = []
    #
    # for legal in legal_array:
    #     documents.append(legal.pop("id") + ";" + legal.pop("name") + ";" + legal.pop("place_of_origin") + ";" + legal.pop("place_of_residence"))
    #     metadata.append(legal)

    client = QdrantClient(url="http://localhost:6333")
    encoder = SentenceTransformer("all-MiniLM-L6-v2")
    client.upload_points(
        collection_name="legal_collection",
        points=[
            models.PointStruct(
                # id=idx, vector=encoder.encode(doc["id"] + ";" + doc["name"] + ";" + doc["place_of_origin"] + ";" + doc["place_of_residence"]).tolist(), payload=doc
                id=idx, vector=encoder.encode(doc["id"] + ";" + doc["name"]).tolist(), payload=doc
            )
            for idx, doc in enumerate(legal_array)
        ],
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
        # Initialize encoder model
        self.model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
        # initialize Qdrant client
        self.qdrant_client = QdrantClient("http://localhost:6333")

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
    # legal_dataset = read_data_from_file('/Users/thuyenhx/Downloads/DSDT.xlsx.xlsx')
    # load_to_vector_points_v2(legal_dataset)
    # load_to_vector_points_v2(legal_dataset[0:1000])

    neural_searcher = NeuralSearcher(collection_name="legal_collection")
    result1 = neural_searcher.search("")
    print('result1: ' + str(result1))

    # client.create_collection(
    #     collection_name="legal_collection",
    #     vectors_config=models.VectorParams(
    #         size=encoder.get_sentence_embedding_dimension(),  # Vector size is defined by used model
    #         distance=models.Distance.COSINE,
    #     ),
    # )

    # if not client.collection_exists("legal_collection"):
    #     client.create_collection(
    #         collection_name="legal_collection",
    #         vectors_config=client.get_fastembed_vector_params(),
    #         # comment this line to use dense vectors only
    #         sparse_vectors_config=client.get_fastembed_sparse_vector_params(),
    # )
