
import streamlit as st
import pandas as pd
from io import StringIO
import datetime

from sentence_transformer import load_weighted_model, load_sentence_transformer_model, compare
from legal_identity_matching import ExtendedLegalIdentity
from load_vector_point import NeuralSearcher
from legal_identity_extract import LegalIdentity

@st.cache_resource
def get_neural_searcher(collection_name="legal_collection_test3"):
    return NeuralSearcher(collection_name=collection_name)

@st.cache_data
def get_weighted_model():
    return load_weighted_model(model_path='models', version='v1')

@st.cache_data
def get_sentence_transformer_model():
    return load_sentence_transformer_model(model_name='sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')

def detect_text_from_binary(bytes_data):
    """Detects text in the file."""
    from google.cloud import vision

    client = vision.ImageAnnotatorClient()
    image = vision.Image(content=bytes_data)

    response = client.text_detection(image=image)
    texts = response.text_annotations
    return texts[0].description

def text_ocr_from_file():
    container = st.container(border=True)
    with container:
        uploaded_file = st.file_uploader("Choose a file")
        if uploaded_file is not None:
            bytes_data = uploaded_file.getvalue()
            if bytes_data is not None:
                st.image(bytes_data, caption="CCCD Test Image", width=100)
                result = detect_text_from_binary(bytes_data)
                identity = LegalIdentity(result)

                if identity is not None:
                    col1, col2 = st.columns(2)
                    with col1:
                        id_number_ocr = st.text_input("ID Number", identity.id_number, key="id_number_ocr")
                        full_name_ocr = st.text_input("Name", identity.full_name, key="full_name_ocr")
                        dob_ocr = st.date_input("Date of birth", datetime.date(2025, 3, 1), format='DD/MM/YYYY', key="dob_ocr")
                        gender_ocr = st.text_input("Gender", identity.gender, key="gender_ocr")
                    with col2:
                        nationality_ocr = st.text_input("Nationality", identity.nationality, key="nationality_ocr")
                        place_of_origin_ocr = st.text_input("Place of Origin", identity.place_of_origin, key="place_of_origin_ocr")
                        place_of_residence_ocr = st.text_input("Place of Residence", identity.place_of_residence, key="place_of_residence_ocr")

                neural_searcher = get_neural_searcher(collection_name="legal_collection_test3")
                vectorizer, svd = get_weighted_model()
                tokenizer, model = get_sentence_transformer_model()

                click_search_ocr = st.button("Search", type="primary", key="click_search_ocr")
                if click_search_ocr:
                    legal_info = ExtendedLegalIdentity(
                        text='',
                        id_number=id_number_ocr,
                        full_name=full_name_ocr,
                        dob=dob_ocr,
                        gender=gender_ocr,
                        nationality=nationality_ocr,
                        place_of_origin=place_of_origin_ocr,
                        place_of_residence=place_of_residence_ocr,
                        hometown='',
                        occupation='',
                    )

                    result = neural_searcher.seach_by_legal_identity(legal_info, vectorizer=vectorizer, svd=svd, tokenizer=tokenizer, model=model)
                    semantic_result = []
                    for info, score in result:
                        similar_info = ExtendedLegalIdentity(
                            text="",
                            id_number=info['id_number'],
                            full_name=info['full_name'],
                            dob=info['dob'],
                            gender=info['gender'],
                            nationality=info['nationality'],
                            place_of_origin=info['place_of_origin'],
                            place_of_residence=info['place_of_residence'],
                            hometown=info['hometown'],
                            occupation=info['occupation'],
                            similar_score=score,
                        )
                        semantic_score = compare(legal_info, similar_info, vectorizer=vectorizer, svd=svd,
                                                 tokenizer=tokenizer, model=model)
                        print('semantic_score: ' + str(semantic_score) + ', similar_score: ' + str(
                            score) + ', similar_info: ' + str(similar_info))
                        # print('similar_score: ' + str(score) + ', similar_info: ' + str(similar_info))

                        similar_info.semantic_score = semantic_score
                        semantic_result.append(similar_info)

                    df = pd.DataFrame([item.to_dict(include_vector=False) for item in semantic_result])
                    st.write('Results: ')
                    st.dataframe(df)


def text_from_form():
    col1, col2 = st.columns(2)
    with col1:
        id_number = st.text_input("ID Number", "", key="id_number")
        full_name = st.text_input("Name", "", key="full_name")
        dob = st.date_input("Date of birth", datetime.date(2025, 3, 1), format='DD/MM/YYYY',  key="dob")
        gender = st.text_input("Gender", "", key="gender")
    with col2:
        nationality = st.text_input("Nationality", "", key="nationality")
        place_of_origin = st.text_input("Place of Origin", "", key="place_of_origin")
        place_of_residence = st.text_input("Place of Residence", "", key="place_of_residence")

    click_search = st.button("Search", type="primary")

    neural_searcher = get_neural_searcher(collection_name="legal_collection_test3")
    vectorizer, svd = get_weighted_model()
    tokenizer, model = get_sentence_transformer_model()

    if click_search:
        legal_info = ExtendedLegalIdentity(
            text='',
            id_number=id_number,
            full_name=full_name,
            dob=dob,
            gender=gender,
            nationality=nationality,
            place_of_origin=place_of_origin,
            place_of_residence=place_of_residence,
            hometown='',
            occupation='',
        )

        result = neural_searcher.seach_by_legal_identity(legal_info, vectorizer=vectorizer, svd=svd, tokenizer=tokenizer, model=model)
        semantic_result = []
        for info, score in result:
            similar_info = ExtendedLegalIdentity(
                text="",
                id_number=info['id_number'],
                full_name=info['full_name'],
                dob=info['dob'],
                gender=info['gender'],
                nationality=info['nationality'],
                place_of_origin=info['place_of_origin'],
                place_of_residence=info['place_of_residence'],
                hometown=info['hometown'],
                occupation=info['occupation'],
                similar_score=score,
            )
            semantic_score = compare(legal_info, similar_info, vectorizer=vectorizer, svd=svd, tokenizer=tokenizer, model=model)
            print('semantic_score: ' + str(semantic_score) + ', similar_score: ' + str(score) + ', similar_info: ' + str(similar_info))
            # print('similar_score: ' + str(score) + ', similar_info: ' + str(similar_info))

            similar_info.semantic_score = semantic_score
            semantic_result.append(similar_info)

        df = pd.DataFrame([item.to_dict(include_vector=False) for item in semantic_result])
        st.write('Results: ')
        st.dataframe(df)


tab1, tab2 = st.tabs(["Search by Text", "Search by OCR"])

with tab1:
    text_from_form()
with tab2:
    text_ocr_from_file()