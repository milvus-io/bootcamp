import streamlit as st
from streamlit import cache_resource
from pymilvus.model.hybrid import BGEM3EmbeddingFunction
from pymilvus import (
    Collection,
    AnnSearchRequest,
    WeightedRanker,
    connections,
)

# Logo
st.image("./pics/Milvus_Logo_Official.png", width=200)


@cache_resource
def get_model():
    ef = BGEM3EmbeddingFunction(use_fp16=False, device="cpu")
    return ef


@cache_resource
def get_collection():
    col_name = "hybrid_demo"
    connections.connect("default", uri="milvus.db")
    col = Collection(col_name)
    return col


def search_from_source(source, query):
    return [f"{source} Result {i+1} for {query}" for i in range(5)]


st.title("Milvus Hybrid Search Demo")

query = st.text_input("Enter your search query:")
search_button = st.button("Search")


@cache_resource
def get_tokenizer():
    ef = get_model()
    tokenizer = ef.model.tokenizer
    return tokenizer


def doc_text_colorization(query, docs):
    tokenizer = get_tokenizer()
    query_tokens_ids = tokenizer.encode(query, return_offsets_mapping=True)
    query_tokens = tokenizer.convert_ids_to_tokens(query_tokens_ids)
    colored_texts = []

    for doc in docs:
        ldx = 0
        landmarks = []
        encoding = tokenizer.encode_plus(doc, return_offsets_mapping=True)
        tokens = tokenizer.convert_ids_to_tokens(encoding["input_ids"])[1:-1]
        offsets = encoding["offset_mapping"][1:-1]
        for token, (start, end) in zip(tokens, offsets):
            if token in query_tokens:
                if len(landmarks) != 0 and start == landmarks[-1]:
                    landmarks[-1] = end
                else:
                    landmarks.append(start)
                    landmarks.append(end)
        close = False
        color_text = ""
        for i, c in enumerate(doc):
            if ldx == len(landmarks):
                pass
            elif i == landmarks[ldx]:
                if close is True:
                    color_text += "]"
                else:
                    color_text += ":red["
                close = not close
                ldx = ldx + 1
            color_text += c
        if close is True:
            color_text += "]"
        colored_texts.append(color_text)
    return colored_texts


def hybrid_search(query_embeddings, sparse_weight=1.0, dense_weight=1.0):
    col = get_collection()
    sparse_search_params = {"metric_type": "IP"}
    sparse_req = AnnSearchRequest(
        query_embeddings["sparse"], "sparse_vector", sparse_search_params, limit=10
    )
    dense_search_params = {"metric_type": "IP"}
    dense_req = AnnSearchRequest(
        query_embeddings["dense"], "dense_vector", dense_search_params, limit=10
    )
    rerank = WeightedRanker(sparse_weight, dense_weight)
    res = col.hybrid_search(
        [sparse_req, dense_req], rerank=rerank, limit=10, output_fields=["text"]
    )
    if len(res):
        return [hit.fields["text"] for hit in res[0]]
    else:
        return []


# Display search results when the button is clicked
if search_button and query:
    ef = get_model()
    query_embeddings = ef([query])
    col1, col2, col3 = st.columns(3)
    with col1:
        st.header("Dense")
        results = hybrid_search(query_embeddings, sparse_weight=0.0, dense_weight=1.0)
        for result in results:
            st.markdown(result)

    with col2:
        st.header("Sparse")
        results = hybrid_search(query_embeddings, sparse_weight=1.0, dense_weight=0.0)
        colored_results = doc_text_colorization(query, results)
        for result in colored_results:
            st.markdown(result)

    with col3:
        st.header("Hybrid")
        results = hybrid_search(query_embeddings, sparse_weight=0.7, dense_weight=1.0)
        colored_results = doc_text_colorization(query, results)
        for result in colored_results:
            st.markdown(result)
