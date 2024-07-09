import streamlit as st
from encoder import emb_text
from pymilvus import MilvusClient


# Get a test embedding to determine the embedding dimension
def get_embedding_dim(client):
    test_embedding = emb_text(client, "This is a test")
    embedding_dim = len(test_embedding)
    return embedding_dim


def get_collection_name():
    return "my_rag_collection"


def get_collection(milvus_client, text_lines, client):
    collection_name = get_collection_name()

    # Drop collection if it exists
    if milvus_client.has_collection(collection_name):
        milvus_client.drop_collection(collection_name)

    # Check if the collection exists
    if not milvus_client.has_collection(collection_name):
        milvus_client.create_collection(
            collection_name=collection_name,
            dimension=get_embedding_dim(client),
            metric_type="IP",  # Inner product distance
            consistency_level="Strong",  # Strong consistency level
        )


# Initialize Milvus client
@st.cache_resource
def get_milvus_client(uri):
    return MilvusClient(uri=uri)


def get_search_results(milvus_client, client, collection_name, question):
    search_res = milvus_client.search(
        collection_name=collection_name,
        data=[emb_text(client, question)],  # Convert question to embedding vector
        limit=3,  # Return top 3 results
        search_params={"metric_type": "IP", "params": {}},  # Inner product distance
        output_fields=["text"],  # Return the text field
    )
    return search_res
