import streamlit as st
from pymilvus import MilvusClient
import os
from PIL import Image
from insert import insert_embeddings


@st.cache_resource
def get_milvus_client(uri):
    return MilvusClient(uri=uri)


@st.cache_resource
def get_db():
    if not os.path.exists("example.db"):
        client = get_milvus_client(uri="example.db")
        client.create_collection(
            collection_name="image_embeddings",
            vector_field_name="vector",
            dimension=512,
            auto_id=True,
            enable_dynamic_field=True,
            metric_type="COSINE",
        )
        insert_embeddings(client)

    else:
        client = get_milvus_client(uri="example.db")
    return client
