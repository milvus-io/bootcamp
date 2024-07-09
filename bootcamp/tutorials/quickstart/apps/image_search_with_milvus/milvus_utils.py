import streamlit as st
from pymilvus import MilvusClient
import os
from PIL import Image
from encoder import load_model

extractor = load_model("resnet34")

def insert_embeddings(client):
    print('inserting')
    global extractor
    root = "./train"
    for dirpath, foldername, filenames in os.walk(root):
        for filename in filenames:
            if filename.endswith(".JPEG"):
                filepath = os.path.join(dirpath, filename)
                img = Image.open(filepath)
                image_embedding = extractor(img)
                client.insert(
                    "image_embeddings",
                    {"vector": image_embedding, "filename": filepath},
                )

@st.cache_resource
def get_milvus_client(uri):
    return MilvusClient(uri=uri)

@st.cache_resource
def db_exists_check():
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