import streamlit as st
from openai import OpenAI


# Cache for embeddings
@st.cache_resource
def get_embedding_cache():
    return {}


embedding_cache = get_embedding_cache()


def emb_text(client: OpenAI, text: str, model: str = "text-embedding-3-small"):
    if text in embedding_cache:
        return embedding_cache[text]
    else:
        embedding = client.embeddings.create(input=text, model=model).data[0].embedding
        embedding_cache[text] = embedding
        return embedding
