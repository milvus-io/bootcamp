import streamlit as st


# Cache for embeddings
@st.cache_resource
def get_embedding_cache():
    return {}


embedding_cache = get_embedding_cache()


# Define a function to get text embeddings
def emb_text(client, text):
    if text in embedding_cache:
        return embedding_cache[text]
    else:
        embedding = (
            client.embeddings.create(input=text, model="zilliz-text-embedding-3-small")
            .data[0]
            .embedding
        )
        embedding_cache[text] = embedding
        return embedding
