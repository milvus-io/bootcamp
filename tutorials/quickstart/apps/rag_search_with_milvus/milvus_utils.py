import streamlit as st
from pymilvus import MilvusClient


@st.cache_resource
def get_milvus_client(uri: str, token: str = None) -> MilvusClient:
    return MilvusClient(uri=uri, token=token)


def create_collection(
    milvus_client: MilvusClient, collection_name: str, dim: int, drop_old: bool = True
):
    if milvus_client.has_collection(collection_name) and drop_old:
        milvus_client.drop_collection(collection_name)
    if milvus_client.has_collection(collection_name):
        raise RuntimeError(
            f"Collection {collection_name} already exists. Set drop_old=True to create a new one instead."
        )
    return milvus_client.create_collection(
        collection_name=collection_name,
        dimension=dim,
        metric_type="IP",
        consistency_level="Strong",
        auto_id=True,
    )


def get_search_results(milvus_client, collection_name, query_vector, output_fields):
    search_res = milvus_client.search(
        collection_name=collection_name,
        data=[query_vector],
        limit=3,
        search_params={"metric_type": "IP", "params": {}},  # Inner product distance
        output_fields=output_fields,
    )
    return search_res
