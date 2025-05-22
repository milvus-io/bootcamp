import streamlit as st
from retrieve import Retriever
from pymilvus import MilvusClient
from reranker import GenerativeReranker
from cfg import Config
from PIL import Image
import json
import os
import numpy as np
import cv2

st.set_page_config(layout="wide")


@st.cache_resource
def get_retriever():
    retriever = Retriever()
    return retriever


@st.cache_resource
def get_milvus_client():
    config = Config()
    client = MilvusClient(config.milvus_uri)
    return client


class Interface:
    def __init__(self):
        self.init_sidebar()
        print("Interface Inited")

    def init_sidebar(self):
        st.sidebar.image("./pics/milvus_logo.png", width=200)
        st.sidebar.title("Multi-Modal Retrieval")

        # Upload file and cache it in session state
        uploaded_file = st.sidebar.file_uploader("Upload File")
        if uploaded_file is not None:
            st.session_state["uploaded_file"] = uploaded_file
            st.sidebar.image(
                uploaded_file, caption="Query Image", use_column_width=True
            )

        # Text input and cache it in session state
        text_input = st.sidebar.text_input("Instruction:")
        st.session_state["text_input"] = text_input

        cols = st.sidebar.columns([2, 1])

        if cols[0].button("Search"):
            self.search()

        if cols[1].button("Ask GPT"):
            self.show_best()

    def search(self):
        config = Config()
        text = st.session_state.get("text_input", "")
        uploaded_file = st.session_state.get("uploaded_file", "")
        with open("temp.jpg", "wb") as f:
            f.write(uploaded_file.getbuffer())

        retriever = get_retriever()
        client = get_milvus_client()

        emb = retriever.encode_query("temp.jpg", text)

        st.session_state["cached_results"] = []
        st.session_state["searched_results"] = []

        search_results = client.search(
            collection_name="cir_demo_large",
            data=[emb.flatten()],
            output_fields=["spec"],
            limit=100,  # Max. number of search results to return
            search_params={"metric_type": "COSINE", "params": {}},  # Search parameters
        )

        cols = st.columns(5)
        image_folder = config.download_path + "/{}"
        st.session_state["searched_results"] = search_results[0]
        for i, info in enumerate(search_results[0]):
            imgname = json.loads(info["entity"]["spec"])["images"]["large"][0]
            imgname = image_folder.format(os.path.basename(imgname))
            st.session_state["cached_results"].append(imgname)
            img = Image.open(imgname)
            cols[i % 5].image(img, use_column_width=True)

    def get_cached_results(self):
        config = Config()
        cached_results = st.session_state.get("cached_results", [])
        img_data = []
        for imgname in cached_results:
            img = cv2.imread(imgname)
            if img is not None:
                img_data.append(img)
        return img_data

    @st.experimental_dialog("Best Item")
    def show_best_item_dialog(self, top_images, top_infos, best_index, explanation):
        st.image(
            top_images[best_index][:, :, ::-1],
            caption="Best Item",
            use_column_width=True,
        )
        st.write("Explanation:", explanation)

    def show_best(self):
        config = Config()
        inst_text = st.session_state.get("text_input", "")
        uploaded_file = st.session_state.get("uploaded_file", "")

        with open("temp.jpg", "wb") as f:
            f.write(uploaded_file.getbuffer())
        reranker = GenerativeReranker(api_type=config.api_type)

        top_images = self.get_cached_results()[:25]

        top_infos = {"product": [], "instruction": ""}
        search_results = st.session_state.get("searched_results", [])[:25]

        for i, info in enumerate(search_results):
            img_info = info["entity"]
            top_infos["product"].append(img_info)

        top_infos["instruction"] = inst_text

        query_image = Image.open(uploaded_file).convert("RGB")
        query_array = np.array(query_image)[:, :, ::-1]

        best_index = reranker.get_best_item(
            query_array, inst_text, top_images, top_infos
        )

        explanation = reranker.explain(uploaded_file, inst_text, top_images, top_infos)
        print("best_index:", best_index)

        self.show_best_item_dialog(top_images, top_infos, best_index, explanation)

        cols = st.columns(5)
        image_folder = config.download_path + "/{}"

        full_search_results = st.session_state.get("searched_results", [])

        for i, info in enumerate(full_search_results):
            imgname = json.loads(info["entity"]["spec"])["images"]["large"][0]
            imgname = image_folder.format(os.path.basename(imgname))
            img = Image.open(imgname)
            cols[i % 5].image(img, use_column_width=True)


if __name__ == "__main__":
    interface = Interface()
