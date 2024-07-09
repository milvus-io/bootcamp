import streamlit as st
import torch
import timm
from sklearn.preprocessing import normalize
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
import os

# Create path to the model file (may subject to change depending on where you store the file)
documents_dir = os.path.expanduser("~/Documents")
folder_name = "streamlit_app_image"
MODEL_PATH = os.path.join(documents_dir, folder_name, "feature_extractor_model.pth")
# Ensure the directory exists
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)


class FeatureExtractor:
    def __init__(self, modelname):
        self.model = timm.create_model(
            modelname, pretrained=True, num_classes=0, global_pool="avg"
        )
        self.model.eval()
        self.input_size = self.model.default_cfg["input_size"]
        config = resolve_data_config({}, model=modelname)
        self.preprocess = create_transform(**config)

    def save(self, path):
        torch.save(self.model.state_dict(), path)

    @staticmethod
    def load(path, modelname):
        model = FeatureExtractor(modelname)
        model.model.load_state_dict(torch.load(path))
        model.model.eval()
        return model

    def __call__(self, input):
        input_image = input.convert("RGB")
        input_image = self.preprocess(input_image)
        input_tensor = input_image.unsqueeze(0)
        with torch.no_grad():
            output = self.model(input_tensor)
        feature_vector = output.squeeze().numpy()
        return normalize(feature_vector.reshape(1, -1), norm="l2").flatten()


@st.cache_resource
def load_model(modelname, path=MODEL_PATH):
    if os.path.exists(path):
        return FeatureExtractor.load(path, modelname)
    else:
        model = FeatureExtractor(modelname)
        model.save(path)
        return model
