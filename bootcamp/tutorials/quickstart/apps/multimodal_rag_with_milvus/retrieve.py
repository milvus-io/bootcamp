import os
import pickle
from typing import Dict
from magiclens.data_utils import build_circo_dataset
from magiclens.data_utils import build_fiq_dataset
from flax import serialization
import jax
import jax.numpy as jnp
from magiclens.model import MagicLens
from scenic.projects.baselines.clip import tokenizer as clip_tokenizer
import numpy as np
from PIL import Image
from cfg import Config

config = Config()
jax.config.update("jax_platform_name", config.device)


def process_img(image_path: str, size: int) -> np.ndarray:
    """Process a single image to the desired size and normalize."""
    img = Image.open(image_path).convert("RGB")
    img = img.resize((size, size), Image.BILINEAR)
    img = np.array(img) / 255.0
    img = img[np.newaxis, ...]
    return img


def load_model(model_size: str, model_path: str) -> Dict:
    """Load and initialize the model."""
    model = MagicLens(model_size)
    rng = jax.random.PRNGKey(0)
    dummy_input = {
        "ids": jnp.ones((1, 1, 77), dtype=jnp.int32),
        "image": jnp.ones((1, 224, 224, 3), dtype=jnp.float32),
    }
    params = model.init(rng, dummy_input)
    print("Model initialized")

    with open(model_path, "rb") as f:
        model_bytes = pickle.load(f)
    params = serialization.from_bytes(params, model_bytes)
    print("Model loaded")
    return model, params


model, model_params = load_model(config.model_type, config.model_path)


@jax.jit
def apply_model(params, image, ids):
    return model.apply(params, {"ids": ids, "image": image})


class Retriever:
    def __init__(self):
        global model_params
        self.tokenizer = clip_tokenizer.build_tokenizer()
        self.model_params = model_params

    def encode_query(self, img_path, text):
        img = process_img(img_path, 224)
        tokens = self.tokenizer(text)
        res = apply_model(self.model_params, img, tokens)
        return np.array(res["multimodal_embed"])
