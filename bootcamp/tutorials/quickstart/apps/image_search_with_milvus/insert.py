import requests
import zipfile
import certifi
import os
from encoder import load_model
from milvus_utils import get_db
from PIL import Image


def download_file(url, dest):
    response = requests.get(url, verify=certifi.where())
    with open(dest, "wb") as f:
        f.write(response.content)


# Download and unzip data if not already done
zip_path = "reverse_image_search.zip"
if not os.path.exists(zip_path):
    url = "https://github.com/milvus-io/pymilvus-assets/releases/download/imagedata/reverse_image_search.zip"
    download_file(url, zip_path)
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(".")

extractor = load_model("resnet34")


root = "./train"
client = get_db()
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
