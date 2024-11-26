import os
import sys
from glob import glob
from PIL import Image
from tqdm import tqdm

from encoder import FeatureExtractor
from milvus_utils import get_milvus_client, create_collection

from dotenv import load_dotenv

load_dotenv()

COLLECTION_NAME = os.getenv("COLLECTION_NAME")
MILVUS_ENDPOINT = os.getenv("MILVUS_ENDPOINT")
MILVUS_TOKEN = os.getenv("MILVUS_TOKEN")
MODEL_NAME = os.getenv("MODEL_NAME")
MODEL_DIM = os.getenv("MODEL_DIM")

data_dir = sys.argv[-1]
image_encoder = FeatureExtractor(MODEL_NAME)
milvus_client = get_milvus_client(uri=MILVUS_ENDPOINT, token=MILVUS_TOKEN)

# Create collection
create_collection(
    milvus_client=milvus_client, collection_name=COLLECTION_NAME, dim=int(MODEL_DIM)
)

# Load images from directory and generate embeddings
image_paths = glob(os.path.join(data_dir, "**/*.JPEG"))
data = []
for i, filepath in enumerate(tqdm(image_paths, desc="Generating embeddings ...")):
    try:
        image = Image.open(filepath)
        image_embedding = image_encoder(image)
        data.append({"vector": image_embedding, "filename": filepath})
    except Exception as e:
        print(
            f"Skipping file: {filepath} due to an error occurs during the embedding process:\n{e}"
        )
        continue

# Insert data into Milvus
mr = milvus_client.insert(
    collection_name=COLLECTION_NAME,
    data=data,
)
print("Total number of inserted entities/images:", mr["insert_count"])
