import sys
import os
import ssl
import certifi
from glob import glob
from tqdm import tqdm

from encoder import emb_text, OpenAI
from milvus_utils import get_milvus_client, create_collection

from dotenv import load_dotenv


load_dotenv()
COLLECTION_NAME = os.getenv("COLLECTION_NAME")
MILVUS_ENDPOINT = os.getenv("MILVUS_ENDPOINT")
MILVUS_TOKEN = os.getenv("MILVUS_TOKEN")


def get_text(data_dir):
    """Load documents and split each into chunks.

    Return:
        A dictionary of text chunks with the filepath as key value.
    """
    text_dict = {}
    for file_path in glob(os.path.join(data_dir, "**/*.md"), recursive=True):
        if file_path.endswith(".md"):
            with open(file_path, "r") as file:
                file_text = file.read().strip()
            text_dict[file_path] = file_text.split("# ")
    return text_dict


# Get clients
milvus_client = get_milvus_client(uri=MILVUS_ENDPOINT, token=MILVUS_TOKEN)
openai_client = OpenAI()

# Set SSL context
ssl_context = ssl.create_default_context(cafile=certifi.where())

# Get text data from data directory
data_dir = sys.argv[-1]
text_dict = get_text(data_dir)

# Create collection
dim = len(emb_text(openai_client, "test"))
create_collection(milvus_client=milvus_client, collection_name=COLLECTION_NAME, dim=dim)

# Insert data
data = []
count = 0
for i, filepath in enumerate(tqdm(text_dict, desc="Creating embeddings")):
    chunks = text_dict[filepath]
    for line in chunks:
        try:
            vector = emb_text(openai_client, line)
            data.append({"vector": vector, "text": line})
            count += 1
        except Exception as e:
            print(
                f"Skipping file: {filepath} due to an error occurs during the embedding process:\n{e}"
            )
            continue
print("Total number of loaded documents:", count)

# Insert data into Milvus collection
mr = milvus_client.insert(collection_name=COLLECTION_NAME, data=data)
print("Total number of entities/chunks inserted:", mr["insert_count"])
