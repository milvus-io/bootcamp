import urllib.request
import ssl
import certifi
from tqdm import tqdm
from encoder import emb_text
from milvus_utils import get_collection, get_collection_name
from ask_llm import get_azure_client
from milvus_utils import get_milvus_client

# Set SSL context
ssl_context = ssl.create_default_context(cafile=certifi.where())

# URL for the file to be retrieved
url = "https://raw.githubusercontent.com/milvus-io/milvus/master/DEVELOPMENT.md"
file_path = "./Milvus_DEVELOPMENT.md"


def get_text():
    # Retrieve the URL content
    with urllib.request.urlopen(url, context=ssl_context) as response:
        with open(file_path, "wb") as file:
            file.write(response.read())

    # Read the downloaded file
    with open(file_path, "r") as file:
        file_text = file.read()

    # Split text into lines
    text_lines = file_text.split("# ")
    return text_lines


client = get_azure_client()
milvus_client = get_milvus_client(uri="./milvus_demo.db")

text_lines = get_text()
collection_name = get_collection_name()
get_collection(milvus_client, text_lines, client)

# Prepare data for insertion into Milvus
data = []
for i, line in enumerate(tqdm(text_lines, desc="Creating embeddings")):
    data.append({"id": i, "vector": emb_text(client, line), "text": line})

# Insert data into Milvus collection
milvus_client.insert(collection_name=collection_name, data=data)
