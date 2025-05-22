# Build RAG on Arm Architecture

[Arm](https://www.arm.com/) CPUs are extensively utilized across a wide range of applications, including traditional machine learning (ML) and artificial intelligence (AI) use cases. 

In this tutorial, you learn how to build a Retrieval-Augmented Generation (RAG) application on Arm-based infrastructures. For vector storage, we utilize [Zilliz Cloud](https://zilliz.com/cloud), the fully-managed Milvus vector database. Zilliz Cloud is available on major cloud such as AWS, GCP and Azure. In this demo we use Zilliz Cloud deployed on AWS with Arm machines. For LLM, we use the `Llama-3.1-8B` model on the AWS Arm-based server CPU using `llama.cpp`. 


## Prerequisite
To run this example, we recommend you to use [AWS Graviton](https://aws.amazon.com/ec2/graviton/), which provides a cost-effective way to run ML workloads on Arm-based servers. This notebook has been tested on an AWS Graviton3 `c7g.2xlarge` instance with Ubuntu 22.04 LTS system.

You need at least four cores and 8GB of RAM to run this example. Configure disk storage up to at least 32 GB. We recommend that you use an instance of the same or better specification.

After you launch the instance, connect to it and run the following commands to prepare the environment.

Install python on the server:

```bash
sudo apt update
sudo apt install python-is-python3 python3-pip python3-venv -y
```

Create and activate a virtual environment:

```bash
python -m venv venv
source venv/bin/activate
```

Install the required python dependencies:

```shell
pip install --upgrade pymilvus openai requests langchain-huggingface huggingface_hub tqdm
```


## Offline Data Loading


### Create the Collection
We use [Zilliz Cloud](https://zilliz.com/cloud) deployed on AWS with Arm-based machines to store and retrieve the vector data. To quick start, simply [register an account](https://docs.zilliz.com/docs/register-with-zilliz-cloud) on Zilliz Cloud for free.

> In addition to Zilliz Cloud, self-hosted Milvus is also a (more complicated to set up) option. We can also deploy [Milvus Standalone](https://milvus.io/docs/install_standalone-docker-compose.md) and [Kubernetes](https://milvus.io/docs/install_cluster-milvusoperator.md) on ARM-based machines. For more information about Milvus installation, please refer to the [installation documentation](https://milvus.io/docs/install-overview.md).

We set the `uri` and `token` as the [Public Endpoint and Api key](https://docs.zilliz.com/docs/on-zilliz-cloud-console#free-cluster-details) in Zilliz Cloud.
```python
from pymilvus import MilvusClient

milvus_client = MilvusClient(
    uri="<your_zilliz_public_endpoint>", token="<your_zilliz_api_key>"
)

collection_name = "my_rag_collection"

```
Check if the collection already exists and drop it if it does.
```python
if milvus_client.has_collection(collection_name):
    milvus_client.drop_collection(collection_name)
```
Create a new collection with specified parameters. 

If we don't specify any field information, Milvus will automatically create a default `id` field for primary key, and a `vector` field to store the vector data. A reserved JSON field is used to store non-schema-defined fields and their values.
```python
milvus_client.create_collection(
    collection_name=collection_name,
    dimension=384,
    metric_type="IP",  # Inner product distance
    consistency_level="Strong",  # Strong consistency level
)
```
We use inner product distance as the default metric type. For more information about distance types, you can refer to [Similarity Metrics page](https://milvus.io/docs/metric.md?tab=floating)

### Prepare the data

We use the FAQ pages from the [Milvus Documentation 2.4.x](https://github.com/milvus-io/milvus-docs/releases/download/v2.4.6-preview/milvus_docs_2.4.x_en.zip) as the private knowledge in our RAG, which is a good data source for a simple RAG pipeline.

Download the zip file and extract documents to the folder `milvus_docs`.

```shell
wget https://github.com/milvus-io/milvus-docs/releases/download/v2.4.6-preview/milvus_docs_2.4.x_en.zip
unzip -q milvus_docs_2.4.x_en.zip -d milvus_docs
```

We load all markdown files from the folder `milvus_docs/en/faq`. For each document, we just simply use "# " to separate the content in the file, which can roughly separate the content of each main part of the markdown file.


```python
from glob import glob

text_lines = []

for file_path in glob("milvus_docs/en/faq/*.md", recursive=True):
    with open(file_path, "r") as file:
        file_text = file.read()

    text_lines += file_text.split("# ")
```

### Insert data
We prepare a simple but efficient embedding model [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) that can convert text into embedding vectors.
```python
from langchain_huggingface import HuggingFaceEmbeddings

embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
```

Iterate through the text lines, create embeddings, and then insert the data into Milvus.

Here is a new field `text`, which is a non-defined field in the collection schema. It will be automatically added to the reserved JSON dynamic field, which can be treated as a normal field at a high level.
```python
from tqdm import tqdm

data = []

text_embeddings = embedding_model.embed_documents(text_lines)

for i, (line, embedding) in enumerate(
    tqdm(zip(text_lines, text_embeddings), desc="Creating embeddings")
):
    data.append({"id": i, "vector": embedding, "text": line})

milvus_client.insert(collection_name=collection_name, data=data)
```
```text
Creating embeddings: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 72/72 [00:18<00:00,  3.91it/s]
```


## Launch LLM Service on Arm

In this section, we will build and launch the `llama.cpp` service on the Arm-based CPU.

### Llama 3.1 model & llama.cpp

The [Llama-3.1-8B model](https://huggingface.co/cognitivecomputations/dolphin-2.9.4-llama3.1-8b-gguf) from Meta belongs to the Llama 3.1 model family and is free to use for research and commercial purposes. Before you use the model, visit the Llama [website](https://llama.meta.com/llama-downloads/) and fill in the form to request access.

[llama.cpp](https://github.com/ggerganov/llama.cpp) is an open source C/C++ project that enables efficient LLM inference on a variety of hardware - both locally, and in the cloud. You can conveniently host a Llama 3.1 model using `llama.cpp`.


### Download and build llama.cpp

Run the following commands to install make, cmake, gcc, g++, and other essential tools required for building llama.cpp from source:

```bash
sudo apt install make cmake -y
sudo apt install gcc g++ -y
sudo apt install build-essential -y
```

You are now ready to start building `llama.cpp`. 

Clone the source repository for llama.cpp:

```bash
git clone https://github.com/ggerganov/llama.cpp
```

By default, `llama.cpp` builds for CPU only on Linux and Windows. You don't need to provide any extra switches to build it for the Arm CPU that you run it on.

Run `make` to build it:

```bash
cd llama.cpp
make GGML_NO_LLAMAFILE=1 -j$(nproc)
```

Check that `llama.cpp` has built correctly by running the help command:

```bash
./llama-cli -h
```

If `llama.cpp` has been built correctly, you will see the help option displayed. The output snippet looks like this:

```output
example usage:

  text generation:     ./llama-cli -m your_model.gguf -p "I believe the meaning of life is" -n 128

  chat (conversation): ./llama-cli -m your_model.gguf -p "You are a helpful assistant" -cnv
```


You can now download the model using the huggingface cli:

```bash
huggingface-cli download cognitivecomputations/dolphin-2.9.4-llama3.1-8b-gguf dolphin-2.9.4-llama3.1-8b-Q4_0.gguf --local-dir . --local-dir-use-symlinks False
```
The GGUF model format, introduced by the llama.cpp team, uses compression and quantization to reduce weight precision to 4-bit integers, significantly decreasing computational and memory demands and making Arm CPUs effective for LLM inference.


### Re-quantize the model weights

To re-quantize, run

```bash
./llama-quantize --allow-requantize dolphin-2.9.4-llama3.1-8b-Q4_0.gguf dolphin-2.9.4-llama3.1-8b-Q4_0_8_8.gguf Q4_0_8_8
```

This will output a new file, `dolphin-2.9.4-llama3.1-8b-Q4_0_8_8.gguf`, which contains reconfigured weights that allow `llama-cli` to use SVE 256 and MATMUL_INT8 support.

> This requantization is optimal specifically for Graviton3. For Graviton2, the optimal requantization should be performed in the `Q4_0_4_4` format, and for Graviton4, the `Q4_0_4_8` format is the most suitable for requantization.

### Start the LLM Service
You can utilize the llama.cpp server program and send requests via an OpenAI-compatible API. This allows you to develop applications that interact with the LLM multiple times without having to repeatedly start and stop it. Additionally, you can access the server from another machine where the LLM is hosted over the network.

Start the server from the command line, and it listens on port 8080:

```shell
./llama-server -m dolphin-2.9.4-llama3.1-8b-Q4_0_8_8.gguf -n 2048 -t 64 -c 65536  --port 8080
```
```text
'main: server is listening on 127.0.0.1:8080 - starting the main loop
```

You can also adjust the parameters of the launched LLM to adapt it to your server hardware to obtain ideal performance. For more parameter information, see the `llama-server --help` command.

If you struggle to perform this step, you can refer to the [official documents](https://learn.arm.com/learning-paths/servers-and-cloud-computing/llama-cpu/llama-chatbot/) for more information.

You have started the LLM service on your Arm-based CPU. Next, we directly interact with the service using the OpenAI SDK.


## Online RAG

### LLM Client and Embedding Model

We initialize the LLM client and prepare the embedding model.

For the LLM, we use the OpenAI SDK to request the Llama service launched before. We don't need to use any API key because it is actually our local llama.cpp service.

```python
from openai import OpenAI

llm_client = OpenAI(base_url="http://localhost:8080/v1", api_key="no-key")
```
Generate a test embedding and print its dimension and first few elements.

```python
test_embedding = embedding_model.embed_query("This is a test")
embedding_dim = len(test_embedding)
print(embedding_dim)
print(test_embedding[:10])
```

```text
384
[0.03061249852180481, 0.013831384479999542, -0.02084377221763134, 0.016327863559126854, -0.010231520049273968, -0.0479842908680439, -0.017313342541456223, 0.03728749603033066, 0.04588735103607178, 0.034405000507831573]
```

### Retrieve data for a query

Let's specify a frequent question about Milvus.
```python
question = "How is data stored in milvus?"
```
Search for the question in the collection and retrieve the semantic top-3 matches.

```python
search_res = milvus_client.search(
    collection_name=collection_name,
    data=[
        embedding_model.embed_query(question)
    ],  # Use the `emb_text` function to convert the question to an embedding vector
    limit=3,  # Return top 3 results
    search_params={"metric_type": "IP", "params": {}},  # Inner product distance
    output_fields=["text"],  # Return the text field
)
```
Let's take a look at the search results of the query
```python
import json

retrieved_lines_with_distances = [
    (res["entity"]["text"], res["distance"]) for res in search_res[0]
]
print(json.dumps(retrieved_lines_with_distances, indent=4))
```

```shell
[
    [
        " Where does Milvus store data?\n\nMilvus deals with two types of data, inserted data and metadata. \n\nInserted data, including vector data, scalar data, and collection-specific schema, are stored in persistent storage as incremental log. Milvus supports multiple object storage backends, including [MinIO](https://min.io/), [AWS S3](https://aws.amazon.com/s3/?nc1=h_ls), [Google Cloud Storage](https://cloud.google.com/storage?hl=en#object-storage-for-companies-of-all-sizes) (GCS), [Azure Blob Storage](https://azure.microsoft.com/en-us/products/storage/blobs), [Alibaba Cloud OSS](https://www.alibabacloud.com/product/object-storage-service), and [Tencent Cloud Object Storage](https://www.tencentcloud.com/products/cos) (COS).\n\nMetadata are generated within Milvus. Each Milvus module has its own metadata that are stored in etcd.\n\n###",
        0.6488019824028015
    ],
    [
        "How does Milvus flush data?\n\nMilvus returns success when inserted data are loaded to the message queue. However, the data are not yet flushed to the disk. Then Milvus' data node writes the data in the message queue to persistent storage as incremental logs. If `flush()` is called, the data node is forced to write all data in the message queue to persistent storage immediately.\n\n###",
        0.5974207520484924
    ],
    [
        "What is the maximum dataset size Milvus can handle?\n\n  \nTheoretically, the maximum dataset size Milvus can handle is determined by the hardware it is run on, specifically system memory and storage:\n\n- Milvus loads all specified collections and partitions into memory before running queries. Therefore, memory size determines the maximum amount of data Milvus can query.\n- When new entities and and collection-related schema (currently only MinIO is supported for data persistence) are added to Milvus, system storage determines the maximum allowable size of inserted data.\n\n###",
        0.5833579301834106
    ]
]
```
### Use LLM to get a RAG response

Convert the retrieved documents into a string format.
```python
context = "\n".join(
    [line_with_distance[0] for line_with_distance in retrieved_lines_with_distances]
)
```
Define system and user prompts for the Language Model. This prompt is assembled with the retrieved documents from Milvus.

```python
SYSTEM_PROMPT = """
Human: You are an AI assistant. You are able to find answers to the questions from the contextual passage snippets provided.
"""
USER_PROMPT = f"""
Use the following pieces of information enclosed in <context> tags to provide an answer to the question enclosed in <question> tags.
<context>
{context}
</context>
<question>
{question}
</question>
"""
```
Use LLM to generate a response based on the prompts. We set the `model` parameter to `not-used` since it is a redundant parameter for the llama.cpp service.

```python
response = llm_client.chat.completions.create(
    model="not-used",
    messages=[
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_PROMPT},
    ],
)
print(response.choices[0].message.content)

```
```text
Milvus stores data in two types: inserted data and metadata. Inserted data, including vector data, scalar data, and collection-specific schema, are stored in persistent storage as incremental log. Milvus supports multiple object storage backends such as MinIO, AWS S3, Google Cloud Storage (GCS), Azure Blob Storage, Alibaba Cloud OSS, and Tencent Cloud Object Storage (COS). Metadata are generated within Milvus and each Milvus module has its own metadata that are stored in etcd.
```
Congratulations! You have built a RAG application on top of the Arm-based infrastructures.
