# Build RAG with Milvus + PII Masker

PII (Personally Identifiable Information) is a type of sensitive data that can be used to identify individuals. 

[PII Masker](https://github.com/HydroXai/pii-masker-v1/tree/main), developed by [HydroX AI](https://www.hydrox.ai/), is an advanced open-source tool designed to protect your sensitive data by leveraging cutting-edge AI models. Whether you're handling customer data, performing data analysis, or ensuring compliance with privacy regulations, PII Masker provides a robust, scalable solution to keep your information secure.

In this tutorial, we will show how to use PII Masker with Milvus to protect private data in RAG(Retrieval-Augmented Generation) applications. By combining the strengths of PII Masker's data masking capabilities with Milvus's efficient data retrieval, you can create secure, privacy-compliant pipelines for handling sensitive information with confidence. This approach ensures your applications are equipped to meet privacy standards and protect user data effectively.

## Preparation

### Get started with PII Masker

Follow the [installation guide](https://github.com/HydroXai/pii-masker-v1/tree/main?tab=readme-ov-file#-installation) of PII Masker to install the required dependencies and download the model. Here is a simple guide:

```shell
git clone https://github.com/HydroXai/pii-masker-v1.git
cd pii-masker-v1/pii-masker
```

Download model from 
`https://huggingface.co/hydroxai/pii_model_weight`, and replace it with files in: `pii-masker/output_model/deberta3base_1024/`



### Dependencies and Environment


```shell
pip install --upgrade pymilvus openai requests tqdm dataset
```

We will use OpenAI as the LLM in this example. You should prepare the [api key](https://platform.openai.com/docs/quickstart) `OPENAI_API_KEY` as an environment variable.


```shell
export OPENAI_API_KEY=sk-***********
```

Then you can create a python or jupyter notebook to run the following code.

### Prepare the data

Let's generate some fake lines which contain PII information for testing or demonstration purposes. 



```python
text_lines = [
    "Alice Johnson, a resident of Dublin, Ireland, attended a flower festival at Hyde Park on May 15, 2023. She entered the park at noon using her digital passport, number 23456789. Alice spent the afternoon admiring various flowers and plants, attending a gardening workshop, and having a light snack at one of the food stalls. While there, she met another visitor, Mr. Thompson, who was visiting from London. They exchanged tips on gardening and shared contact information: Mr. Thompson's address was 492, Pine Lane, and his cell phone number was +018.221.431-4517. Alice gave her contact details: home address, Ranch 16",
    "Hiroshi Tanaka, a businessman from Tokyo, Japan, went to attend a tech expo at the Berlin Convention Center on November 10, 2023. He registered for the event at 9 AM using his digital passport, number Q-24567680. Hiroshi networked with industry professionals, participated in panel discussions, and had lunch with some potential partners. One of the partners he met was from Munich, and they decided to keep in touch: the partner's office address was given as house No. 12, Road 7, Block E. Hiroshi offered his business card with the address, 654 Sakura Road, Tokyo.",
    "In an online forum discussion about culinary exchanges around the world, several participants shared their experiences. One user, Male, with the email 2022johndoe@example.com, shared his insights. He mentioned his ID code 1A2B3C4D5E and reference number L87654321 while residing in Italy but originally from Australia. He provided his +0-777-123-4567 and described his address at 456, Flavorful Lane, Pasta, IT, 00100.",
    "Another user joined the conversation on the topic of international volunteering opportunities. Identified as Female, she used the email 2023janedoe@example.com to share her story. She noted her 9876543210123 and M1234567890123 while residing in Germany but originally from Brazil. She provided her +0-333-987-6543 and described her address at 789, Sunny Side Street, Berlin, DE, 10178.",
]
```

### Mask the data with PIIMasker

Let's initialize the PIIMasker object and load the model.


```python
from model import PIIMasker

masker = PIIMasker()
```

We then masks PII from a list of text lines and prints the masked results.


```python
masked_results = []
for full_text in text_lines:
    masked_text, _ = masker.mask_pii(full_text)
    masked_results.append(masked_text)

for res in masked_results:
    print(res + "\n")
```
    Alice [B-NAME] , a resident of Dublin Ireland attended flower festival at Hyde Park on May 15 2023 [B-PHONE_NUM] She entered the park noon using her digital passport number 23 [B-ID_NUM] [B-NAME] afternoon admiring various flowers and plants attending gardening workshop having light snack one food stalls While there she met another visitor Mr Thompson who was visiting from London They exchanged tips shared contact information : ' s address 492 [I-STREET_ADDRESS] his cell phone + [B-PHONE_NUM] [B-NAME] details home Ranch [B-STREET_ADDRESS]
    
    Hiroshi [B-NAME] [I-STREET_ADDRESS] a businessman from Tokyo Japan went to attend tech expo at the Berlin Convention Center on November 10 2023 . He registered for event 9 AM using his digital passport number Q [B-ID_NUM] [B-NAME] with industry professionals participated in panel discussions and had lunch some potential partners One of he met was Munich they decided keep touch : partner ' s office address given as house No [I-STREET_ADDRESS] [B-NAME] business card 654 [B-STREET_ADDRESS]
    
    In an online forum discussion about culinary exchanges around the world [I-STREET_ADDRESS] several participants shared their experiences [I-STREET_ADDRESS] One user Male with email 2022 [B-EMAIL] his insights He mentioned ID code 1 [B-ID_NUM] [I-PHONE_NUM] reference number L [B-ID_NUM] residing in Italy but originally from Australia provided + [B-PHONE_NUM] [I-PHONE_NUM] described address at 456 [I-STREET_ADDRESS]
    
    Another user joined the conversation on topic of international volunteering opportunities . Identified as Female , she used email 2023 [B-EMAIL] share her story She noted 98 [B-ID_NUM] [I-PHONE_NUM] M [B-ID_NUM] residing in Germany but originally from Brazil provided + [B-PHONE_NUM] [I-PHONE_NUM] described address at 789 [I-STREET_ADDRESS] DE 10 178
    


### Prepare the Embedding Model

We initialize the OpenAI client to prepare the embedding model.


```python
from openai import OpenAI

openai_client = OpenAI()
```

Define a function to generate text embeddings using OpenAI client. We use the `text-embedding-3-small` model as an example.


```python
def emb_text(text):
    return (
        openai_client.embeddings.create(input=text, model="text-embedding-3-small")
        .data[0]
        .embedding
    )
```

Generate a test embedding and print its dimension and first few elements.


```python
test_embedding = emb_text("This is a test")
embedding_dim = len(test_embedding)
print(embedding_dim)
print(test_embedding[:10])
```

    1536
    [0.009889289736747742, -0.005578675772994757, 0.00683477520942688, -0.03805781528353691, -0.01824733428657055, -0.04121600463986397, -0.007636285852640867, 0.03225184231996536, 0.018949154764413834, 9.352207416668534e-05]


## Load data into Milvus 

### Create the Collection


```python
from pymilvus import MilvusClient

milvus_client = MilvusClient(uri="./milvus_demo.db")
```

> As for the argument of `MilvusClient`:
> - Setting the `uri` as a local file, e.g.`./milvus.db`, is the most convenient method, as it automatically utilizes [Milvus Lite](https://milvus.io/docs/milvus_lite.md) to store all data in this file.
> - If you have large scale of data, say more than a million vectors, you can set up a more performant Milvus server on [Docker or Kubernetes](https://milvus.io/docs/quickstart.md). In this setup, please use the server address and port as your uri, e.g.`http://localhost:19530`. If you enable the authentication feature on Milvus, use "<your_username>:<your_password>" as the token, otherwise don't set the token.
> - If you want to use [Zilliz Cloud](https://zilliz.com/cloud), the fully managed cloud service for Milvus, adjust the `uri` and `token`, which correspond to the [Public Endpoint and Api key](https://docs.zilliz.com/docs/on-zilliz-cloud-console#free-cluster-details) in Zilliz Cloud.

Check if the collection already exists and drop it if it does.


```python
collection_name = "my_rag_collection"

if milvus_client.has_collection(collection_name):
    milvus_client.drop_collection(collection_name)
```

Create a new collection with specified parameters. 

If we don't specify any field information, Milvus will automatically create a default `id` field for primary key, and a `vector` field to store the vector data. A reserved JSON field is used to store non-schema-defined fields and their values.


```python
milvus_client.create_collection(
    collection_name=collection_name,
    dimension=embedding_dim,
    metric_type="IP",  # Inner product distance
    consistency_level="Strong",  # Strong consistency level
)
```

### Insert data
Iterate through the masked text lines, create embeddings, and then insert the data into Milvus.

Here is a new field `text`, which is a non-defined field in the collection schema. It will be automatically added to the reserved JSON dynamic field, which can be treated as a normal field at a high level.


```python
from tqdm import tqdm

data = []

for i, line in enumerate(tqdm(masked_results, desc="Creating embeddings")):
    data.append({"id": i, "vector": emb_text(line), "text": line})

milvus_client.insert(collection_name=collection_name, data=data)
```
    Creating embeddings: 100%|██████████| 4/4 [00:01<00:00,  2.60it/s]





    {'insert_count': 4, 'ids': [0, 1, 2, 3], 'cost': 0}



## Build RAG

### Retrieve data for a query

Let's specify a question about the documents.


```python
question = "What was the office address of Hiroshi's partner from Munich?"
```

Search for the question in the collection and retrieve the semantic top-1 match.


```python
search_res = milvus_client.search(
    collection_name=collection_name,
    data=[
        emb_text(question)
    ],  # Use the `emb_text` function to convert the question to an embedding vector
    limit=1,  # Return top 1 results
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

    [
        [
            "Hiroshi [B-NAME] [I-STREET_ADDRESS] a businessman from Tokyo Japan went to attend tech expo at the Berlin Convention Center on November 10 2023 . He registered for event 9 AM using his digital passport number Q [B-ID_NUM] [B-NAME] with industry professionals participated in panel discussions and had lunch some potential partners One of he met was Munich they decided keep touch : partner ' s office address given as house No [I-STREET_ADDRESS] [B-NAME] business card 654 [B-STREET_ADDRESS]",
            0.6544462442398071
        ]
    ]


### Use LLM to get a RAG response

Convert the retrieved documents into a string format.


```python
context = "\n".join(
    [line_with_distance[0] for line_with_distance in retrieved_lines_with_distances]
)
```

Define system and user prompts for the Lanage Model.

Note: We tell LLM if there are no useful information in the snippets, just say "I don't know".


```python
SYSTEM_PROMPT = """
Human: You are an AI assistant. You are able to find answers to the questions from the contextual passage snippets provided. If there are no useful information in the snippets, just say "I don't know".
AI:
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

Use OpenAI ChatGPT to generate a response based on the prompts.


```python
response = openai_client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_PROMPT},
    ],
)
print(response.choices[0].message.content)
```

    I don't know.


Here we can see, since we have replace the PII with masks, the LLM can not get the PII information in context. So it answers: "I don't know". 
Through this way, we can effectively protect the privacy of users. 
