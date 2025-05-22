# Build RAG with Llama Stack with Milvus  
[Llama Stack](https://github.com/meta-llama/llama-stack/tree/main) is a service-oriented, API-first approach for building production AI applications. It provides a universal stack that allows developers to develop anywhere, deploy everywhere, and leverage production-ready building blocks with true provider independence. The Llama Stack focuses on Meta's Llama models, composability, production-readiness, and a partnering ecosystem.

In this tutorial, we will introduce how to build a Llama Stack Server configured with Milvus, enabling you to import your private data to serve as your knowledge base. We will then perform queries on the server, creating a complete RAG application.

## Preparing the Environment
There are many ways to start the Llama Stack server, such as [as a library](https://llama-stack.readthedocs.io/en/latest/distributions/importing_as_library.html), [building a distribution](https://llama-stack.readthedocs.io/en/latest/distributions/building_distro.html), etc. For each component in the Llama Stack, various providers can also be chosen. Therefore, there are numerous ways to launch the Llama Stack server.  

This tutorial uses the following configuration as an example to start the service. If you wish to start it in another way, please refer to [Starting a Llama Stack Server](https://llama-stack.readthedocs.io/en/latest/distributions/index.html).
- We use Conda to build a custom distribution with Milvus configuration.
- We use [Together AI](https://llama-stack.readthedocs.io/en/latest/distributions/self_hosted_distro/together.html#via-conda) as the LLM provider.
- We use the default `all-MiniLM-L6-v2` as the embedding model.  
> This tutorial mainly refers to the official installation guide of the [Llama Stack documentation](https://llama-stack.readthedocs.io/en/latest/index.html). If you find any outdated parts in this tutorial, you can prioritize following the official guide and create an issue for us.

## Start Llama Stack Server
### Prepare the Environment
Since we need to use Together AI as the LLM service, we must first log in to the official website to apply for an [API key](https://api.together.xyz/settings/api-keys) and set the API key `TOGETHER_API_KEY` as an environment variable. 

Clone the Llama Stack source code
```
git clone https://github.com/meta-llama/llama-stack.git
cd llama-stack
```  
Create a conda environment and install dependencies
```
conda create -n stack python=3.10
conda activate stack

pip install -e .
```  
Modify the content in `llama_stack/llama_stack/template/together/run.yaml`, changing the vector_io section to the relevant Milvus configuration. For example, add:
```yaml
vector_io:
- provider_id: milvus
  provider_type: inline::milvus
  config:
    db_path: ~/.llama/distributions/together/milvus_store.db

#  - provider_id: milvus
#    provider_type: remote::milvus
#    config:
#      uri: http://localhost:19530
#      token: root:Milvus
```
In Llama Stack, Milvus can be configured in two ways: local configuration, which is `inline::milvus`, and remote configuration, which is `remote::milvus`.
- The simplest method is local configuration, which requires setting `db_path`, a path for locally storing [Milvus-Lite](https://milvus.io/docs/quickstart.md) files.
- Remote configuration is suitable for large data storage.
    - If you have a large amount of data, you can set up a performant Milvus server on [Docker or Kubernetes](https://milvus.io/docs/quickstart.md). In this setup, please use the server URI, e.g., `http://localhost:19530`, as your `uri`. The default `token` is `root:Milvus`.
    - If you want to use [Zilliz Cloud](https://zilliz.com/cloud), the fully managed cloud service for Milvus, adjust the `uri` and `token`, which correspond to the [Public Endpoint and API key](https://docs.zilliz.com/docs/on-zilliz-cloud-console#free-cluster-details) in Zilliz Cloud.

### Build distribution from the template
Run the following command to build the distribution:
```
llama stack build --template together --image-type conda
```
A file will be generated at `~/.llama/distributions/together/together-run.yaml`. Then, run this command to start the server:
```
llama stack run --image-type conda ~/.llama/distributions/together/together-run.yaml
```
If everything goes smoothly, you should see the Llama Stack server successfully running on port 8321.

## Perform RAG from client  
Once you have started the server, you can write the client code to access it. Here is a sample code:

```python
import uuid
from llama_stack_client.types import Document
from llama_stack_client.lib.agents.agent import Agent
from llama_stack_client.types.agent_create_params import AgentConfig

# See https://www.together.ai/models for all available models
INFERENCE_MODEL = "meta-llama/Llama-3.3-70B-Instruct-Turbo"
LLAMA_STACK_PORT = 8321


def create_http_client():
    from llama_stack_client import LlamaStackClient

    return LlamaStackClient(
        base_url=f"http://localhost:{LLAMA_STACK_PORT}"  # Your Llama Stack Server URL
    )


client = create_http_client()

# Documents to be used for RAG
urls = ["chat.rst", "llama3.rst", "memory_optimizations.rst", "lora_finetune.rst"]
documents = [
    Document(
        document_id=f"num-{i}",
        content=f"https://raw.githubusercontent.com/pytorch/torchtune/main/docs/source/tutorials/{url}",
        mime_type="text/plain",
        metadata={},
    )
    for i, url in enumerate(urls)
]

# Register a vector database
vector_db_id = f"test-vector-db-{uuid.uuid4().hex}"
client.vector_dbs.register(
    vector_db_id=vector_db_id,
    embedding_model="all-MiniLM-L6-v2",
    embedding_dimension=384,
    provider_id="milvus",
)

print("inserting...")
# Insert the documents into the vector database
client.tool_runtime.rag_tool.insert(
    documents=documents, vector_db_id=vector_db_id, chunk_size_in_tokens=1024,
)

agent_config = AgentConfig(
    model=INFERENCE_MODEL,
    # Define instructions for the agent ( aka system prompt)
    instructions="You are a helpful assistant",
    enable_session_persistence=False,
    # Define tools available to the agent
    toolgroups=[{"name": "builtin::rag", "args": {"vector_db_ids": [vector_db_id]}}],
)

rag_agent = Agent(client, agent_config)
session_id = rag_agent.create_session("test-session")
print("finish init agent...")
user_prompt = (
    "What are the top 5 topics that were explained? Only list succinct bullet points."
)

# Get the final answer from the agent
response = rag_agent.create_turn(
    messages=[{"role": "user", "content": user_prompt}],
    session_id=session_id,
    stream=False,
)
print(f"Response: ")
print(response.output_message.content)
```
Run this code to perform the RAG query.
If everything is working properly, the output should look like this:

```log
inserting...
finish init agent...
Response: 
* Fine-Tuning Llama3 with Chat Data
* Evaluating fine-tuned Llama3-8B models with EleutherAI's Eval Harness
* Generating text with our fine-tuned Llama3 model
* Faster generation via quantization
* Fine-tuning on a custom chat dataset
```


