## Milvus Bootcamp

This bootcamp is designed to accelerate users of Milvus on their AI journey.

Three ways to use this workshop
1. **Self-service training.** You can bring your own data, and follow along in the ["Cheat Sheet"](MilvusCheatSheet.md), starting with the "Getting Started with Milvus Tutorial". Best Practice theory is mixed-in along with Tutorial instructions.  You'll need time to customize the notebooks below to your data.
2. **Zilliz-led hands-on, training.**  For best results, we will ask for a sample of your anonymized POC data at least 1 week in advance.
3. **Milvus Starling Hero-led hands-on, training.** See the separate instructions. For best results, ask for a sample of anonymized POC data at least 1 week in advance. You'll need time to customize the notebooks to their data.

For a quick-start 


### 📖 Tutorial Notebooks

| Description | Reading | Milvus APIs used | Notebook | Video
|:--------------------------------------------------------------|:-------|:-------|:-------|:-------|
| High-level introduction to Vectors, Embeddings, Vector search, Milvus vector database. | [Vector Database 101](https://zilliz.com/learn/what-is-vector-database)  |  |  | 
| Getting started connecting to Milvus. <br>- Local server (lite, docker, k8s). <br>- Zilliz cloud flavors free tier (serverless) or paid (managed aws, google, azure). | [Getting Started with a Milvus Connection](https://zilliz.com/blog/getting-started-with-a-milvus-connection) | [- Milvus Connections](https://milvus.io/docs/manage_connection.md) | [Connecting to Milvus/Zilliz notebook](milvus_connect.ipynb) | COMING SOON! |
| Loading and searching IMDB movie reviews with Milvus Client (no schema). <br>- Load IMDB moview reviews from a .csv file. <br>- Chunk and tokenize the data into tensors using open-source LLM. <br>- Save and search the tensors using nearest-neighbor algorithms. | [How to choose the right vector index](https://zilliz.com/learn/choosing-right-vector-index-for-your-project)| [- Milvus Search](https://milvus.io/api-reference/pymilvus/v2.3.x/Collection/search().md) <br>[- Milvus Client](https://pymilvus.readthedocs.io/en/latest/_modules/milvus/client/stub.html) <br> [- HNSW Index](https://milvus.io/docs/v2.0.x/index.md) | [Milvus search with lab exercises notebook](Retrieval/imdb_milvus_client.ipynb) | COMING SOON! |
| Building a RAG Chatbot on website data using open source LLMs (& also using OpenAI).  <br>- Suck in data from a website <br>- Open-source LLMs (HuggingFace) for retrieval and chat<br>- OpenAI ChatGPT <br>- Grounding LLMs with source citations (preventing hallucinations) | [Build an Open Source Chatbot](https://zilliz.com/blog/building-open-source-chatbot-using-milvus-and-langchain-in-5-minutes)| APIs same as above plus <br> [- LangChain](https://milvus.io/docs/integrate_with_langchain.md) | [OSS ReadTheDocs RAG notebook](RAG/readthedocs_zilliz_langchain.ipynb) | COMING SOON! |
| Evaluating RAG using Ragas and OpenAI | [Optimizing RAG Evaluation Methodology](https://zilliz.com/blog/how-to-evaluate-retrieval-augmented-generation-rag-applications)| [- Fiqa data](https://github.com/explodinggradients/ragas/blob/main/experiments/baselines/fiqa/dataset-exploration-and-baseline.ipynb) <br>[- Ragas](https://github.com/explodinggradients/ragas) | [Evaluation Ragas notebook](https://github.com/milvus-io/bootcamp/blob/master/evaluation/evaluate_fiqa_customized_RAG.ipynb) | COMING SOON! |
| Build an agent on top of the OpenAI Assistant API with a custom retriever tool. | [Limitations of OpenAI for custom RAG](https://zilliz.com/blog/customizing-openai-built-in-retrieval-using-milvus-vector-database)| [- LlamaIndex](https://milvus.io/docs/integrate_with_llama.md) <br>[OpenAI Assistants](https://platform.openai.com/docs/assistants/overview)| [OpenAI Assistant with LlamaIndex notebook](OpenAIAssistants/milvus_agent_llamaindex.ipynb) | COMING SOON! |