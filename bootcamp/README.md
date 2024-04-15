## Milvus Bootcamp

This bootcamp is designed to accelerate users of Milvus on their AI journey.

Three ways to use this bootcamp.
1. **Self-service training.** You can bring your own data, and follow along in the ["Cheat Sheet"](MilvusCheatSheet.md), starting with "Getting Started with Milvus Tutorial". Best Practice theory (watch for 💡👉🏼 pointers) is mixed-in along with Tutorial instructions.  You'll need time to customize the notebooks below to your data.
2. **Zilliz-led hands-on, training.**  For best results, we will ask for a sample of your anonymized POC data at least 1 week in advance.
3. **Milvus Starling Hero-led hands-on, training.** See the separate instructions. For best results, ask for a sample of anonymized POC data at least 1 week in advance. You'll need time to customize the notebooks to their data.


### 📖 Tutorial Notebooks

| Description | Reading | Milvus APIs used | Notebook | Video
|:--------------------------------------------------------------|:-------|:-------|:-------|:-------|
| HIGH-LEVEL INTRO TO VECTORS, EMBEDDINGS, VECTOR SEARCH, MILVUS VECTOR DATABASE | [Vector Database 101](https://zilliz.com/learn/what-is-vector-database)  |  |  | 
| CONNECT TO MILVUS OR ZILLIZ | [Getting Started with a Milvus Connection](https://zilliz.com/blog/getting-started-with-a-milvus-connection) | [- Milvus Connections](https://milvus.io/docs/manage_connection.md) | [Connect to Milvus/Zilliz notebook](milvus_connect.ipynb) | COMING SOON! |
| RETRIEVAL <br>-Choose an embedding model <br>-Choose a vector index <br>-Load data <br>-Semantic search | [How to choose the right vector index](https://zilliz.com/learn/choosing-right-vector-index-for-your-project)| [- Milvus Search](https://milvus.io/docs/single-vector-search.md) <br> [- HNSW Index](https://milvus.io/docs/index.md) | [Milvus search with lab exercises notebook](Retrieval/imdb_milvus_client.ipynb) | COMING SOON! |
| RAG <br>- Open-source (HuggingFace) and closed-source LLMs (OpenAI) <br>- Grounding LLMs with source citations | [Build an Open Source Chatbot](https://zilliz.com/blog/building-open-source-chatbot-using-milvus-and-langchain-in-5-minutes)| APIs same as above plus <br> [- LangChain](https://milvus.io/docs/integrate_with_langchain.md) | [ReadTheDocs RAG notebook](RAG/readthedocs_zilliz_langchain.ipynb) | COMING SOON! |
| EVALUATE RAG | [Optimizing RAG Evaluation Methodology](https://zilliz.com/blog/how-to-evaluate-retrieval-augmented-generation-rag-applications)| [- Fiqa data](https://huggingface.co/datasets/explodinggradients/fiqa) <br>[- Ragas](https://github.com/explodinggradients/ragas) | [Evaluation Ragas notebook](https://github.com/milvus-io/bootcamp/blob/master/evaluation/evaluate_fiqa_customized_RAG.ipynb) | COMING SOON! |
| AGENTS | [Limitations of OpenAI for custom RAG](https://zilliz.com/blog/customizing-openai-built-in-retrieval-using-milvus-vector-database)| APIs same as above plus <br> [- LlamaIndex](https://milvus.io/docs/integrate_with_llamaindex.md) <br>[- OpenAI Assistants](https://platform.openai.com/docs/assistants/overview)| [OpenAI Assistant with LlamaIndex notebook](OpenAIAssistants/milvus_agent_llamaindex.ipynb) | COMING SOON! |