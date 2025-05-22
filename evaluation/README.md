# Evaluation 

Evaluate the performance of RAG pipelines based on [ragas](https://github.com/explodinggradients/ragas).


- [evaluate_fiqa_openai.ipynb](evaluate_fiqa_openai.ipynb) Use Ragas to evaluate the OpenAI Assistant
- [evaluate_fiqa_customized_RAG.ipynb](evaluate_fiqa_customized_RAG.ipynb) Use Ragas to evaluate the customized RAG pipeline based on milvus

The following outlines a comparison between two experimental setups:

|  | OpenAI assistant | Customized RAG pipeline |
| --- | --- | --- |
| LLM model | gpt-4-1106-preview | gpt-4-1106-preview |
| Vector DB | Not Disclosed | milvus |
| Embedding model | Not Disclosed | BAAI/bge-base-en |
| Chunk size | Not Disclosed | 1000 |
| Chunk overlap | Not Disclosed | 40 |
| topk | Not Disclosed | 5 |
| Use Agent | Yes | Yes |
