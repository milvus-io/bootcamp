# Search with Jev and Milvus

Milvus retrieves candidate evidence. Jev makes bounded semantic judgments about that evidence: which results to keep, where to search, and when to stop. These short, runnable notebooks show the inputs, questions, outputs and Python control flow for each step.

The examples use synthetic documents and small candidate pools. They demonstrate integration patterns, not retrieval-quality benchmarks. Each notebook is self-contained and uses Milvus Lite by default, with server and Zilliz Cloud connection options.

## Choose a scenario

| Notebook | What you will build |
| --- | --- |
| [Rerank search results with Jev](https://github.com/milvus-io/bootcamp/blob/master/bootcamp/RAG/search_with_jev/rerank_search_results.ipynb) | Retrieve candidates with Milvus, then ask Jev which passages answer a question and satisfy a business preference. |
| [Filter retrieved context with Jev](https://github.com/milvus-io/bootcamp/blob/master/bootcamp/RAG/search_with_jev/filter_search_context.ipynb) | Keep passages that provide evidence for the current question before building a generation prompt. |
| [Decide when to stop searching with Jev](https://github.com/milvus-io/bootcamp/blob/master/bootcamp/RAG/search_with_jev/decide_search_stopping.ipynb) | Run at most two retrieval rounds and stop as soon as the evidence supports the answer. |
| [Rerank graph relations with Jev](https://github.com/milvus-io/bootcamp/blob/master/bootcamp/RAG/search_with_jev/rerank_graph_relations.ipynb) | Retrieve relationship records from Milvus, score both direct evidence and useful bridge relations, and carry the relationship order into the source-document order. |
| [Route search queries with Jev](https://github.com/milvus-io/bootcamp/blob/master/bootcamp/RAG/search_with_jev/route_search_queries.ipynb) | Choose a known search scope, then let Milvus apply an exact metadata filter. |
| [Check whether a cached answer can be reused](https://github.com/milvus-io/bootcamp/blob/master/bootcamp/RAG/search_with_jev/validate_semantic_cache.ipynb) | Use Milvus to find a similar cached request, then check task and output compatibility with Jev. |
| [Curate documents before indexing with Jev](https://github.com/milvus-io/bootcamp/blob/master/bootcamp/RAG/search_with_jev/curate_search_data.ipynb) | Judge whether incoming documents contain substantive operational guidance. |
| [Screen retrieved passages with Jev](https://github.com/milvus-io/bootcamp/blob/master/bootcamp/RAG/search_with_jev/check_search_guardrails.ipynb) | Flag text that attempts to redirect an assistant rather than provide evidence. |
| [Evaluate search evidence with Jev](https://github.com/milvus-io/bootcamp/blob/master/bootcamp/RAG/search_with_jev/evaluation_with_jev.ipynb) | Use Jev as a judge after retrieval: assess passage relevance, evidence sufficiency and whether a proposed answer is supported. |

## Run a notebook

Clone the repository and open `bootcamp/RAG/search_with_jev`. The examples are tested with Python 3.12:

```bash
uv sync --python 3.12
export TYPESAFE_API_KEY=your_key
uv run jupyter lab
```

Alternatively, use the Colab badge in an individual notebook and uncomment its dependency installation cell. Obtain a TypeSafe API key from [TypeSafe](https://console.typesafe.ai/). The examples pin `jev-1.13.0`; set `JEV_MODEL` if your account needs another supported version. Calls use credits and send the displayed sample data to TypeSafe.

The small TF-IDF encoder avoids a GPU, an embedding-model download and a second API key. Replace it with your preferred dense embedding model for real semantic retrieval. Metadata constraints and authorization remain in application code; Jev evaluates the semantic conditions after those boundaries are applied.

Markdown files are the editable source. Regenerate notebooks with `uv run python sync_notebooks.py`, then verify them with `uv run python sync_notebooks.py --check`. This wraps `jupyter-switch` while preserving Markdown line breaks and saved outputs for unchanged code. Example outputs vary by model version. An HTTP or response-validation error stops execution; production applications should define an explicit fallback.

## Related projects and experiments

These projects provide larger examples beyond the synthetic notebooks. Follow each link for its current integration status and evaluation scope.

| Project | Search use case | Jev work |
| --- | --- | --- |
| [MemSearch](https://github.com/zilliztech/memsearch) | Persistent Markdown memory for coding agents | [Merged reranker and bilingual evaluation](https://github.com/zilliztech/memsearch/pull/758) |
| [Vector Graph RAG](https://github.com/zilliztech/vector-graph-rag) | Vector and graph retrieval for multi-hop questions | [Merged relationship reranking and evaluation](https://github.com/zilliztech/vector-graph-rag/pull/48) |
| [DeepSearcher](https://github.com/zilliztech/deep-searcher) | Iterative search over private knowledge | [Search-stopping experiment on a personal fork](https://github.com/zc277584121/deep-searcher/tree/experiment/jev-search-stopping/evaluation/jev_stopping), not an upstream feature |
| [GPTCache](https://github.com/zilliztech/GPTCache) | Reuse answers to compatible requests | [Jev implementation PR](https://github.com/zilliztech/GPTCache/pull/701) and [cache-compatibility benchmark PR](https://github.com/zilliztech/GPTCache/pull/702) |

The linked studies have different tasks, datasets and comparison methods. Their results are not scores for these notebooks. Some studies use private data and publish aggregates rather than a fully reproducible public corpus. API estimates and simulated animations should not be interpreted as controlled end-to-end speed measurements.

## References

- [TypeSafe primitives and independent batched questions](https://docs.typesafe.ai/primitives)
- [Milvus Lite](https://milvus.io/docs/milvus_lite.md)
- [Zilliz Cloud](https://zilliz.com/cloud)
