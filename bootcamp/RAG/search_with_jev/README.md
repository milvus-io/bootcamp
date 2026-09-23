# Search with Jev and Milvus

A search result can be similar to a question without answering it. A cached answer can match the topic but use the wrong software version. An agent can retrieve useful evidence and still need one more search.

[Milvus](https://milvus.io/) finds candidate records using vector similarity and metadata filters. [Jev](https://docs.typesafe.ai/introduction) evaluates text against instructions and returns structured judgments, such as a relevance score or a choice from predefined options. Your application uses those judgments to rank results, select context or decide its next step.

This collection shows how to combine the two across nine search scenarios. The notebooks use small synthetic datasets so you can inspect every candidate and decision. They are runnable integration examples, not quality benchmarks.

## Choose a scenario

| Notebook | What you will build |
| --- | --- |
| [Rerank search results with Jev](https://github.com/milvus-io/bootcamp/blob/master/bootcamp/RAG/search_with_jev/rerank_search_results.ipynb) | Retrieve candidates with Milvus, then ask Jev which passages answer a question and satisfy a business preference. |
| [Filter retrieved context with Jev](https://github.com/milvus-io/bootcamp/blob/master/bootcamp/RAG/search_with_jev/filter_search_context.ipynb) | Keep passages that provide evidence for the current question before building a generation prompt. |
| [Decide when to stop searching with Jev](https://github.com/milvus-io/bootcamp/blob/master/bootcamp/RAG/search_with_jev/decide_search_stopping.ipynb) | Let Gemini Flash generate evidence-driven queries; Jev gates stopping within three rounds, followed by a grounded answer or abstention. |
| [Rerank graph relations with Jev](https://github.com/milvus-io/bootcamp/blob/master/bootcamp/RAG/search_with_jev/rerank_graph_relations.ipynb) | Retrieve relationship records from Milvus, score both direct evidence and useful bridge relations, and carry the relationship order into the source-document order. |
| [Route search queries with Jev](https://github.com/milvus-io/bootcamp/blob/master/bootcamp/RAG/search_with_jev/route_search_queries.ipynb) | Choose a known search scope, then let Milvus apply an exact metadata filter. |
| [Check whether a cached answer can be reused](https://github.com/milvus-io/bootcamp/blob/master/bootcamp/RAG/search_with_jev/validate_semantic_cache.ipynb) | Use Milvus to find a similar cached request, then check task and output compatibility with Jev. |
| [Curate documents before indexing with Jev](https://github.com/milvus-io/bootcamp/blob/master/bootcamp/RAG/search_with_jev/curate_search_data.ipynb) | Judge whether incoming documents contain substantive operational guidance. |
| [Screen retrieved passages with Jev](https://github.com/milvus-io/bootcamp/blob/master/bootcamp/RAG/search_with_jev/check_search_guardrails.ipynb) | Flag text that attempts to redirect an assistant rather than provide evidence. |
| [Evaluate search evidence with Jev](https://github.com/milvus-io/bootcamp/blob/master/bootcamp/RAG/search_with_jev/evaluation_with_jev.ipynb) | Use Jev as a judge after retrieval: assess passage relevance, evidence sufficiency and whether a proposed answer is supported. |

Start with [reranking search results](https://github.com/milvus-io/bootcamp/blob/master/bootcamp/RAG/search_with_jev/rerank_search_results.ipynb) for a typical retrieval pipeline. It covers documentation search and Markdown memories for a coding agent: retrieve a broader set with Milvus, then use Jev to choose the most useful evidence. For an iterative agent, try [search stopping](https://github.com/milvus-io/bootcamp/blob/master/bootcamp/RAG/search_with_jev/decide_search_stopping.ipynb), where a generation model writes follow-up queries and Jev judges whether the accumulated evidence is sufficient.

## Use Jev through Milvus Model

[Milvus Model](https://github.com/milvus-io/milvus-model) provides an application-side `JevRerankFunction`: pass a query and candidate document texts, and receive scored results with their original indices, sorted by relevance. Use those indices to reorder the records returned by Milvus.

The [Jev integration](https://github.com/milvus-io/milvus-model/pull/90) has been merged. See the [implementation and constructor options](https://github.com/milvus-io/milvus-model/blob/main/src/pymilvus/model/reranker/jev.py) for the current API. It accepts `TYPESAFE_API_KEY` and defaults to `jev-latest`. Check that your installed package includes this integration before importing it; a merged change does not establish availability in a published package.

The current wrapper uses a claim-and-evidence relevance prompt. Check that this criterion fits your task. For custom judgments such as memory compatibility, stopping or routing, follow the linked tutorials using the TypeSafe API directly. The notebooks use that API directly and do not require the Milvus Model wrapper. Neither approach adds a Jev model to the Milvus server.

## Run a notebook

You need a Gemini API key from [Google AI Studio](https://aistudio.google.com/apikey) and a TypeSafe API key from the [TypeSafe console](https://console.typesafe.ai/). Gemini creates embeddings; Jev evaluates the sample text. The search-stopping example also uses Gemini Flash to generate queries and a final answer. These services may consume API credits.

Clone this repository and open `bootcamp/RAG/search_with_jev`. The examples were tested with Python 3.12:

```bash
uv sync --python 3.12
export GEMINI_API_KEY=your_gemini_key
export TYPESAFE_API_KEY=your_typesafe_key
uv run jupyter lab
```

Alternatively, [open the reranking notebook in Colab](https://colab.research.google.com/github/milvus-io/bootcamp/blob/master/bootcamp/RAG/search_with_jev/rerank_search_results.ipynb), or use the Colab badge in another notebook. Follow its setup cell to install dependencies.

The examples use [Gemini embeddings](https://ai.google.dev/gemini-api/docs/embeddings) and [Milvus Lite](https://milvus.io/docs/milvus_lite.md), with connection options for a Milvus server or [Zilliz Cloud](https://zilliz.com/cloud). They pin `jev-1.13.0` for the recorded examples; `JEV_MODEL` can select another supported model. The stopping notebook uses `gemini-3.8-flash`, configurable through `GEMINI_GENERATION_MODEL`. Outputs can change with model versions.

Keep exact constraints such as tenant access and software versions in application logic and Milvus filters. Use Jev for semantic decisions within those constraints, and tune selection thresholds on examples from your own workload.

## Implementations and evaluations in other projects

| Project | Search use case | Jev work |
| --- | --- | --- |
| [MemSearch](https://github.com/zilliztech/memsearch) | Persistent Markdown memory for coding agents | [Jev implementation](https://github.com/zilliztech/memsearch/blob/main/src/memsearch/jev_reranker.py) · [Evaluation](https://github.com/zilliztech/memsearch/blob/main/evaluation/reranking-evaluation.md) |
| [Vector Graph RAG](https://github.com/zilliztech/vector-graph-rag) | Vector and graph retrieval for multi-hop questions | [Jev implementation](https://github.com/zilliztech/vector-graph-rag/blob/main/src/vector_graph_rag/llm/jev.py) · [Evaluation](https://github.com/zilliztech/vector-graph-rag/blob/main/evaluation/jev/README.md) |
| [DeepSearcher](https://github.com/zilliztech/deep-searcher) | Iterative search over private knowledge | [Experiment runner](https://github.com/zilliztech/deep-searcher/blob/master/evaluation/jev_stopping/run_full100.py) · [Search-stopping evaluation](https://github.com/zilliztech/deep-searcher/blob/master/evaluation/jev_stopping/README.md) (standalone experiment) |
| [GPTCache](https://github.com/zilliztech/GPTCache) | Reuse answers to compatible requests | [Jev implementation](https://github.com/zilliztech/GPTCache/blob/main/gptcache/similarity_evaluation/jev.py) · [Evaluation](https://github.com/zilliztech/GPTCache/blob/main/examples/benchmark/reuse_compatibility/README.md) |

These links lead to implementation code and task-specific evaluation records. DeepSearcher's link is a standalone stopping experiment, not a default search-agent feature. The studies use different datasets and evaluation methods; consult each report before comparing results.
