# Search with Jev and Milvus

Vector search finds information related to a query. Building a useful search application also involves decisions: which passages actually answer the question, whether an earlier answer can be reused, and whether an agent has enough evidence to stop searching.

Milvus and Jev address different parts of this workflow. [Milvus](https://milvus.io/) stores embeddings and retrieves candidate records, with metadata filters for constraints such as product version or knowledge-base scope. [Jev](https://docs.typesafe.ai/introduction) evaluates the meaning of the retrieved text against instructions. Your application can use its judgments to select evidence or control the next search step.

## What does Jev do?

A Jev request supplies context and one or more judgment questions. Its [typed outputs](https://docs.typesafe.ai/primitives) include a choice among fixed options, an ordered score and a yes/no probability. These outputs let application code make a decision without parsing a free-form explanation. A generation model can still write an answer or a follow-up search query when needed.

For example, a user asks how to install Atlas v2. Milvus can restrict retrieval to v2 documentation and return similar passages about installation, upgrades and troubleshooting. Jev then evaluates which passages explain the initial setup. The application passes the selected evidence to an answer-generating model.

The responsibilities are straightforward:

1. **Retrieve with Milvus:** find candidates within the required metadata constraints.
2. **Judge with Jev:** evaluate those candidates against the question and a task-specific criterion.
3. **Act in application code:** reorder results, filter context, reuse an answer or continue searching.

Some decisions happen before retrieval. Jev can choose a search scope or assess incoming documents before they enter a collection. Access control and exact filters remain the application's responsibility.

## Explore the search scenarios

The [Search with Jev collection](https://github.com/milvus-io/bootcamp/tree/master/bootcamp/RAG/search_with_jev) contains nine runnable tutorials. Each uses a small synthetic dataset and shows the retrieved records, judgments and resulting action.

### Select better evidence

- [Rerank search results](https://github.com/milvus-io/bootcamp/blob/master/bootcamp/RAG/search_with_jev/rerank_search_results.ipynb): reorder documentation and coding-agent memories. A memory about a laptop port error may resemble a container connection problem; a more useful memory records the actual container-host fix.
- [Filter retrieved context](https://github.com/milvus-io/bootcamp/blob/master/bootcamp/RAG/search_with_jev/filter_search_context.ipynb): distinguish initial-installation instructions from upgrade and troubleshooting passages after Milvus applies the version filter.
- [Rerank graph relations](https://github.com/milvus-io/bootcamp/blob/master/bootcamp/RAG/search_with_jev/rerank_graph_relations.ipynb): answer a question about a book author's birthplace by selecting both the book-to-author bridge and the author-to-birthplace relation, then preserve that ranking when fetching source passages.

### Control search and answer reuse

- [Decide when to stop searching](https://github.com/milvus-io/bootcamp/blob/master/bootcamp/RAG/search_with_jev/decide_search_stopping.ipynb): a generation model proposes searches from accumulated evidence, while Jev judges whether the original question is answerable. The examples cover a direct answer, a two-hop question and an unavailable fact that reaches the search limit without an answer.
- [Route search queries](https://github.com/milvus-io/bootcamp/blob/master/bootcamp/RAG/search_with_jev/route_search_queries.ipynb): select documentation, billing or memory search, then apply the corresponding Milvus filter. An out-of-scope query takes a separate path.
- [Validate semantic cache reuse](https://github.com/milvus-io/bootcamp/blob/master/bootcamp/RAG/search_with_jev/validate_semantic_cache.ipynb): retrieve a similar cached request, then check whether its answer also satisfies the new request's task, language and context requirements.

### Improve and inspect the knowledge pipeline

- [Curate documents before indexing](https://github.com/milvus-io/bootcamp/blob/master/bootcamp/RAG/search_with_jev/curate_search_data.ipynb): distinguish substantive operational guidance from promotional or incomplete material, with separate index, review and exclude actions.
- [Screen retrieved passages](https://github.com/milvus-io/bootcamp/blob/master/bootcamp/RAG/search_with_jev/check_search_guardrails.ipynb): identify text that tries to redirect an assistant, while retaining ordinary security advice. This is an additional screening step, not a security guarantee.
- [Evaluate search evidence](https://github.com/milvus-io/bootcamp/blob/master/bootcamp/RAG/search_with_jev/evaluation_with_jev.ipynb): judge passage relevance, whether evidence is sufficient and whether an answer makes unsupported claims. The examples deliberately remove evidence or add an unsupported statement to make the distinction visible.

## A ready-made reranking interface

[Milvus Model](https://github.com/milvus-io/milvus-model) provides an application-side `JevRerankFunction`: pass a query and candidate document texts, and receive scored results with their original indices, sorted by relevance. Use those indices to reorder the records returned by Milvus.

The [Jev integration](https://github.com/milvus-io/milvus-model/pull/90) has been merged. See the [implementation and constructor options](https://github.com/milvus-io/milvus-model/blob/main/src/pymilvus/model/reranker/jev.py) for the current API. It accepts `TYPESAFE_API_KEY` and defaults to `jev-latest`. Check that your installed package includes this integration before importing it; a merged change does not establish availability in a published package.

The current wrapper uses a claim-and-evidence relevance prompt. Check that this criterion fits your task. For custom judgments such as memory compatibility, stopping or routing, follow the linked tutorials using the TypeSafe API directly. The notebooks use that API directly and do not require the Milvus Model wrapper. Neither approach adds a Jev model to the Milvus server.

## Try it with Milvus

[Open the reranking tutorial in Colab](https://colab.research.google.com/github/milvus-io/bootcamp/blob/master/bootcamp/RAG/search_with_jev/rerank_search_results.ipynb) to start with candidate retrieval and ranking. For local setup and the full tutorial list, see the [collection README](https://github.com/milvus-io/bootcamp/blob/master/bootcamp/RAG/search_with_jev/README.md).

The examples use a [Gemini API key](https://aistudio.google.com/apikey) for embeddings and a [TypeSafe API key](https://console.typesafe.ai/) for Jev. The agentic-search tutorial also uses Gemini for query and answer generation. Sample text is sent to these API providers, and calls may consume credits.

The tutorials run with [Milvus Lite](https://milvus.io/docs/milvus_lite.md) by default and include connection options for a Milvus server or [Zilliz Cloud](https://zilliz.com/cloud). The same division of work applies across deployments: Milvus retrieves candidates, and the application sends the relevant text to Jev for judgment.

Treat the examples as starting points for your own criteria and thresholds. A relevance score does not guarantee an answer is correct, and these small teaching datasets do not establish production accuracy or speed.

## Explore implementations and evaluation results

The following open-source projects apply these ideas to larger search workflows. Their linked reports explain the datasets, comparisons and limitations of each experiment.

| Project | Search use case | Jev work |
| --- | --- | --- |
| [MemSearch](https://github.com/zilliztech/memsearch) | Persistent Markdown memory for coding agents | [Jev implementation](https://github.com/zilliztech/memsearch/blob/main/src/memsearch/jev_reranker.py) · [Evaluation](https://github.com/zilliztech/memsearch/blob/main/evaluation/reranking-evaluation.md) |
| [Vector Graph RAG](https://github.com/zilliztech/vector-graph-rag) | Vector and graph retrieval for multi-hop questions | [Jev implementation](https://github.com/zilliztech/vector-graph-rag/blob/main/src/vector_graph_rag/llm/jev.py) · [Evaluation](https://github.com/zilliztech/vector-graph-rag/blob/main/evaluation/jev/README.md) |
| [DeepSearcher](https://github.com/zilliztech/deep-searcher) | Iterative search over private knowledge | [Experiment runner](https://github.com/zilliztech/deep-searcher/blob/master/evaluation/jev_stopping/run_full100.py) · [Search-stopping evaluation](https://github.com/zilliztech/deep-searcher/blob/master/evaluation/jev_stopping/README.md) (standalone experiment) |
| [GPTCache](https://github.com/zilliztech/GPTCache) | Reuse answers to compatible requests | [Jev implementation](https://github.com/zilliztech/GPTCache/blob/main/gptcache/similarity_evaluation/jev.py) · [Evaluation](https://github.com/zilliztech/GPTCache/blob/main/examples/benchmark/reuse_compatibility/README.md) |

DeepSearcher's contribution is a standalone search-stopping experiment. The other implementation links show task-specific Jev integrations. Results from these projects should be read in their own evaluation context, rather than treated as a shared benchmark.
