# MemPalace with Milvus

[MemPalace](https://github.com/MemPalace/mempalace) is a memory layer for coding agents and long-running development workflows. It organizes project knowledge into wings, rooms, and drawers, then makes the original content searchable across sessions.

In this tutorial, we will use the MemPalace CLI to mine a real subset of the public [Milvus documentation](https://github.com/milvus-io/milvus-docs) and store it in [Milvus](https://milvus.io/). The corpus contains documentation about analyzers, tokenizers, and token filters. These closely related pages provide enough distractors to make the retrieval examples meaningful.

The example uses Milvus Lite, so it runs locally without Docker or a separate database server. The same MemPalace configuration can also point to Milvus server or Zilliz Cloud for shared deployments.

## Prerequisites

Install MemPalace with its optional Milvus dependencies from PyPI. The command intentionally does not pin a version, so a new installation resolves the latest available release.

```shell
uv tool install "mempalace[milvus]"
```

You also need Git to download the documentation corpus.

This tutorial uses MemPalace's local MiniLM embedding model, so it does not require an external model API key. The first mining or search command may download a small ONNX embedding model.

## Configure the workspace

Create a workspace with separate directories for the documentation and the palace:

```shell
mkdir -p mempalace-milvus-demo
cd mempalace-milvus-demo

export PALACE_DIR="$PWD/palace"
export DOCS_REPO="$PWD/milvus-docs"
export PROJECT_DIR="$PWD/milvus-analyzer-docs"
export MEMPALACE_EMBEDDING_MODEL="minilm"
export MEMPALACE_EMBEDDING_DEVICE="cpu"
export MEMPALACE_EMBEDDING_THREADS="2"
```

We pass `--backend milvus` to the MemPalace commands below. Because no remote Milvus URI is configured, MemPalace creates a local Milvus Lite database at `$PALACE_DIR/milvus.db`.

> As for the argument of `MilvusClient` used by the backend:
>
> - Setting the `uri` to a local path, such as `./milvus.db`, is the most convenient option. It automatically uses [Milvus Lite](https://milvus.io/docs/milvus_lite.md) to store data locally.
> - For a larger deployment, you can use a [Milvus server](https://milvus.io/docs/quickstart.md) and set the URI to its endpoint, such as `http://localhost:19530`.
> - To use [Zilliz Cloud](https://zilliz.com/cloud), set the URI and token to the cluster's [Public Endpoint and API key](https://docs.zilliz.com/docs/on-zilliz-cloud-console#free-cluster-details).

## Download the Milvus documentation corpus

The Milvus documentation repository is much larger than this example needs. Use Git sparse checkout to download only the Analyzer documentation directory from the `v3.0.x` branch:

```shell
git clone \
  --depth 1 \
  --filter=blob:none \
  --sparse \
  --branch v3.0.x \
  https://github.com/milvus-io/milvus-docs.git \
  "$DOCS_REPO"

git -C "$DOCS_REPO" sparse-checkout set \
  site/en/userGuide/schema/analyzer

cp -R \
  "$DOCS_REPO/site/en/userGuide/schema/analyzer" \
  "$PROJECT_DIR"
```

At the time of writing, this directory contains 31 Markdown pages. They include general Analyzer guides and three groups of closely related pages:

```text
milvus-analyzer-docs/
├── analyzer/       # Built-in language analyzers
├── filter/         # Token filters
├── tokenizer/      # Tokenizers
└── *.md            # Analyzer overviews and selection guides
```

Confirm the number of source pages:

```shell
find "$PROJECT_DIR" -type f -name "*.md" | wc -l
```

Reference output:

```text
31
```

The exact count may change as the Milvus documentation branch is updated.

## Define the MemPalace rooms

MemPalace can detect rooms during `mempalace init`, but its initialization flow also performs project-wide heuristic entity classification and writes the accepted results to an entity registry. That classification step is not needed to define this documentation corpus, so we provide the small taxonomy directly. During mining, MemPalace may still attach deterministic heuristic entity metadata and build internal hallway links; those associations do not decide which room receives a file or change the room-scoped searches below.

Create `$PROJECT_DIR/mempalace.yaml` with the following content:

```yaml
wing: milvus_analyzer_docs
rooms:
  - name: analyzer
    description: Built-in language analyzers and analyzer selection guides
    keywords:
      - analyzer
  - name: filter
    description: Token filters used in analyzer pipelines
    keywords:
      - filter
  - name: tokenizer
    description: Tokenizers and language identification
    keywords:
      - tokenizer
  - name: general
    description: Analyzer documentation that does not fit another room
    keywords: []
```

The wing represents the whole documentation corpus. A room represents a topic area. MemPalace routes a file by checking its directory first, then its filename, then room keywords in its content. A file under `filter/`, for example, goes directly to the `filter` room.

Each file is then split into overlapping text chunks. Every chunk becomes a drawer containing the verbatim Markdown and metadata such as `wing`, `room`, `source_file`, `chunk_index`, and source line numbers. The rooms and drawers remain logical metadata inside MemPalace's Milvus collections; MemPalace does not create a separate Milvus collection for every room.

## Mine the documentation into Milvus

Mine the project with the Milvus backend:

```shell
mempalace \
  --palace "$PALACE_DIR" \
  mine "$PROJECT_DIR" \
  --backend milvus
```

Reference output from the validated documentation snapshot:

```text
=======================================================
  Done.
  Files processed: 31
  Files skipped (already filed or other): 0
  Drawers filed: 473

  By room:
    filter               16 files
    analyzer              8 files
    tokenizer             7 files
=======================================================
```

MemPalace reads the Markdown without summarizing or rewriting it, computes local embeddings, and stores the drawers in Milvus. On the tested documentation snapshot, 31 files produced 473 drawers.

Check the resulting rooms and drawer counts:

```shell
mempalace --palace "$PALACE_DIR" status --backend milvus
```

Reference output:

```text
=======================================================
  MemPalace Status -- 473 drawers
=======================================================

  WING: milvus_analyzer_docs
    ROOM: analyzer               212 drawers
    ROOM: filter                 156 drawers
    ROOM: tokenizer              105 drawers

=======================================================
```

The exact drawer count can change when the upstream documentation changes because longer pages produce more chunks.

## Semantic search

Use `mempalace search` to retrieve documentation by meaning. The following question does not name a specific file or Analyzer feature:

```shell
mempalace \
  --palace "$PALACE_DIR" \
  search "How should I analyze documents that mix several languages?" \
  --backend milvus \
  --wing milvus_analyzer_docs \
  --results 3
```

Reference output (scores may vary):

```text
Results for: "How should I analyze documents that mix several languages?"
Wing: milvus_analyzer_docs

[1] milvus_analyzer_docs / analyzer
    Source: multi-language-analyzers.md
    Match: cosine_sim=0.334 bm25=2.469
[2] milvus_analyzer_docs / analyzer
    Source: multi-language-analyzers.md
[3] milvus_analyzer_docs / analyzer
    Source: multi-language-analyzers.md
```

In the validated run, all three results came from `multi-language-analyzers.md`, even though the corpus also contained pages for individual language analyzers, tokenizers, and filters.

## Search within a room

Room filters are useful when related concepts appear throughout the corpus. The following query searches only the `filter` room for a way to make equivalent terms match:

```shell
mempalace \
  --palace "$PALACE_DIR" \
  search "How can equivalent terms such as USA and United States match one another?" \
  --backend milvus \
  --wing milvus_analyzer_docs \
  --room filter \
  --results 3
```

Reference output (scores may vary):

```text
Results for: "How can equivalent terms such as USA and United States match one another?"
Wing: milvus_analyzer_docs
Room: filter

[1] milvus_analyzer_docs / filter
    Source: synonym-filter.md
    Match: cosine_sim=0.765 bm25=2.573
[2] milvus_analyzer_docs / filter
    Source: stemmer-filter.md
[3] milvus_analyzer_docs / filter
    Source: stop-filter.md
```

The top result should come from `synonym-filter.md`. The room constraint is applied through drawer metadata before the vector search, so tokenizer and language-analyzer drawers are excluded from this search.

## Search for exact terms

The MemPalace CLI combines semantic similarity with BM25 signals when ranking the vector-search candidates. Exact configuration names and feature names can therefore improve the ranking without switching to a separate CLI search mode.

```shell
mempalace \
  --palace "$PALACE_DIR" \
  search "language_identifier tokenizer" \
  --backend milvus \
  --wing milvus_analyzer_docs \
  --room tokenizer \
  --results 3
```

Reference output (scores may vary):

```text
Results for: "language_identifier tokenizer"
Wing: milvus_analyzer_docs
Room: tokenizer

[1] milvus_analyzer_docs / tokenizer
    Source: language-identifier.md
    Match: cosine_sim=0.420 bm25=0.969
[2] milvus_analyzer_docs / tokenizer
    Source: language-identifier.md
[3] milvus_analyzer_docs / tokenizer
    Source: lindera-tokenizer.md
```

The results should favor `language-identifier.md`, which documents the `language_identifier` tokenizer used to select analyzers based on detected language.

## Inspect the Milvus collections

MemPalace manages its Milvus schema automatically. To confirm what was stored, save the following script as `inspect_milvus.py`. It opens the same Milvus Lite database, inspects the collections, and counts drawers by room:

```python
import os
from collections import Counter

from pymilvus import MilvusClient


client = MilvusClient(uri=os.environ["MEMPALACE_MILVUS_LITE_PATH"])

for collection_name in sorted(client.list_collections()):
    stats = client.get_collection_stats(collection_name)
    schema = client.describe_collection(collection_name)
    fields = [field["name"] for field in schema["fields"]]
    print(f"{collection_name}: rows={stats['row_count']}, fields={fields}")

client.load_collection("mempalace_drawers")
rows = client.query(
    collection_name="mempalace_drawers",
    filter='metadata["wing"] == "milvus_analyzer_docs"',
    limit=2000,
    output_fields=["metadata"],
)
room_counts = Counter(row["metadata"]["room"] for row in rows)
print("Drawers by room:", dict(sorted(room_counts.items())))
```

Run the script with the same optional dependency set used by the CLI:

```shell
export MEMPALACE_MILVUS_LITE_PATH="$PALACE_DIR/milvus.db"
uv run --with "mempalace[milvus]" inspect_milvus.py
```

Reference output:

```text
mempalace_closets: rows=74, fields=['id', 'document', 'metadata', 'vector', 'sparse']
mempalace_drawers: rows=473, fields=['id', 'document', 'metadata', 'vector', 'sparse']
Drawers by room: {'analyzer': 212, 'filter': 156, 'tokenizer': 105}
```

For the tested documentation snapshot, `mempalace_drawers` contained 473 rows and `mempalace_closets` contained 74 internal navigation records. Closet and drawer counts do not need to match. The drawer metadata showed 212 drawers in `analyzer`, 156 in `filter`, and 105 in `tokenizer`.

This inspection runs in a new process and reopens the database created by the CLI, which also confirms that the data persists across commands.

## Optional: use Milvus server or Zilliz Cloud

For a shared deployment, set the Milvus connection environment variables before running the same MemPalace CLI commands. Leave them unset to use the local Milvus Lite database shown above.

For Milvus server:

```shell
export MEMPALACE_MILVUS_URI="http://localhost:19530"
export MEMPALACE_MILVUS_DB_NAME="default"
export MEMPALACE_MILVUS_NAMESPACE="team-memory"
```

For Zilliz Cloud:

```shell
export MEMPALACE_MILVUS_URI="https://your-cluster.api.region.zillizcloud.com"
export MEMPALACE_MILVUS_TOKEN="your-api-key"
export MEMPALACE_MILVUS_DB_NAME="default"
export MEMPALACE_MILVUS_NAMESPACE="team-memory"
```

The end-to-end commands in this tutorial were validated with Milvus Lite. The server and cloud settings above are optional deployment configurations and were not required for the local validation.

## Conclusion

MemPalace gives agents a structured way to preserve project knowledge: a wing separates the corpus, rooms provide topic-level scope, and drawers retain the original source text. In this example, 31 closely related Milvus documentation pages become hundreds of searchable drawers rather than a few hand-written records. Milvus provides persistent vector, sparse, text, and metadata storage behind that structure.
