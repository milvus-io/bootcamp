# Build Long-Term Agent Memory with EverOS and Milvus

[EverOS](https://github.com/EverMind-AI/EverOS) is a Markdown-first memory system for AI agents. It extracts durable memories from conversations, keeps Markdown as the source of truth, and builds a searchable derived index.

In this tutorial, we will build a project assistant that remembers release decisions across separate conversations. We will add two conversations about the Project Atlas launch and eight unrelated conversations about other projects. EverOS will use an LLM to extract the memories, while [Milvus](https://milvus.io/) stores the BM25 and vector indexes used for hybrid search.

```text
Conversations
      |
      v
EverOS + LLM ------> Markdown memory files
      |
      | embedding model
      v
Milvus ------> BM25 + vector hybrid search
```

The LLM and embedding model have different responsibilities. The LLM turns a conversation into structured memories. The embedding model converts those memories and later search queries into vectors. The basic hybrid search in this tutorial does not require a reranking model.

## Prerequisites

You need:

- Python 3.12 or later
- [`uv`](https://docs.astral.sh/uv/)
- A running Milvus Server
- An [OpenAI API key](https://platform.openai.com/api-keys)

This tutorial connects to Milvus Server at `http://localhost:19530`. EverOS also supports Zilliz Cloud through the same URI and token settings; the selected database must allow EverOS to create seven collections. Its Milvus backend expects a remote endpoint and does not accept a Milvus Lite file path.

## Install EverOS

Create a local project and install EverOS with its optional Milvus dependencies:

```shell
mkdir everos-milvus-demo
cd everos-milvus-demo

uv init --bare --python 3.12
uv add "everos[milvus]"
```

The command intentionally does not pin a version, so a new installation resolves the latest compatible EverOS release.

Initialize a separate memory root for the tutorial:

```shell
export EVEROS_ROOT="$PWD/everos-data"
uv run everos init --root "$EVEROS_ROOT"
```

EverOS creates `everos.toml` and `ome.toml` under this directory. It will also write the extracted memories here.

## Configure OpenAI and Milvus

Set the OpenAI API key and configure EverOS through environment variables:

```shell
export OPENAI_API_KEY="YOUR_OPENAI_API_KEY"
export MILVUS_URI="http://localhost:19530"

export EVEROS_INDEX__BACKEND="milvus"
export EVEROS_MILVUS__URI="$MILVUS_URI"
export EVEROS_MILVUS__COLLECTION_PREFIX="everos_bootcamp"

export EVEROS_LLM__MODEL="gpt-4.1-mini"
export EVEROS_LLM__API_KEY="$OPENAI_API_KEY"
export EVEROS_LLM__BASE_URL="https://api.openai.com/v1"

export EVEROS_EMBEDDING__MODEL="text-embedding-3-small"
export EVEROS_EMBEDDING__API_KEY="$OPENAI_API_KEY"
export EVEROS_EMBEDDING__BASE_URL="https://api.openai.com/v1"
export EVEROS_EMBEDDING__DIMENSIONS="1024"

export EVEROS_MEMORIZE__MODE="chat"
```

EverOS uses OpenAI for both memory extraction and embeddings. The embedding dimension is set to `1024` to match the Milvus schemas managed by EverOS.

The `chat` memory mode keeps this example focused on user memories. EverOS manages the Milvus collections and their schemas, so you do not need to create them yourself.

## Start EverOS

Start the EverOS HTTP server:

```shell
uv run everos server start --root "$EVEROS_ROOT"
```

Keep this terminal open. EverOS connects to Milvus and creates seven derived-index collections with the configured prefix during startup.

Open another terminal in the same project directory and check the service:

```shell
curl http://127.0.0.1:8000/health
```

Reference output:

```json
{
  "status": "ok",
  "version": "1.3.0",
  "capabilities": {
    "llm": true,
    "embed": true,
    "rerank": false,
    "multimodal_llm": false,
    "parser": true
  },
  "cascade": {
    "healthy": true,
    "pending": 0
  }
}
```

The response contains additional health fields. The important values for this tutorial are `status: "ok"`, `llm: true`, `embed: true`, and `cascade.healthy: true`.

## Add project conversations

The following Python program sends ten independent conversations to EverOS. Atlas has separate launch and rollback discussions. Eight conversations about other projects provide distractors so that the later search has to identify the correct project memories.

Save the following code as `add_memories.py`:

```python
import json
import time
from urllib.request import Request, urlopen


API_URL = "http://127.0.0.1:8000/api/v2/memory"
NOW = int(time.time() * 1000)

conversations = [
    (
        "atlas-release",
        [
            {
                "sender_id": "maya",
                "sender_name": "Maya",
                "role": "user",
                "timestamp": NOW,
                "content": (
                    "For Project Atlas, we decided to launch with a 10% canary "
                    "on September 30. Promote to all users only after the checkout "
                    "error rate stays below 1% for 30 minutes."
                ),
            },
            {
                "sender_id": "assistant",
                "role": "assistant",
                "timestamp": NOW + 1_000,
                "content": (
                    "Understood. I will remember the Atlas launch date, canary "
                    "percentage, and promotion gate."
                ),
            },
        ],
    ),
    (
        "atlas-rollback",
        [
            {
                "sender_id": "maya",
                "sender_name": "Maya",
                "role": "user",
                "timestamp": NOW + 10_000,
                "content": (
                    "The Atlas rollback owner is Priya. Roll back immediately if "
                    "checkout errors exceed 2% for five minutes, and keep the "
                    "previous container image available for 24 hours."
                ),
            },
            {
                "sender_id": "assistant",
                "role": "assistant",
                "timestamp": NOW + 11_000,
                "content": (
                    "Got it. Priya owns rollback, with the 2% five-minute trigger "
                    "and a 24-hour image retention window."
                ),
            },
        ],
    ),
    (
        "orion-pricing",
        [
            {
                "sender_id": "maya",
                "sender_name": "Maya",
                "role": "user",
                "timestamp": NOW + 20_000,
                "content": (
                    "Project Orion will test annual billing with the education "
                    "segment. The pricing review is scheduled for October 12."
                ),
            },
            {
                "sender_id": "assistant",
                "role": "assistant",
                "timestamp": NOW + 21_000,
                "content": (
                    "I will remember Orion's annual billing experiment and October "
                    "pricing review."
                ),
            },
        ],
    ),
    (
        "vega-mobile",
        [
            {
                "sender_id": "maya",
                "sender_name": "Maya",
                "role": "user",
                "timestamp": NOW + 30_000,
                "content": (
                    "For Project Vega, the mobile team chose offline drafts as the "
                    "next milestone. Elena will review the interaction design on "
                    "October 18."
                ),
            },
            {
                "sender_id": "assistant",
                "role": "assistant",
                "timestamp": NOW + 31_000,
                "content": (
                    "Noted. Vega's next milestone is offline drafts, followed by "
                    "Elena's design review."
                ),
            },
        ],
    ),
    (
        "nova-warehouse",
        [
            {
                "sender_id": "maya",
                "sender_name": "Maya",
                "role": "user",
                "timestamp": NOW + 40_000,
                "content": (
                    "Project Nova will migrate the analytics warehouse to Iceberg. "
                    "Marcus owns the checksum rehearsal scheduled for October 22."
                ),
            },
            {
                "sender_id": "assistant",
                "role": "assistant",
                "timestamp": NOW + 41_000,
                "content": (
                    "I will remember Nova's warehouse migration and Marcus's "
                    "checksum rehearsal."
                ),
            },
        ],
    ),
    (
        "helios-support",
        [
            {
                "sender_id": "maya",
                "sender_name": "Maya",
                "role": "user",
                "timestamp": NOW + 50_000,
                "content": (
                    "Project Helios needs weekend support coverage for the APAC "
                    "region. Imani will publish the rotation schedule on November 1."
                ),
            },
            {
                "sender_id": "assistant",
                "role": "assistant",
                "timestamp": NOW + 51_000,
                "content": (
                    "Noted. Helios needs APAC weekend coverage, and Imani owns the "
                    "rotation schedule."
                ),
            },
        ],
    ),
    (
        "luna-onboarding",
        [
            {
                "sender_id": "maya",
                "sender_name": "Maya",
                "role": "user",
                "timestamp": NOW + 60_000,
                "content": (
                    "Project Luna will replace the onboarding tour with a checklist. "
                    "The localized copy is due from the content team on October 25."
                ),
            },
            {
                "sender_id": "assistant",
                "role": "assistant",
                "timestamp": NOW + 61_000,
                "content": (
                    "I will remember Luna's checklist approach and the localization "
                    "deadline."
                ),
            },
        ],
    ),
    (
        "aurora-observability",
        [
            {
                "sender_id": "maya",
                "sender_name": "Maya",
                "role": "user",
                "timestamp": NOW + 70_000,
                "content": (
                    "Project Aurora will retain detailed telemetry for 30 days. "
                    "The operations team should alert after three consecutive "
                    "heartbeat misses."
                ),
            },
            {
                "sender_id": "assistant",
                "role": "assistant",
                "timestamp": NOW + 71_000,
                "content": (
                    "Understood. Aurora keeps 30 days of telemetry and alerts after "
                    "three missed heartbeats."
                ),
            },
        ],
    ),
    (
        "comet-invoices",
        [
            {
                "sender_id": "maya",
                "sender_name": "Maya",
                "role": "user",
                "timestamp": NOW + 80_000,
                "content": (
                    "Project Comet will add downloadable invoice PDFs for enterprise "
                    "accounts. Finance will approve the tax-field layout on October 28."
                ),
            },
            {
                "sender_id": "assistant",
                "role": "assistant",
                "timestamp": NOW + 81_000,
                "content": (
                    "Noted. Comet covers enterprise invoice PDFs and an October tax "
                    "layout review."
                ),
            },
        ],
    ),
    (
        "solstice-research",
        [
            {
                "sender_id": "maya",
                "sender_name": "Maya",
                "role": "user",
                "timestamp": NOW + 90_000,
                "content": (
                    "Project Solstice is prototyping voice notes for field researchers. "
                    "The research team will interview 12 participants in November."
                ),
            },
            {
                "sender_id": "assistant",
                "role": "assistant",
                "timestamp": NOW + 91_000,
                "content": (
                    "I will remember Solstice's voice-note prototype and the planned "
                    "participant interviews."
                ),
            },
        ],
    ),
]


def post(path, payload):
    request = Request(
        f"{API_URL}/{path}",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(request, timeout=300) as response:
        return json.load(response)["data"]


for session_id, messages in conversations:
    added = post(
        "add",
        {
            "session_id": session_id,
            "app_id": "project-assistant",
            "project_id": "launch-planning",
            "messages": messages,
            "defer_extraction": True,
        },
    )
    flushed = post(
        "flush",
        {
            "session_id": session_id,
            "app_id": "project-assistant",
            "project_id": "launch-planning",
        },
    )
    print(f"{session_id}: {added['status']} -> {flushed['status']}")
```

Run it from the project directory:

```shell
uv run python add_memories.py
```

Reference output:

```text
atlas-release: accumulated -> extracted
atlas-rollback: accumulated -> extracted
orion-pricing: accumulated -> extracted
vega-mobile: accumulated -> extracted
nova-warehouse: accumulated -> extracted
helios-support: accumulated -> extracted
luna-onboarding: accumulated -> extracted
aurora-observability: accumulated -> extracted
comet-invoices: accumulated -> extracted
solstice-research: accumulated -> extracted
```

Setting `defer_extraction` to `true` stores each conversation in the durable buffer without asking the LLM to detect a boundary. The following `/flush` call marks the end of that session and triggers one extraction. EverOS then writes the extracted episode to Markdown and asynchronously embeds it for the Milvus index.

## Inspect the Markdown memory

The generated episode file is stored under the application, project, and user scopes:

```shell
find "$EVEROS_ROOT/project-assistant/launch-planning/users/maya/episodes" \
  -type f -name "*.md"
```

Reference output (the filename date reflects when you run the example):

```text
everos-data/project-assistant/launch-planning/users/maya/episodes/episode-2026-09-08.md
```

Open the file to see the LLM-extracted memories. A shortened excerpt looks like this:

```markdown
## ep_20260908_00000001

**owner_id**: maya
**session_id**: atlas-release
**sender_ids**: [maya, assistant]

### Subject
Maya's Project Atlas Launch Decision: September 30 Canary and Promotion Criteria

### Content
Maya decided that Project Atlas would launch with a 10% canary on September 30.
The promotion to all users would occur only after the checkout error rate remained
below 1% for 30 minutes.
```

Exact wording, identifiers, and timestamps can vary because the memory is extracted by the LLM. The original Markdown files remain the durable source of truth; the Milvus index can be rebuilt from them.

## Search the memories

Use hybrid search to ask what should be remembered before Atlas goes live. Save the following code as `search_memories.py`:

```python
import json
import time
from urllib.request import Request, urlopen


URL = "http://127.0.0.1:8000/api/v2/memory/search"
payload = {
    "user_id": "maya",
    "app_id": "project-assistant",
    "project_id": "launch-planning",
    "query": "What should I remember before Atlas goes live?",
    "method": "hybrid",
    "top_k": 4,
}


def search():
    request = Request(
        URL,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(request, timeout=300) as response:
        return json.load(response)["data"]["episodes"]


expected_sessions = {"atlas-release", "atlas-rollback"}

for _ in range(30):
    episodes = search()
    top_results = episodes[:2]
    if {episode["session_id"] for episode in top_results} == expected_sessions:
        break
    time.sleep(2)
else:
    raise RuntimeError("The expected Atlas memories were not indexed in time")

for rank, episode in enumerate(top_results, start=1):
    print(f"{rank}. {episode['session_id']} | score={episode['score']:.3f}")
    print(f"   {episode['subject']}")
```

Run the search:

```shell
uv run python search_memories.py
```

Reference output (scores and wording may vary):

```text
1. atlas-release | score=0.492
   Project Atlas Launch Plan: 10% Canary Rollout on September 30 with Error Rate Gate
2. atlas-rollback | score=0.400
   Atlas Rollback Plan Details: Priya as Owner, 2% Error Trigger, 24-Hour Image Retention
```

Both Atlas conversations are returned ahead of the eight unrelated conversations. EverOS sends the query to the OpenAI embedding endpoint, asks Milvus for BM25 and vector candidates within Maya's application and project scope, and fuses the two result lists.

## Inspect the Milvus collections

EverOS creates one collection for each supported derived memory kind. Use `MilvusClient` to list their row counts:

```python
import os

from pymilvus import MilvusClient


prefix = "everos_bootcamp"
client = MilvusClient(uri=os.environ.get("MILVUS_URI", "http://localhost:19530"))

memory_kinds = [
    "agent_case",
    "agent_skill",
    "atomic_fact",
    "episode",
    "foresight",
    "knowledge_topic",
    "user_profile",
]

for kind in memory_kinds:
    name = f"{prefix}_{kind}"
    if client.has_collection(collection_name=name):
        result = client.query(
            collection_name=name,
            filter="",
            output_fields=["count(*)"],
        )
        print(f"{kind}: {result[0]['count(*)']} rows")

client.close()
```

Reference output from the validated run:

```text
agent_case: 0 rows
agent_skill: 0 rows
atomic_fact: 50 rows
episode: 10 rows
foresight: 0 rows
knowledge_topic: 0 rows
user_profile: 1 rows
```

The exact number of atomic facts may vary with the LLM output. The ten episode rows correspond to the ten flushed conversations. The other collections are available for EverOS memory modes and features that this focused example does not exercise.

## Use another Milvus deployment

To use another Milvus Server endpoint or Zilliz Cloud, update `EVEROS_MILVUS__URI`. Set `EVEROS_MILVUS__TOKEN` when the endpoint requires authentication. The ingestion and search code remains unchanged.
