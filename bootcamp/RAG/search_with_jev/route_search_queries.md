<a href="https://colab.research.google.com/github/milvus-io/bootcamp/blob/master/bootcamp/RAG/search_with_jev/route_search_queries.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> · [View on GitHub](https://github.com/milvus-io/bootcamp/blob/master/bootcamp/RAG/search_with_jev/route_search_queries.ipynb)

# Route search queries with Jev

Choose a known search scope, then let Milvus apply an exact metadata filter. This example does not grant access or generate arbitrary filter expressions.

## Preparation

Run locally from this directory with `uv sync --python 3.12` and `uv run jupyter lab`, or install the notebook dependencies in Colab:

```python
# In Colab, uncomment this setup cell. Local users should use uv sync --python 3.12.
# %pip install "pymilvus>=2.5,<2.6.10" "milvus-lite>=2.5,<3" "setuptools<71" scikit-learn requests
```

> In Colab, restart the runtime after installing dependencies if needed.

Set `TYPESAFE_API_KEY` in your environment or enter it privately below. Only the small synthetic examples are sent to TypeSafe. Requests consume API credits.

The helper batches independent questions into one request. IDs map responses back to code; the instructions explicitly identify each field being judged. HTTP failures stop the tutorial rather than produce fabricated scores.

```python
import getpass
import json
import math
import os
import time
import uuid

import requests
from pymilvus import DataType, MilvusClient
from sklearn.feature_extraction.text import TfidfVectorizer

if not os.getenv("TYPESAFE_API_KEY"):
    os.environ["TYPESAFE_API_KEY"] = getpass.getpass("TypeSafe API key: ")

MODEL = os.getenv("JEV_MODEL", "jev-1.13.0")
API_URL = "https://api.typesafe.ai/v1/systemone"
call_log = []


def judge(state, questions):
    """Call Jev with bounded retries; stop on invalid or incomplete responses."""
    for attempt in range(3):
        started = time.perf_counter()
        response = requests.post(
            API_URL,
            headers={"Authorization": f"Bearer {os.environ['TYPESAFE_API_KEY']}"},
            json={"model": MODEL, "state": state, "questions": questions},
            timeout=45,
        )
        if response.status_code in (429, 500, 502, 503, 504) and attempt < 2:
            time.sleep(2**attempt)
            continue
        response.raise_for_status()
        body = response.json()
        answers = body["answers"]
        if set(answers) != set(questions):
            raise ValueError("Jev returned missing or unexpected question IDs")
        for key, question in questions.items():
            answer = answers[key]
            if question["type"] == "noul":
                value = float(answer["noul"])
                if not math.isfinite(value) or not 0 <= value <= 1:
                    raise ValueError("Invalid Noul probability")
            elif answer["choice"] not in question["criteria"]:
                raise ValueError("Unexpected Choice option")
        call_log.append(
            {
                "seconds": round(time.perf_counter() - started, 3),
                "usage": body.get("usage", {}),
                "model": MODEL,
            }
        )
        return answers
    raise RuntimeError("Jev request failed")


def noul(instructions):
    return {
        "type": "noul",
        "instructions": instructions,
        "criteria": {
            "true": "The stated condition is supported by the supplied data.",
            "false": "The condition is unsupported or contradicted.",
        },
    }
```

## Prepare a small corpus

All names and records below are synthetic teaching examples.

```python
documents = [
    {
        "id": 1,
        "text": "Atlas installation quickstart: install Docker, download the compose file, then run docker compose up. Includes a complete beginner walkthrough.",
        "category": "docs",
        "version": "v2",
    },
    {
        "id": 2,
        "text": "Atlas production installation: configure TLS, backups, health checks and recovery procedures. Intended for experienced operators.",
        "category": "docs",
        "version": "v2",
    },
    {
        "id": 3,
        "text": "Atlas installation announcement: our new release is faster. This announcement contains no installation commands.",
        "category": "news",
        "version": "v2",
    },
    {
        "id": 4,
        "text": "Atlas billing: invoices are available from the billing settings page.",
        "category": "billing",
        "version": "v2",
    },
    {
        "id": 5,
        "text": "Atlas installation v1: use the legacy setup script. This procedure is obsolete for v2.",
        "category": "docs",
        "version": "v1",
    },
    {
        "id": 6,
        "text": "Database integration tests failed because DATABASE_URL used localhost inside a container. Fix: use the compose service hostname db.",
        "category": "memory",
        "version": "v2",
    },
    {
        "id": 7,
        "text": "Database integration tests are run with pytest tests/integration. This note records the command, not a connection failure fix.",
        "category": "memory",
        "version": "v2",
    },
]
```

## Store and retrieve with Milvus

For a small dependency footprint, this example uses TF-IDF lexical vectors. This is not a semantic embedding benchmark; substitute your embedding model when using a real corpus.

> A local file URI uses Milvus Lite. Set `MILVUS_URI` to a server endpoint for Docker/Kubernetes, or set both `MILVUS_URI` and `MILVUS_TOKEN` for Zilliz Cloud. Each run creates a uniquely named collection and cleans up only that collection.

```python
client = MilvusClient(
    uri=os.getenv("MILVUS_URI", "./search_with_jev.db"),
    token=os.getenv("MILVUS_TOKEN", ""),
)
collection_name = "jev_demo_" + uuid.uuid4().hex[:12]

# Tiny lexical vectors keep this tutorial CPU-only and avoid a second API key.
# Replace this encoder with your production embedding model for semantic retrieval.
encoder = TfidfVectorizer()
vectors = encoder.fit_transform([row["text"] for row in documents]).toarray()

schema = client.create_schema(auto_id=False, enable_dynamic_field=True)
schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True)
schema.add_field(
    field_name="vector", datatype=DataType.FLOAT_VECTOR, dim=vectors.shape[1]
)
schema.add_field(field_name="text", datatype=DataType.VARCHAR, max_length=8192)
index_params = client.prepare_index_params()
index_params.add_index(
    field_name="vector", index_type="AUTOINDEX", metric_type="COSINE"
)
if not client.has_collection(collection_name):
    client.create_collection(
        collection_name=collection_name,
        schema=schema,
        index_params=index_params,
        # consistency_level="Strong",
    )
client.insert(
    collection_name,
    [dict(row, vector=vector.tolist()) for row, vector in zip(documents, vectors)],
)


def retrieve(query, limit=5, filter_expr=""):
    vector = encoder.transform([query]).toarray()[0]
    if not vector.any():
        return []
    hits = client.search(
        collection_name=collection_name,
        data=[vector.tolist()],
        anns_field="vector",
        limit=limit,
        filter=filter_expr,
        output_fields=["*"],
        search_params={"metric_type": "COSINE"},
        consistency_level="Strong",
    )[0]
    return [
        dict(
            {key: value for key, value in hit["entity"].items() if key != "vector"},
            id=hit["id"],
            retrieval_score=hit["distance"],
        )
        for hit in hits
    ]
```

## Choose a bounded route

The application maps fixed choices to filters. A fallback handles questions outside the catalog.

```python
query = "Where can I download my Atlas invoice?"
answer = judge(
    {"query": query},
    {
        "route": {
            "type": "choice",
            "instructions": "Which catalog best matches the information needed by `query`?",
            "criteria": {
                "docs": "Installation and operation documentation",
                "billing": "Invoices and billing help",
                "memory": "Previous coding-session fixes",
                "other": "None of these catalogs",
            },
        }
    },
)
route = answer["route"]["choice"]
filters = {
    "docs": 'category == "docs"',
    "billing": 'category == "billing"',
    "memory": 'category == "memory"',
}
results = retrieve(query, filter_expr=filters[route]) if route in filters else []
print({"route": route, "results": [row["text"] for row in results]})
```

## Inspect usage and clean up

The raw usage fields and request duration help inspect this run. They are not a latency benchmark.

```python
print(json.dumps(call_log, indent=2))
client.drop_collection(collection_name)
client.close()
```

The thresholds in this example are starting points, not calibrated production defaults. Independent questions share state but do not see each other's answers. See the [Jev primitives](https://docs.typesafe.ai/primitives) and the [cookbook index](https://github.com/milvus-io/bootcamp/blob/master/bootcamp/RAG/search_with_jev/README.md).
