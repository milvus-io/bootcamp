
# Getting Started with Milvus and n8n  


## Introduction to n8n and the Milvus Vector Store Node


[n8n](https://n8n.io/) is a powerful open-source workflow automation platform that allows you to connect various applications, services, and APIs together to create automated workflows without coding. With its node-based visual interface, n8n enables users to build complex automation processes by simply connecting nodes that represent different services or actions. It is self-hostable, highly extensible, and supports both fair-code and enterprise licensing.

The **Milvus Vector Store** node in n8n integrates [Milvus](https://milvus.io/) into your automation workflows. This allows you to perform semantic search, power retrieval-augmented generation (RAG) systems, and build intelligent AI applications—all within the n8n ecosystem.

> This documentation is primarily based on the official [n8n Milvus Vector Store documentation](https://docs.n8n.io/integrations/builtin/cluster-nodes/root-nodes/n8n-nodes-langchain.vectorstoremilvus/). If you find any outdated or inconsistent content, please prioritize the official documentation and feel free to raise an issue for us.



## Key Features

With the Milvus Vector Store node in n8n, you can:

- Interact with your Milvus database as a [vector store](https://docs.n8n.io/glossary/#ai-vector-store)
- Insert documents into Milvus
- Get documents from Milvus
- Retrieve documents to provide them to a retriever connected to a [chain](https://docs.n8n.io/glossary/#ai-chain)
- Connect directly to an [agent](https://docs.n8n.io/glossary/#ai-agent) as a [tool](https://docs.n8n.io/glossary/#ai-tool)
- Filter documents based on metadata


## Node Usage Patterns

You can use the Milvus Vector Store node in n8n in the following patterns.

### Use as a regular node to insert and retrieve documents

You can use the Milvus Vector Store as a regular node to insert, or get documents. This pattern places the Milvus Vector Store in the regular connection flow without using an agent.

See this [example template](https://n8n.io/workflows/3573-create-a-rag-system-with-paul-essays-milvus-and-openai-for-cited-answers/) for how to build a system that stores documents in Milvus and retrieves them to support cited, chat-based answers.


### Connect directly to an AI agent as a tool

You can connect the Milvus Vector Store node directly to the tool connector of an [AI agent](https://docs.n8n.io/integrations/builtin/cluster-nodes/root-nodes/n8n-nodes-langchain.agent/) to use a vector store as a resource when answering queries.

Here, the connection would be: AI agent (tools connector) -> Milvus Vector Store node. See this [example template](https://n8n.io/workflows/3576-paul-graham-essay-search-and-chat-with-milvus-vector-database/) where data is embedded and indexed in Milvus, and the AI Agent uses the vector store as a knowledge tool for question-answering.


### Use a retriever to fetch documents

You can use the [Vector Store Retriever](https://docs.n8n.io/integrations/builtin/cluster-nodes/sub-nodes/n8n-nodes-langchain.retrievervectorstore/) node with the Milvus Vector Store node to fetch documents from the Milvus Vector Store node. This is often used with the [Question and Answer Chain](https://docs.n8n.io/integrations/builtin/cluster-nodes/root-nodes/n8n-nodes-langchain.chainretrievalqa/) node to fetch documents from the vector store that match the given chat input.

A typical node connection flow looks like this: Question and Answer Chain (Retriever connector) -> Vector Store Retriever (Vector Store connector) -> Milvus Vector Store.

Check out this [workflow example](https://n8n.io/workflows/3574-create-a-paul-graham-essay-qanda-system-with-openai-and-milvus-vector-database/) to see how to ingest external data into Milvus and build a chat-based semantic Q&A system.


### Use the Vector Store Question Answer Tool to answer questions

Another pattern uses the [Vector Store Question Answer Tool](https://docs.n8n.io/integrations/builtin/cluster-nodes/sub-nodes/n8n-nodes-langchain.toolvectorstore/) to summarize results and answer questions from the Milvus Vector Store node. Rather than connecting the Milvus Vector Store directly as a tool, this pattern uses a tool specifically designed to summarizes data in the vector store.

The connections flow would look like this: AI agent (tools connector) -> Vector Store Question Answer Tool (Vector Store connector) -> Milvus Vector store.



## Node Operation Modes


The Milvus Vector Store node supports multiple operation modes, each tailored for different workflow use cases. Understanding these modes helps design more effective workflows.

We will provide a high-level overview of the available operation modes and options below. For a complete list of input parameters and configuration options for each mode, please refer to the [official documentation](https://docs.n8n.io/integrations/builtin/cluster-nodes/root-nodes/n8n-nodes-langchain.vectorstoremilvus/).

---

### Operation Modes Overview

The Milvus Vector Store node supports four distinct modes:

* **Get Many**: Retrieve multiple documents based on semantic similarity to a prompt.
* **Insert Documents**: Insert new documents into your Milvus collection.
* **Retrieve Documents (As Vector Store for Chain/Tool)**: Use the node as a retriever within a chain-based system.
* **Retrieve Documents (As Tool for AI Agent)**: Use the node as a tool resource for an AI agent during question-answering tasks.

### Additional Node Options

* **Metadata Filter** (Get Many mode only): Filter results based on custom metadata keys. Multiple fields apply an AND condition.
* **Clear Collection** (Insert Documents mode only): Remove existing documents from the collection prior to inserting new ones.


### Related Resources

* [n8n Milvus Integration Documentation](https://docs.n8n.io/integrations/builtin/cluster-nodes/root-nodes/n8n-nodes-langchain.vectorstoremilvus/)
* [LangChain Milvus Documentation](https://js.langchain.com/docs/integrations/vectorstores/milvus/)
* [n8n Advanced AI Documentation](https://docs.n8n.io/advanced-ai/)
