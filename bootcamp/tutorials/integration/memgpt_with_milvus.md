# MemGPT with Milvus Integration

[MemGPT](https://memgpt.readme.io/docs/index) makes it easy to build and deploy stateful LLM agents. With Milvus integration, you can build agents with connections to external data sources (RAG).

In this example, we're going to use MemGPT to chat with a custom data source which is stored in Milvus. 

### Configuration

To run MemGPT, you should make sure the Python version >= 3.10.

To enable the Milvus backend, make sure to install the required dependencies with:

```shell
pip install 'pymemgpt[milvus]'
```

You can configure Milvus connection via command 

```shell
memgpt configure
```

```shell
...
? Select storage backend for archival data: milvus
? Enter the Milvus connection URI (Default: ~/.memgpt/milvus.db): ~/.memgpt/milvus.db
```
You just set the URI to the local file path, e.g. `~/.memgpt/milvus.db`, which will automatically invoke the local Milvus service instance through Milvus Lite.

If you have large scale of data such as more than a million docs, we recommend setting up a more performant Milvus server on [docker or kubenetes](https://milvus.io/docs/quickstart.md).
And in this case, your URI should be the server URI, e.g. `http://localhost:19530`.

### Creating an external data source

To feed external data into a MemGPT chatbot, we first need to create a data source.

To download the MemGPT research paper we'll use `curl` (you can also just download the PDF from your browser):

```shell
# we're saving the file as "memgpt_research_paper.pdf"
curl -L -o memgpt_research_paper.pdf https://arxiv.org/pdf/2310.08560.pdf
```

Now that we have the paper downloaded, we can create a MemGPT data source using `memgpt load`:

```shell
memgpt load directory --name memgpt_research_paper --input-files=memgpt_research_paper.pdf
```

```text
Loading files: 100%|███████████████████████████████████| 1/1 [00:00<00:00,  3.94file/s]
Loaded 74 passages and 13 documents from memgpt_research_paper
```

### Attaching the data source to a MemGPT agent

Now that we've created this data source, we can attach it to a MemGPT chatbot at any time.

Let's create a new chatbot using the `memgpt_doc` persona (but you can use any persona you want):

```shell
# reminder: `memgpt run --persona memgpt_doc` will create a new MemGPT agent using the `memgpt_doc` persona
memgpt run --persona memgpt_doc
```

Once we're chatting with the agent, we can "attach" the data source to the agent's archival memory:

```text
? Would you like to select an existing agent? No

🧬 Creating new agent...
->  🤖 Using persona profile: 'sam_pov'
->  🧑 Using human profile: 'basic'
🎉 Created new agent 'PoliteButterfly' (id=d26e1981-ff36-4095-97a0-61a1601dfb5d)

Hit enter to begin (will request first MemGPT message)

💭 Interesting, I've got a first-time user. Time to present myself and get to understand the user's needs. I wonder what brings Chad here today.
🤖 Greetings Chad! I'm MemGPT. How may I assist you today?

> Enter your message: /attach
? Select data source memgpt_research_paper
100%|███████████████████████████████████| 1/1 [00:00<00:00,  4.81it/s]
```

### Testing out our new chatbot

Now that the data has been loaded into the chatbot's memory, we can start to ask questions about it:

```text
> Enter your message: The paper I loaded into your archival memory is called MemGPT. Can you tell me more about it?

💭 It seems Chad would like to know more about the paper titled 'MemGPT' that's stored in my archival memory. It's time to search the archival memory for details and background data to provide a comprehensive
understanding of the paper. Let's remember to page through results until I've gathered enough information to answer Chad's enquiry.
⚡🧠 [function] searching memory with archival_memory_search
💭 Based on the found results, MemGPT is an AI system designed for tasks like extended conversations and document analysis. It efficiently manages different tiers of storage to provide extended context within
the LLM’s limited cap. Also, noteworthy from my findings is that MemGPT performs impressively in deep memory retrieval (DMR) and document QA tasks. Let's prepare a comprehensive response for Chad.
🤖 Certainly, Chad. MemGPT, or MemoryGPT, is a system specially designed for tasks like extended conversations and document analysis which are traditionally hindered by the limited context windows of modern 
Large Language Models (LLMs). It uses a technique inspired by hierarchical memory systems in traditional operating systems, called virtual context management. This technique allows for an effective provision 
of extended context within an LLM’s limited context window by intelligently managing different storage tiers, much like paging between physical memory and disk. Notably, MemGPT outperforms the fixed-context 
baselines in deep memory retrieval and it's able to analyze large documents with precision.

```

