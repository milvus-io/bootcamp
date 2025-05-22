# Advanced Video Search: Leveraging Twelve Labs and Milvus for Semantic Retrieval
> TLDR: Learn how to create a semantic video search application by integrating Twelve Labs' Embed API for generating multimodal embeddings with Milvus. It covers the entire process from setting up the development environment to implementing advanced features like hybrid search and temporal video analysis, providing a comprehensive foundation for building sophisticated video content analysis and retrieval systems.

## Introduction
Welcome to this comprehensive tutorial on implementing semantic video search using [Twelve Labs Embed API](https://docs.twelvelabs.io/docs/create-embeddings) and Milvus. In this guide, we'll explore how to harness the power of [Twelve Labs' advanced multimodal embeddings](https://www.twelvelabs.io/blog/multimodal-embeddings) and [Milvus' efficient vector database](https://milvus.io/intro) to create a robust video search solution. By integrating these technologies, developers can unlock new possibilities in video content analysis, enabling applications such as content-based video retrieval, recommendation systems, and sophisticated search engines that understand the nuances of video data.

This tutorial will walk you through the entire process, from setting up your development environment to implementing a functional semantic video search application. We'll cover key concepts such as generating multimodal embeddings from videos, storing them efficiently in Milvus, and performing similarity searches to retrieve relevant content. Whether you're building a video analytics platform, a content discovery tool, or enhancing your existing applications with video search capabilities, this guide will provide you with the knowledge and practical steps to leverage the combined strengths of Twelve Labs and Milvus in your projects.


## Prerequisites
Before we begin, ensure you have the following:

A Twelve Labs API key (sign up at https://api.twelvelabs.io if you don't have one)
Python 3.7 or later installed on your system

## Setting Up the Development Environment
Create a new directory for your project and navigate to it:

```shell
mkdir video-search-tutorial
cd video-search-tutorial
```

Set up a virtual environment (optional but recommended):

```shell
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
```

Install the required Python libraries:

```shell
pip install twelvelabs pymilvus
```

Create a new Python file for your project:

```shell
touch video_search.py
```

This video_search.py file will be the main script we use for the tutorial. Next, set up your Twelve Labs API key as an environment variable for security:

```shell
export TWELVE_LABS_API_KEY='your_api_key_here'
```

## Connecting to Milvus
To establish a connection with Milvus, we'll use the MilvusClient class. This approach simplifies the connection process and allows us to work with a local file-based Milvus instance, which is perfect for our tutorial.

```python
from pymilvus import MilvusClient

# Initialize the Milvus client
milvus_client = MilvusClient("milvus_twelvelabs_demo.db")

print("Successfully connected to Milvus")
```

This code creates a new Milvus client instance that will store all data in a file named milvus_twelvelabs_demo.db. This file-based approach is ideal for development and testing purposes.


## Creating a Milvus Collection for Video Embeddings
Now that we're connected to Milvus, let's create a collection to store our video embeddings and associated metadata. We'll define the collection schema and create the collection if it doesn't already exist.

```python
# Initialize the collection name
collection_name = "twelvelabs_demo_collection"

# Check if the collection already exists and drop it if it does
if milvus_client.has_collection(collection_name=collection_name):
    milvus_client.drop_collection(collection_name=collection_name)

# Create the collection
milvus_client.create_collection(
    collection_name=collection_name,
    dimension=1024  # The dimension of the Twelve Labs embeddings
)

print(f"Collection '{collection_name}' created successfully")
```
In this code, we first check if the collection already exists and drop it if it does. This ensures we start with a clean slate. We create the collection with a dimension of 1024, which matches the output dimension of Twelve Labs' embeddings.

## Generating Embeddings with Twelve Labs Embed API
To generate embeddings for our videos using the Twelve Labs Embed API, we'll use the Twelve Labs Python SDK. This process involves creating an embedding task, waiting for its completion, and retrieving the results. Here's how to implement this:

First, ensure you have the Twelve Labs SDK installed and import the necessary modules:

```python
from twelvelabs import TwelveLabs
from twelvelabs.models.embed import EmbeddingsTask
import os

# Retrieve the API key from environment variables
TWELVE_LABS_API_KEY = os.getenv('TWELVE_LABS_API_KEY')
```

## Initialize the Twelve Labs client:
```python
twelvelabs_client = TwelveLabs(api_key=TWELVE_LABS_API_KEY)
```

Create a function to generate embeddings for a given video URL:

```python
def generate_embedding(video_url):
	"""
    Generate embeddings for a given video URL using the Twelve Labs API.

    This function creates an embedding task for the specified video URL using
    the Marengo-retrieval-2.6 engine. It monitors the task progress and waits
    for completion. Once done, it retrieves the task result and extracts the
    embeddings along with their associated metadata.

    Args:
        video_url (str): The URL of the video to generate embeddings for.

    Returns:
        tuple: A tuple containing two elements:
            1. list: A list of dictionaries, where each dictionary contains:
                - 'embedding': The embedding vector as a list of floats.
                - 'start_offset_sec': The start time of the segment in seconds.
                - 'end_offset_sec': The end time of the segment in seconds.
                - 'embedding_scope': The scope of the embedding (e.g., 'shot', 'scene').
            2. EmbeddingsTaskResult: The complete task result object from Twelve Labs API.

    Raises:
        Any exceptions raised by the Twelve Labs API during task creation,
        execution, or retrieval.
    """

    # Create an embedding task
    task = twelvelabs_client.embed.task.create(
        engine_name="Marengo-retrieval-2.6",
        video_url=video_url
    )
    print(f"Created task: id={task.id} engine_name={task.engine_name} status={task.status}")

    # Define a callback function to monitor task progress
    def on_task_update(task: EmbeddingsTask):
        print(f"  Status={task.status}")

    # Wait for the task to complete
    status = task.wait_for_done(
        sleep_interval=2,
        callback=on_task_update
    )
    print(f"Embedding done: {status}")

    # Retrieve the task result
    task_result = twelvelabs_client.embed.task.retrieve(task.id)

    # Extract and return the embeddings
    embeddings = []
    for v in task_result.video_embeddings:
        embeddings.append({
            'embedding': v.embedding.float,
            'start_offset_sec': v.start_offset_sec,
            'end_offset_sec': v.end_offset_sec,
            'embedding_scope': v.embedding_scope
        })
    
    return embeddings, task_result
```

Use the function to generate embeddings for your videos:

```python
# Example usage
video_url = "https://example.com/your-video.mp4"

# Generate embeddings for the video
embeddings, task_result = generate_embedding(video_url)

print(f"Generated {len(embeddings)} embeddings for the video")
for i, emb in enumerate(embeddings):
    print(f"Embedding {i+1}:")
    print(f"  Scope: {emb['embedding_scope']}")
    print(f"  Time range: {emb['start_offset_sec']} - {emb['end_offset_sec']} seconds")
    print(f"  Embedding vector (first 5 values): {emb['embedding'][:5]}")
    print()
```

This implementation allows you to generate embeddings for any video URL using the Twelve Labs Embed API. The generate_embedding function handles the entire process, from creating the task to retrieving the results. It returns a list of dictionaries, each containing an embedding vector along with its metadata (time range and scope).Remember to handle potential errors, such as network issues or API limits, in a production environment. You might also want to implement retries or more robust error handling depending on your specific use case.

## Inserting Embeddings into Milvus
After generating embeddings using the Twelve Labs Embed API, the next step is to insert these embeddings along with their metadata into our Milvus collection. This process allows us to store and index our video embeddings for efficient similarity search later.

Here's how to insert the embeddings into Milvus:

```python
def insert_embeddings(milvus_client, collection_name, task_result, video_url):
    """
    Insert embeddings into the Milvus collection.

    Args:
        milvus_client: The Milvus client instance.
        collection_name (str): The name of the Milvus collection to insert into.
        task_result (EmbeddingsTaskResult): The task result containing video embeddings.
        video_url (str): The URL of the video associated with the embeddings.

    Returns:
        MutationResult: The result of the insert operation.

    This function takes the video embeddings from the task result and inserts them
    into the specified Milvus collection. Each embedding is stored with additional
    metadata including its scope, start and end times, and the associated video URL.
    """
    data = []

    for i, v in enumerate(task_result.video_embeddings):
        data.append({
            "id": i,
            "vector": v.embedding.float,
            "embedding_scope": v.embedding_scope,
            "start_offset_sec": v.start_offset_sec,
            "end_offset_sec": v.end_offset_sec,
            "video_url": video_url
        })

    insert_result = milvus_client.insert(collection_name=collection_name, data=data)
    print(f"Inserted {len(data)} embeddings into Milvus")
    return insert_result

# Usage example
video_url = "https://example.com/your-video.mp4"

# Assuming this function exists from previous step
embeddings, task_result = generate_embedding(video_url)

# Insert embeddings into the Milvus collection
insert_result = insert_embeddings(milvus_client, collection_name, task_result, video_url)
print(insert_result)
```
This function prepares the data for insertion, including all relevant metadata such as the embedding vector, time range, and the source video URL. It then uses the Milvus client to insert this data into the specified collection.


## Performing Similarity Search
Once we have our embeddings stored in Milvus, we can perform similarity searches to find the most relevant video segments based on a query vector. Here's how to implement this functionality:

```python
def perform_similarity_search(milvus_client, collection_name, query_vector, limit=5):
    """
    Perform a similarity search on the Milvus collection.

    Args:
        milvus_client: The Milvus client instance.
        collection_name (str): The name of the Milvus collection to search in.
        query_vector (list): The query vector to search for similar embeddings.
        limit (int, optional): The maximum number of results to return. Defaults to 5.

    Returns:
        list: A list of search results, where each result is a dictionary containing
              the matched entity's metadata and similarity score.

    This function searches the specified Milvus collection for embeddings similar to
    the given query vector. It returns the top matching results, including metadata
    such as the embedding scope, time range, and associated video URL for each match.
    """
    search_results = milvus_client.search(
        collection_name=collection_name,
        data=[query_vector],
        limit=limit,
        output_fields=["embedding_scope", "start_offset_sec", "end_offset_sec", "video_url"]
    )

    return search_results
    
# define the query vector
# We use the embedding inserted previously as an example. In practice, you can replace it with any video embedding you want to query.
query_vector = task_result.video_embeddings[0].embedding.float

# Perform a similarity search on the Milvus collection
search_results = perform_similarity_search(milvus_client, collection_name, query_vector)

print("Search Results:")
for i, result in enumerate(search_results[0]):
    print(f"Result {i+1}:")
    print(f"  Video URL: {result['entity']['video_url']}")
    print(f"  Time Range: {result['entity']['start_offset_sec']} - {result['entity']['end_offset_sec']} seconds")
    print(f"  Similarity Score: {result['distance']}")
    print()
```

This implementation does the following:

1. Defines a function perform_similarity_search that takes a query vector and searches for similar embeddings in the Milvus collection.
2. Uses the Milvus client's search method to find the most similar vectors.
3. Specifies the output fields we want to retrieve, including metadata about the matching video segments.
4. Provides an example of how to use this function with a query video, first generating its embedding and then using it to search.
5. Prints the search results, including relevant metadata and similarity scores.

By implementing these functions, you've created a complete workflow for storing video embeddings in Milvus and performing similarity searches. This setup allows for efficient retrieval of similar video content based on the multimodal embeddings generated by Twelve Labs' Embed API.

## Optimizing Performance
Alright, let's take this app to the next level! When dealing with large-scale video collections, **performance is key**. To optimize, we should implement [batch processing for embedding generation and insertion into Milvus](https://milvus.io/docs/v2.3.x/bulk_insert.md). This way, we can handle multiple videos simultaneously, significantly reducing overall processing time. Additionally, we could leverage [Milvus' partitioning feature](https://milvus.io/docs/v2.2.x/partition_key.md) to organize our data more efficiently, perhaps by video categories or time periods. This would speed up queries by allowing us to search only relevant partitions.

Another optimization trick is to **use caching mechanisms for frequently accessed embeddings or search results**. This could dramatically improve response times for popular queries. Don't forget to [fine-tune Milvus' index parameters](https://milvus.io/docs/index-vector-fields.md?tab=floating) based on your specific dataset and query patterns - a little tweaking here can go a long way in boosting search performance.

## Advanced Features
Now, let's add some cool features to make our app stand out! We could implement **a hybrid search that combines text and video queries**. As a matter of fact, [Twelve Labs Embed API can also generate text embeddings for your text queries](https://docs.twelvelabs.io/docs/create-text-embeddings). Imagine allowing users to input both a text description and a sample video clip - we'd generate embeddings for both and perform a weighted search in Milvus. This would give us super precise results.

Another awesome addition would be **temporal search within videos**. [We could break down long videos into smaller segments, each with its own embedding](https://docs.twelvelabs.io/docs/create-video-embeddings#customize-your-embeddings). This way, users could find specific moments within videos, not just entire clips. And hey, why not throw in some basic video analytics? We could use the embeddings to cluster similar video segments, detect trends, or even identify outliers in large video collections.


## Error Handling and Logging
Let's face it, things can go wrong, and when they do, we need to be prepared. **Implementing robust error handling is crucial**. We should [wrap our API calls and database operations in try-except blocks](https://softwareengineering.stackexchange.com/questions/64180/good-use-of-try-catch-blocks), providing informative error messages to users when something fails. For network-related issues, [implementing retries with exponential backoff](https://learn.microsoft.com/en-us/dotnet/architecture/microservices/implement-resilient-applications/implement-retries-exponential-backoff) can help handle temporary glitches gracefully.

**As for logging, it's our best friend for debugging and monitoring**. We should use [Python's logging module](https://blog.sentry.io/logging-in-python-a-developers-guide/) to track important events, errors, and performance metrics throughout our application. Let's set up different log levels - DEBUG for development, INFO for general operation, and ERROR for critical issues. And don't forget to implement log rotation to manage file sizes. With proper logging in place, we'll be able to quickly identify and resolve issues, ensuring our video search app runs smoothly even as it scales up.

## Conclusion
Congratulations! You've now built a powerful semantic video search application using Twelve Labs' Embed API and Milvus. This integration allows you to process, store, and retrieve video content with unprecedented accuracy and efficiency. By leveraging multimodal embeddings, you've created a system that understands the nuances of video data, opening up exciting possibilities for content discovery, recommendation systems, and advanced video analytics.

As you continue to develop and refine your application, remember that the combination of Twelve Labs' advanced embedding generation and Milvus' scalable vector storage provides a robust foundation for tackling even more complex video understanding challenges. We encourage you to experiment with the advanced features discussed and push the boundaries of what's possible in video search and analysis.

