from typing import List
import time
import numpy as np
from langchain.text_splitter import RecursiveCharacterTextSplitter
import torch
from torch.nn import functional as F

# Output words instead of scores.
def sentiment_score_to_name(score: float):
    if score > 0:
        return "Positive"
    elif score <= 0:
        return "Negative"

# Split data into train, valid, test. 
def partition_dataset(df_input, new_columns, smoke_test=False):
    """Splits data, assuming original, input dataframe contains 50K rows.

    Args:
        df_input (pandas.DataFrame): input data frame
        smoke_test (boolean): if True, use smaller number of rows for testing
    
    Returns:
        df_train, df_val, df_test (pandas.DataFrame): train, valid, test splits.
    """

    # Shuffle data and split into train/val/test.
    df_shuffled = df_input.sample(frac=1, random_state=1).reset_index()
    df_shuffled.columns = new_columns

    df_train = df_shuffled.iloc[:35_000]
    df_val = df_shuffled.iloc[35_000:40_000]
    df_test = df_shuffled.iloc[40_000:]

    # Save train/val/test split data locally in separate files.
    df_train.to_csv("train.csv", index=False, encoding="utf-8")
    df_val.to_csv("val.csv", index=False, encoding="utf-8")
    df_test.to_csv("test.csv", index=False, encoding="utf-8")

    return df_shuffled, df_train, df_val, df_test

##########
# Functions for LangChain chunking and embedding.
##########'

def recursive_splitter_wrapper(text, chunk_size):

    # Default chunk overlap is 10% chunk_size.
    chunk_overlap = np.round(chunk_size * 0.10, 0)

    # Use langchain's convenient recursive chunking method.
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )
    chunks: List[str] = text_splitter.split_text(text)

    # Replace special characters with spaces.
    chunks = [text.replace("<br /><br />", " ") for text in chunks]

    return chunks

# Use recursive splitter to chunk text.
def imdb_chunk_text(encoder, batch_size, df, chunk_size):

    batch = df.head(batch_size).copy()
    print(f"chunk size: {chunk_size}")
    print(f"original shape: {batch.shape}")
    
    start_time = time.time()
    # 1. Change primary key type to string.
    batch["movie_index"] = batch["movie_index"].apply(lambda x: str(x))

    # 2. Truncate reviews to 512 characters.
    batch['chunk'] = batch['text'].apply(recursive_splitter_wrapper, chunk_size=chunk_size)
    # Explode the 'chunk' column to create new rows for each chunk.
    batch = batch.explode('chunk', ignore_index=True)
    print(f"new shape: {batch.shape}")

    # 3. Add embeddings as new column in df.
    review_embeddings = torch.tensor(encoder.encode(batch['chunk']))
    # Normalize embeddings to unit length.
    review_embeddings = F.normalize(review_embeddings, p=2, dim=1)
    # Quick check if embeddings are normalized.
    norms = np.linalg.norm(review_embeddings, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5) == True

    # 4. Convert embeddings to list of `numpy.ndarray`, each containing `numpy.float32` numbers.
    converted_values = list(map(np.float32, review_embeddings))
    batch['vector'] = converted_values

    # 5. Reorder columns for conveneince, so index first, labels at end.
    new_order = ["movie_index", "text", "chunk", "vector", "label_int", "label"]
    batch = batch[new_order]

    end_time = time.time()
    print(f"Chunking + embedding time for {batch_size} docs: {end_time - start_time} sec")


    return batch

##########
# Functions to process Milvus Search API responses.
##########

# Stuff answers into a context string and stuff metadata into a list of dicts.
def assemble_retrieved_context(retrieved_results, metadata_fields=[], num_shot_answers=3):
    
    # Assemble the context as a stuffed string.
    # Also save the context metadata to retrieve along with the answer.
    context = []
    context_metadata = []
    i = 1
    for r in retrieved_results[0]:
        if i <= num_shot_answers:
            if len(metadata_fields) > 0:
                metadata = {}
                for field in metadata_fields:
                    metadata[field] = getattr(r.entity, field, None)
                context_metadata.append(metadata)
            context.append(r.entity.text)
        i += 1

    return context, context_metadata

def mc_search_imdb(query, encoder, milvus_collection, metadata_fields=[], top_k=3, 
                   search_params={}, filter_expr=""):

    # Embed the query using same embedding model used to create the Milvus collection.
    embedded_question = torch.tensor(encoder.encode([query]))
    # Normalize embeddings to unit length.
    embedded_question = F.normalize(embedded_question, p=2, dim=1)
    # Quick check if embeddings are normalized.
    norms = np.linalg.norm(embedded_question, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5) == True
    # Convert the embeddings to list of list of np.float32.
    embedded_question = list(map(np.float32, embedded_question))

    print(f"filter_expr: {filter_expr}")

    # Run semantic vector search using your query and the vector database.
    results = milvus_collection.search(
        data=embedded_question, 
        anns_field="vector", 
        # No params for AUTOINDEX
        param=search_params,
        # Milvus can utilize metadata to enhance the search experience in boolean expressions.
        expr=filter_expr,
        output_fields=metadata_fields, 
        limit=top_k,
        consistency_level="Eventually"
        )
    
    # Assemble and print the results.
    context, context_metadata = assemble_retrieved_context(
        results, 
        metadata_fields, 
        num_shot_answers=top_k)
    return context, context_metadata

##########
# Functions to process Milvus Client API responses.
##########

def client_assemble_retrieved_context(retrieved_results):
    # Assemble results.

    # Results returned from MilvusClient are in the form list of lists of dicts.
    distances = []
    texts = []
    movie_indexes = []
    labels = []
    for result in retrieved_results[0]:
            distances.append(result['distance'])
            texts.append(result['entity']['chunk'])
            movie_indexes.append(result['entity']['movie_index'])
            labels.append(result['entity']['label'])

    # Assemble all the results in a zipped list.
    formatted_results = list(zip(distances, movie_indexes, labels, texts))
    return formatted_results

# Take as input a user query and conduct semantic vector search using the query.
def mc_client_search_imdb(query, encoder, milvus_collection, search_params, 
                          metadata_fields=[], top_k=3, COLLECTION_NAME = 'movies'):

    # Embed the query using same embedding model used to create the Milvus collection.
    query_embeddings = torch.tensor(encoder.encode(query))
    # Normalize embeddings to unit length.
    query_embeddings = F.normalize(query_embeddings, p=2, dim=1)
    # Quick check if embeddings are normalized.
    norms = np.linalg.norm(query_embeddings, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5) == True
    # Convert the embeddings to list of list of np.float32.
    query_embeddings = list(map(np.float32, query_embeddings))

    # MilvusClient search API call slightly different.
    results = milvus_collection.search(
        COLLECTION_NAME,
        data=query_embeddings, 
        search_params=search_params,
        output_fields=["movie_index", "chunk", "label"], 
        limit=top_k,
        consistency_level="Eventually",
        )

    # Assemble all the results in a zipped list.
    formatted_results = client_assemble_retrieved_context(results)
    return formatted_results