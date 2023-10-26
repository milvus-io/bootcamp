import numpy as np
import torch
from torch.nn import functional as F

# Output words instead of scores.
def sentiment_score_to_name(score: float):
    if score > 0:
        return "Positive"
    elif score <= 0:
        return "Negative"

# Split data into train, valid, test. 
def partition_dataset(df_input, smoke_test=False):
    """Splits data, assuming original, input dataframe contains 50K rows.

    Args:
        df_input (pandas.DataFrame): input data frame
        smoke_test (boolean): if True, use smaller number of rows for testing
    
    Returns:
        df_train, df_val, df_test (pandas.DataFrame): train, valid, test splits.
    """

    # Shuffle data and split into train/val/test.
    df_shuffled = df_input.sample(frac=1, random_state=1).reset_index()
    # Add a corpus index.
    columns = ['movie_index', 'text', 'label_int', 'label']
    df_shuffled.columns = columns

    df_train = df_shuffled.iloc[:35_000]
    df_val = df_shuffled.iloc[35_000:40_000]
    df_test = df_shuffled.iloc[40_000:]

    # Save train/val/test split data locally in separate files.
    df_train.to_csv("train.csv", index=False, encoding="utf-8")
    df_val.to_csv("val.csv", index=False, encoding="utf-8")
    df_test.to_csv("test.csv", index=False, encoding="utf-8")

    return df_shuffled, df_train, df_val, df_test

# Take as input a user query and conduct semantic vector search using the query.
def mc_search_imdb(query, retriever, milvus_collection, search_params, top_k, 
                   milvus_client=False, COLLECTION_NAME = 'movies'):

    # Embed the query using same embedding model used to create the Milvus collection.
    query_embeddings = torch.tensor(retriever.encode(query))
    # Normalize embeddings to unit length.
    query_embeddings = F.normalize(query_embeddings, p=2, dim=1)
    # Quick check if embeddings are normalized.
    norms = np.linalg.norm(query_embeddings, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5) == True
    # Convert the embeddings to list of list of np.float32.
    query_embeddings = list(map(np.float32, query_embeddings))

    # Run semantic vector search using your query and the vector database.
    # Assemble results.
    distances = []
    texts = []
    movie_indexes = []
    labels = []
    if milvus_client:
        # MilvusClient search API call slightly different.
        results = milvus_collection.search(
            COLLECTION_NAME,
            data=query_embeddings, 
            search_params=search_params,
            output_fields=["movie_index", "chunk", "label"], 
            limit=top_k,
            consistency_level="Eventually",
            )
        # Results returned from MilvusClient are in the form list of lists of dicts.
        for result in results[0]:
            distances.append(result['distance'])
            texts.append(result['entity']['chunk'])
            movie_indexes.append(result['entity']['movie_index'])
            labels.append(result['entity']['label'])
    else:
        # Milvus server search API call.
        results = milvus_collection.search(
            data=query_embeddings, 
            anns_field="vector", 
            param=search_params,
            output_fields=["movie_index", "chunk", "label"], 
            limit=top_k,
            consistency_level="Eventually"
            )
        # Assemble results from Milvus server.
        distances = results[0].distances
        for result in results[0]:
            texts.append(result.entity.get("chunk"))
            movie_indexes.append(result.entity.get("movie_index"))
            labels.append(result.entity.get("label"))

    # Assemble all the results in a zipped list.
    formatted_results = list(zip(distances, movie_indexes, texts, labels))

    return formatted_results

