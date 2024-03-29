# script.py
######################################################
# Install Ray.
# !pip install -U "ray[data,train,tune,serve,default]"
#
# Start Ray headnode local cluster.
# This will print out the Ray cluster address, which can be passed to start the worker nodes.
# ray start --head --port=6379
#
# For multi-node cluster, start Ray worker nodes.
# ray start --address='http://127.0.0.1:8265'
#
# Submit a Ray job using local .py script.
# See https://docs.ray.io/en/latest/cluster/running-applications/job-submission/index.html 
# export RAY_ADDRESS="http://127.0.0.1:8265"
# ray job submit --working-dir . -- python ray_data_demo.py
#
#   To terminate the Ray runtime, run
#     ray stop
#   To view the status of the cluster, use
#     ray status
#   To monitor and debug Ray, view the dashboard at 
#     127.0.0.1:8265
#   To view the Ray cluster dashboard, visit
#     http://127.0.0.1:8265/#/cluster
######################################################
import ray, pprint, time
from ray.data import read_parquet
from langchain.text_splitter import RecursiveCharacterTextSplitter
import pandas as pd
import numpy as np
from typing import List

# Get the embedding model function.
import pymilvus
print(pymilvus.__version__) # must be >= 2.4.0
from pymilvus.model.hybrid import BGEM3EmbeddingFunction
# Initialize a built-in Milvus sparse-dense-late-interaction-reranking encoder.
# https://huggingface.co/BAAI/bge-m3
encoder = BGEM3EmbeddingFunction(use_fp16=False, device="cpu")
dense_dim = encoder.dim["dense"]
print(f"dense_dim: {dense_dim}")

# Define regular python functions.
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
def imdb_chunk_text(df: pd.DataFrame) -> pd.DataFrame:
    BATCH_SIZE = 100
    CHUNK_SIZE = 512

    batch = df.head(BATCH_SIZE).copy()
    print(f"chunk size: {CHUNK_SIZE}")
    print(f"original shape: {batch.shape}")
    
    start_time = time.time()

    # 1. Chunk the text review into chunk_size.
    batch['chunk'] = batch['text'].apply(recursive_splitter_wrapper, chunk_size=CHUNK_SIZE)
    # Explode the 'chunk' column to create new rows for each chunk.
    batch = batch.explode('chunk', ignore_index=True)
    batch['chunk'] = batch['chunk'].fillna('')
    print(f"new shape: {batch.shape}")

    # 2. Add embeddings as new column in df.
    docs = batch['chunk'].to_list()
    # Ensure docs is a list of strings
    assert isinstance(docs, List)
    assert all(isinstance(doc, str) for doc in docs)
    # Encode the documents. bge-m3 dense embeddings will be already normalized.
    embeddings = encoder(docs)
    batch['vector'] = embeddings['dense']

    # 4. Drop the original 'text' column, keep the new 'chunk' column.
    batch.drop(columns=['text'], axis=1, inplace=True)

    end_time = time.time()
    print(f"Chunking + embedding time for {BATCH_SIZE} docs: {end_time - start_time} sec")
    assert len(batch.vector[0]) == dense_dim
    print(f"type embeddings: {type(batch.vector)} of {type(batch.vector[0])}")
    print(f"of numbers: {type(batch.vector[0][0])}")

    return batch

# # Convert regular python function to ray remote function.
# read_data_remote = ray.remote(read_data).options(num_returns=1)

######################################################
# Main code

FILE_PATH = "local://./data/kaggle_imdb_small.parquet"

# Load and transform data.
ds = (
    # Read data.  
    # This works.
    ray.data.read_parquet(FILE_PATH)

    # Embed text data using .map_batches() pattern.
    # This does not work.
    .map_batches(imdb_chunk_text, batch_format="pandas")
)

print(ds.schema())
pprint.pprint(ds.take(1))


# Error logs
# (MapBatches(imdb_chunk_text) pid=1253) original shape: (5, 14)


# - ReadParquet->SplitBlocks(20): 1 active, 0 queued, [cpu: 1.0, objects: 116.4KB]: 100%|██████████| 17/17 [00:18<00:00, 15.77it/s]

# - MapBatches(imdb_chunk_text): 2 active, 16 queued, [cpu: 2.0, objects: 512.0MB]Running: 2/10.0 CPU, 0/0.0 GPU, 512.1MB/1.0GB object_store_memory:   0%|          | 0/1 [00:18<?, ?it/s]
# - limit=1 3:   0%|          | 0/1 [00:18<?, ?it/s]
                                                                                                                                 


                                                                                                                       

                                                                                                                                  
# (MapBatches(imdb_chunk_text) pid=1253) new shape: (15, 15)


# - ReadParquet->SplitBlocks(20): 1 active, 0 queued, [cpu: 1.0, objects: 116.4KB]: 100%|██████████| 17/17 [00:18<00:00, 15.77it/s]

# - MapBatches(imdb_chunk_text): 2 active, 16 queued, [cpu: 2.0, objects: 512.0MB]Running: 2/10.0 CPU, 0/0.0 GPU, 512.1MB/1.0GB object_store_memory:   0%|          | 0/1 [00:18<?, ?it/s]
# - limit=1 3:   0%|          | 0/1 [00:18<?, ?it/s]
                                                                                                                                 


                                                                                                                       

                                                                                                                                  
# (MapBatches(imdb_chunk_text) pid=1236) Chunking + embedding time for 100 docs: 2.1832687854766846 sec


# - ReadParquet->SplitBlocks(20): 1 active, 0 queued, [cpu: 1.0, objects: 116.4KB]: 100%|██████████| 17/17 [00:20<00:00, 15.77it/s]

# - MapBatches(imdb_chunk_text): 2 active, 16 queued, [cpu: 2.0, objects: 512.0MB]Running: 2/10.0 CPU, 0/0.0 GPU, 512.1MB/1.0GB object_store_memory:   0%|          | 0/1 [00:20<?, ?it/s]
# - limit=1 3:   0%|          | 0/1 [00:20<?, ?it/s]
                                                                                                                                 


                                                                                                                       

                                                                                                                                  
# (MapBatches(imdb_chunk_text) pid=1236) type embeddings: <class 'pandas.core.series.Series'> of <class 'numpy.ndarray'>


# - ReadParquet->SplitBlocks(20): 1 active, 0 queued, [cpu: 1.0, objects: 116.4KB]: 100%|██████████| 17/17 [00:20<00:00, 15.77it/s]

# - MapBatches(imdb_chunk_text): 2 active, 16 queued, [cpu: 2.0, objects: 512.0MB]Running: 2/10.0 CPU, 0/0.0 GPU, 512.1MB/1.0GB object_store_memory:   0%|          | 0/1 [00:20<?, ?it/s]
# - limit=1 3:   0%|          | 0/1 [00:20<?, ?it/s]
                                                                                                                                 


                                                                                                                       

                                                                                                                                  
# (MapBatches(imdb_chunk_text) pid=1236) of numbers: <class 'numpy.float32'>


# - ReadParquet->SplitBlocks(20): 1 active, 0 queued, [cpu: 1.0, objects: 116.4KB]: 100%|██████████| 17/17 [00:20<00:00, 15.77it/s]

# - MapBatches(imdb_chunk_text): 2 active, 16 queued, [cpu: 2.0, objects: 512.0MB]Running: 2/10.0 CPU, 0/0.0 GPU, 512.1MB/1.0GB object_store_memory:   0%|          | 0/1 [00:20<?, ?it/s]
# - limit=1 3:   0%|          | 0/1 [00:20<?, ?it/s]

# - MapBatches(imdb_chunk_text): 3 active, 17 queued, [cpu: 3.0, objects: 768.0MB]:   0%|          | 0/1 [00:20<?, ?it/s]
# - MapBatches(imdb_chunk_text): 3 active, 17 queued, [cpu: 3.0, objects: 768.0MB]:   0%|          | 0/20 [00:20<?, ?it/s]
# Running: 2/10.0 CPU, 0/0.0 GPU, 234.3KB/1.0GB object_store_memory:   0%|          | 0/1 [00:20<?, ?it/s][00:20<06:36, 20.89s/it]

# - ReadParquet->SplitBlocks(20): 0 active, 0 queued, [cpu: 0.0, objects: 124.1KB]: 100%|██████████| 17/17 [00:23<00:00, 15.77it/s]
# - ReadParquet->SplitBlocks(20): 0 active, 0 queued, [cpu: 0.0, objects: 124.1KB]:  85%|████████▌ | 17/20 [00:23<00:00, 15.77it/s]
# - ReadParquet->SplitBlocks(20): 0 active, 0 queued, [cpu: 0.0, objects: 124.1KB]: 100%|██████████| 20/20 [00:23<00:00,  1.56s/it]

# - MapBatches(imdb_chunk_text): 10 active, 9 queued, [cpu: 10.0, objects: 367.3KB]:   5%|▌         | 1/20 [00:23<06:36, 20.89s/it]

                                                                                                                                 


                                                                                                                                 


                                                                                                        
# (MapBatches(imdb_chunk_text) pid=1252) chunk size: 512 [repeated 2x across cluster] (Ray deduplicates logs by default. Set RAY_DEDUP_LOGS=0 to disable log deduplication, or see https://docs.ray.io/en/master/ray-observability/ray-logging.html#log-deduplication for more options.)


# - ReadParquet->SplitBlocks(20): 0 active, 0 queued, [cpu: 0.0, objects: 124.1KB]: 100%|██████████| 20/20 [00:23<00:00,  1.56s/it]

# - MapBatches(imdb_chunk_text): 10 active, 9 queued, [cpu: 10.0, objects: 367.3KB]:   5%|▌         | 1/20 [00:23<06:36, 20.89s/it]
# Running: 2/10.0 CPU, 0/0.0 GPU, 234.3KB/1.0GB object_store_memory:   0%|          | 0/1 [00:23<?, ?it/s]ueued, [cpu: 0.0, objects: 5.2KB]:   0%|          | 0/1 [00:23<?, ?it/s]
                                                                                                                                 


                                                                                                                                 


                                                                                                        
# (MapBatches(imdb_chunk_text) pid=1252) original shape: (5, 14) [repeated 2x across cluster]


# - ReadParquet->SplitBlocks(20): 0 active, 0 queued, [cpu: 0.0, objects: 124.1KB]: 100%|██████████| 20/20 [00:23<00:00,  1.56s/it]

# - MapBatches(imdb_chunk_text): 10 active, 9 queued, [cpu: 10.0, objects: 367.3KB]:   5%|▌         | 1/20 [00:23<06:36, 20.89s/it]
# Running: 2/10.0 CPU, 0/0.0 GPU, 234.3KB/1.0GB object_store_memory:   0%|          | 0/1 [00:23<?, ?it/s]ueued, [cpu: 0.0, objects: 5.2KB]:   0%|          | 0/1 [00:23<?, ?it/s]
                                                                                                                                 


                                                                                                                                 


                                                                                                        
# (MapBatches(imdb_chunk_text) pid=1252) new shape: (26, 15) [repeated 2x across cluster]


# - ReadParquet->SplitBlocks(20): 0 active, 0 queued, [cpu: 0.0, objects: 124.1KB]: 100%|██████████| 20/20 [00:23<00:00,  1.56s/it]

# - MapBatches(imdb_chunk_text): 10 active, 9 queued, [cpu: 10.0, objects: 367.3KB]:   5%|▌         | 1/20 [00:23<06:36, 20.89s/it]
# Running: 2/10.0 CPU, 0/0.0 GPU, 234.3KB/1.0GB object_store_memory:   0%|          | 0/1 [00:23<?, ?it/s]2024-03-29 12:32:34,358tERROR streaming_executor_state.py:446 -- An exception was raised from a task of operator "MapBatches(imdb_chunk_text)". Dataset execution will now abort. To ignore this exception and continue, set DataContext.max_errored_blocks.

# Running: 2/10.0 CPU, 0/0.0 GPU, 234.3KB/1.0GB object_store_memory: : 0it [00:23, ?it/s]                 
                                                                                       

                                                                                                                                 

#                                                                                                 2024-03-29 12:32:34,365 ERROR exceptions.py:69 -- Exception occurred in Ray Data or Ray Core internal code. If you continue to see this error, please open an issue on the Ray project GitHub page with the full stack trace below: https://github.com/ray-project/ray/issues/new/choose
# ray.data.exceptions.SystemException

# The above exception was the direct cause of the following exception:

# Traceback (most recent call last):
#   File "/private/tmp/ray/session_2024-03-29_12-31-52_688933_1207/runtime_resources/working_dir_files/_ray_pkg_efaac20cf7eaaaec/ray_data_demo.py", line 118, in <module>
#     print(ds.schema())
#           ^^^^^^^^^^^
#   File "/opt/miniconda3/envs/py311-ray/lib/python3.11/site-packages/ray/data/dataset.py", line 2538, in schema
#     base_schema = self.limit(1)._plan.schema(fetch_if_missing=fetch_if_missing)
#                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#   File "/opt/miniconda3/envs/py311-ray/lib/python3.11/site-packages/ray/data/_internal/plan.py", line 387, in schema
#     self.execute()
#   File "/opt/miniconda3/envs/py311-ray/lib/python3.11/site-packages/ray/data/exceptions.py", line 83, in handle_trace
#     raise e.with_traceback(None) from SystemException()
# ray.exceptions.RayTaskError(ValueError): ray::MapBatches(imdb_chunk_text)() (pid=1252, ip=127.0.0.1)
#                           ^^^^^^^^^^^^^^^^
#   File "/opt/miniconda3/envs/py311-ray/lib/python3.11/site-packages/ray/air/util/tensor_extensions/pandas.py", line 767, in __init__
#     raise TypeError(
# TypeError: Expected a well-typed ndarray or an object-typed ndarray of ndarray pointers, but got an object-typed ndarray whose subndarrays are of type <class 'numpy.ndarray'>.

# The above exception was the direct cause of the following exception:

# ray::MapBatches(imdb_chunk_text)() (pid=1252, ip=127.0.0.1)
#   File "/opt/miniconda3/envs/py311-ray/lib/python3.11/site-packages/ray/data/_internal/execution/operators/map_operator.py", line 419, in _map_task
#     for b_out in map_transformer.apply_transform(iter(blocks), ctx):
#   File "/opt/miniconda3/envs/py311-ray/lib/python3.11/site-packages/ray/data/_internal/execution/operators/map_transformer.py", line 398, in __call__
#     yield output_buffer.next()
#           ^^^^^^^^^^^^^^^^^^^^
#   File "/opt/miniconda3/envs/py311-ray/lib/python3.11/site-packages/ray/data/_internal/output_buffer.py", line 73, in next
#     block_to_yield = self._buffer.build()
#                      ^^^^^^^^^^^^^^^^^^^^
#   File "/opt/miniconda3/envs/py311-ray/lib/python3.11/site-packages/ray/data/_internal/delegating_block_builder.py", line 64, in build
#     return self._builder.build()
#            ^^^^^^^^^^^^^^^^^^^^^
#   File "/opt/miniconda3/envs/py311-ray/lib/python3.11/site-packages/ray/data/_internal/table_block.py", line 133, in build
#     return self._concat_tables(tables)
#            ^^^^^^^^^^^^^^^^^^^^^^^^^^^
#   File "/opt/miniconda3/envs/py311-ray/lib/python3.11/site-packages/ray/data/_internal/pandas_block.py", line 134, in _concat_tables
#     df = _cast_ndarray_columns_to_tensor_extension(df)
#          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#   File "/opt/miniconda3/envs/py311-ray/lib/python3.11/site-packages/ray/air/util/data_batch_conversion.py", line 324, in _cast_ndarray_columns_to_tensor_extension
#     raise ValueError(
# ValueError: Tried to cast column Actors to the TensorArray tensor extension type but the conversion failed. To disable automatic casting to this tensor extension, set ctx = DataContext.get_current(); ctx.enable_tensor_extension_casting = False.
# (MapBatches(imdb_chunk_text) pid=1252) Chunking + embedding time for 100 docs: 4.913516998291016 sec [repeated 2x across cluster]
# (MapBatches(imdb_chunk_text) pid=1252) type embeddings: <class 'pandas.core.series.Series'> of <class 'numpy.ndarray'> [repeated 2x across cluster]
# (MapBatches(imdb_chunk_text) pid=1252) of numbers: <class 'numpy.float32'> [repeated 2x across cluster]

# ---------------------------------------
# Job 'raysubmit_i3jsEm8Tpckz6wsk' failed
# ---------------------------------------

# Status message: Job entrypoint command failed with exit code 1, last available logs (truncated to 20,000 chars):
#   File "/opt/miniconda3/envs/py311-ray/lib/python3.11/site-packages/ray/data/_internal/pandas_block.py", line 134, in _concat_tables
#     df = _cast_ndarray_columns_to_tensor_extension(df)
#          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#   File "/opt/miniconda3/envs/py311-ray/lib/python3.11/site-packages/ray/air/util/data_batch_conversion.py", line 324, in _cast_ndarray_columns_to_tensor_extension
#     raise ValueError(
# ValueError: Tried to cast column Actors to the TensorArray tensor extension type but the conversion failed. To disable automatic casting to this tensor extension, set ctx = DataContext.get_current(); ctx.enable_tensor_extension_casting = False.
# (MapBatches(imdb_chunk_text) pid=1252) Chunking + embedding time for 100 docs: 4.913516998291016 sec [repeated 2x across cluster]
# (MapBatches(imdb_chunk_text) pid=1252) type embeddings: <class 'pandas.core.series.Series'> of <class 'numpy.ndarray'> [repeated 2x across cluster]
# (MapBatches(imdb_chunk_text) pid=1252) of numbers: <class 'numpy.float32'> [repeated 2x across cluster]
