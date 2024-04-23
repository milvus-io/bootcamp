# script.py
######################################################
# Install Ray.
# !pip install -U "ray[data,default]"

# Install Pymilvus.
# !pip install pymilvus "pymilvus[model]"

# Install other dependencies.
# !pip install pandas numpy langchain boto3
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
import ray, os, pprint, time, uuid
from langchain.text_splitter import RecursiveCharacterTextSplitter
import pandas as pd
import numpy as np

# Get the embedding model function.
# Milvus docs: https://milvus.io/docs/embed-with-bgm-m3.md
import pymilvus, boto3
print(f"Pymilvus: {pymilvus.__version__}") # must be >= 2.4.0
from pymilvus.model.hybrid import BGEM3EmbeddingFunction
from pymilvus import ( RemoteBulkWriter, BulkFileType )
s3 = boto3.client('s3')

# Default chunk size 512 and overlap 10% chunk_size.
chunk_size = 512
chunk_overlap = np.round(chunk_size * 0.10, 0)

# Define a LangChain text splitter.
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=chunk_size,
    chunk_overlap=chunk_overlap,
    length_function=len,  # using built-in Python len function
    )

# Define a regular python function for chunking.
def chunk_row(row, splitter=text_splitter):

    # Copy all columns into metadata.
    metadata = row.copy()
    del metadata['text'] # Remove text from metadata to avoid duplication

    # Split the text into chunks.
    chunks = splitter.create_documents(
        texts=[row["text"]], 
        metadatas=[metadata])
    chunk_list = [{
        # "id": str(uuid.uuid4()),
        "text": chunk.page_content,
        **chunk.metadata} for chunk in chunks]

    return chunk_list

# Define a class to compute embeddings.
class ComputeEmbeddings:
      def __init__(self):
            # Initialize a Milvus built-in sparse-dense-late-interaction-reranking encoder.
            # https://huggingface.co/BAAI/bge-m3
            self.model = BGEM3EmbeddingFunction(use_fp16=False, device="cpu")
            print(f"dense_dim: {self.model.dim['dense']}")
            print(f"sparse_dim: {self.model.dim['sparse']}")

      def __call__(self, batch):

            # Ray data batch is a dictionary where values are array values.
            # BGEM3EmbeddingFunction input is docs as a list of strings.
            docs = list(batch['text'])

            # Encode the documents. bge-m3 dense embeddings is already normalized.
            embeddings = self.model(docs)
            batch['vector_dense'] = embeddings['dense']

            # Sparse embeddings are in scipy.sparse.csr_matrix format.
            embedding_sparse = list(embeddings["sparse"])
            embedding_list = [sparse_matrix.toarray().flatten().astype(np.float32) 
                            for sparse_matrix in embedding_sparse]
            batch['vector_sparse'] = embedding_list

            return batch

######################################################
# Main code

if __name__ == "__main__":

    # FILE_PATH = "local://./data/kaggle_imdb_small.parquet"
    FILE_PATH = "s3://zilliz/kaggle_imdb.parquet"

    # Load and transform data.
    ds = ray.data.read_parquet(FILE_PATH)

    # Chunk the input text.
    chunked_ds = ds.flat_map(chunk_row)

    # Compute embeddings with a class that calls the embeddings model.
    embeddings_ds = \
        chunked_ds.map_batches(ComputeEmbeddings, concurrency=4)
    
    # Careful with print schema, it will cause a blocking operation!
    print(embeddings_ds.schema())

    # Save the embeddings to a parquet file locally or cloud.
    embeddings_ds.write_parquet('s3://zilliz/kaggle_imdb_embeddings2')


#### OUTPUT FROM RUNNING ON THE SMALL PARQUET FILE ######
#### TOTAL JOB DURATION:  18 seconds ####################

# -------------------------------------------------------
# Job 'raysubmit_PfAxdkNLbv2rDNaW' submitted successfully
# -------------------------------------------------------

# Next steps
#   Query the logs of the job:
#     ray job logs raysubmit_PfAxdkNLbv2rDNaW
#   Query the status of the job:
#     ray job status raysubmit_PfAxdkNLbv2rDNaW
#   Request the job to be stopped:
#     ray job stop raysubmit_PfAxdkNLbv2rDNaW

# Tailing logs until the job exits (disable with --no-wait):
# 2.4.0
# 2024-04-01 19:16:03,283	INFO worker.py:1432 -- Using address 127.0.0.1:6379 set in the environment variable RAY_ADDRESS
# 2024-04-01 19:16:03,284	INFO worker.py:1567 -- Connecting to existing Ray cluster at address: 127.0.0.1:6379...
# 2024-04-01 19:16:03,287	INFO worker.py:1743 -- Connected to Ray cluster. View the dashboard at http://127.0.0.1:8265 

# Parquet Files Sample 0:   0%|          | 0/1 [00:00<?, ?it/s]
                                                             
# 2024-04-01 19:16:03,972	INFO streaming_executor.py:115 -- Starting execution of Dataset. Full log is in /tmp/ray/session_2024-04-01_19-09-11_513634_87443/logs/ray-data.log
# 2024-04-01 19:16:03,972	INFO streaming_executor.py:116 -- Execution plan of Dataset: InputDataBuffer[Input] -> TaskPoolMapOperator[ReadParquet] -> ActorPoolMapOperator[FlatMap(chunk_row)->MapBatches(ComputeEmbeddings)] -> LimitOperator[limit=1]

# (_MapWorker pid=88058) 
# Fetching 23 files:   0%|          | 0/23 [00:00<?, ?it/s]
# Fetching 23 files: 100%|██████████| 23/23 [00:00<00:00, 181332.69it/s]

# - ReadParquet->SplitBlocks(20) 1:   0%|          | 0/1 [00:00<?, ?it/s]

# - FlatMap(chunk_row)->MapBatches(ComputeEmbeddings) 2:   0%|          | 0/1 [00:Running 0:   0%|          | 0/1 [00:00<?, ?it/s]
# Running: 4/10.0 CPU, 0/0.0 GPU, 512.1MB/1.0GB object_store_memory:   0%|          | 0/1 [00:01<?, ?it/s]    | 0/1 [00:00<?, ?it/s]

# Column         Type
# ------         ----                                                             
# id             string                                                           
# text           string
# movie_id       int64
# PosterLink     string
# vector_dense   numpy.ndarray(shape=(1024,), dtype=float)
# vector_sparse  numpy.ndarray(shape=(250002,), dtype=float)
# (MapWorker(FlatMap(chunk_row)->MapBatches(ComputeEmbeddings)) pid=88058) /opt/miniconda3/envs/py311-ray/lib/python3.11/multiprocessing/resource_tracker.py:254: UserWarning: resource_tracker: There appear to be 1 leaked semaphore objects to clean up at shutdown
# (MapWorker(FlatMap(chunk_row)->MapBatches(ComputeEmbeddings)) pid=88058)   warnings.warn('resource_tracker: There appear to be %d '
# (_MapWorker pid=88060) 
# Fetching 23 files:   0%|          | 0/23 [00:00<?, ?it/s]
# Fetching 23 files: 100%|██████████| 23/23 [00:00<00:00, 306250.77it/s] [repeated 3x across cluster] (Ray deduplicates logs by default. Set RAY_DEDUP_LOGS=0 to disable log deduplication, or see https://docs.ray.io/en/master/ray-observability/ray-logging.html#log-deduplication for more options.)
# (_MapWorker pid=88059) dense_dim: 1024 [repeated 3x across cluster]
# (_MapWorker pid=88059) sparse_dim: 250002 [repeated 3x across cluster]
# (MapWorker(FlatMap(chunk_row)->MapBatches(ComputeEmbeddings)) pid=88060) /opt/miniconda3/envs/py311-ray/lib/python3.11/multiprocessing/resource_tracker.py:254: UserWarning: resource_tracker: There appear to be 1 leaked semaphore objects to clean up at shutdown [repeated 3x across cluster]
# (MapWorker(FlatMap(chunk_row)->MapBatches(ComputeEmbeddings)) pid=88060)   warnings.warn('resource_tracker: There appear to be %d ' [repeated 3x across cluster]

# ------------------------------------------
# Job 'raysubmit_PfAxdkNLbv2rDNaW' succeeded
# ------------------------------------------
