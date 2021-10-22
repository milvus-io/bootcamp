# Milvus performance with HNSW
This topic displays the performance of create_index & search with different index types in Milvus 1.1 for reference purposes. The test data uses 100m data from an open source dataset, sift1B, with 1 billion 128-dimensional vectors.

The following table displays the terms used in this topic:

| Term   | Description                                    |
| ---------- | ---------------------------------------- |
| nq         | Number of vectors to query. The value is defined during search.    |
| topk       | The most similar topk search results.  |
| total_time | Total time for a batch search.                 |
| avg_time   | Average search time per vector.       |
| M     | maximum degree of nodes on each layer of the graph.  |

Refer to [https://zilliz.com/blog/Accelerating-Similarity-Search-on-Really-Big-Data-with-Vector-Indexing](https://zilliz.com/blog/Accelerating-Similarity-Search-on-Really-Big-Data-with-Vector-Indexing) to learn how to choose indexes.

Refer to [scripts](https://github.com/milvus-io/bootcamp/tree/1.1/benchmark_test/scripts) for benchmark test scripts.

## Hardware Configuration:
| component  | config                             |
| ---------- | ---------------------------------- |
| OS         | Ubuntu LTS 18.04                   |
| CPU        | Intel(R) Xeon(R) Platinum 8163 CPU @ 2.50GHz.  48CPUs |
| GPU0       | NVIDIA GeForce RTX 2080 Ti, 12GB |
| GPU1       | NVIDIA GeForce RTX 2080 Ti, 12GB |
| GPU driver | CUDA 11.4 Driver 470.74          |
| Memory     | 755G        |

## Insert

During this test, the data was imported into Milvus in batches of 100,000, and it took about 1.2S to import 100,000 vectors of 128 dimensions.

## Create Index

HNSW is a graph-based index that can only use CPU for now. The index file is larger than original raw data (62GB vs 49 GB). The memory usage is high and the indexing speed is very slow.

Parameters：`index_file_size` = 2048

| Index Param (M/efConstruction)  | CPU    |
|---------------------------------|--------|
| 16/500                          | 4966s  |
| 32/500                          | 7140s  |

## Search

The larger the search scope `ef`, the higher the accuracy. `ef` has to be in the range [topk, 32768]. With similar accuracy rate, the retrieval speed is the fastest (within 500ms) compared to other index types.


### Recall

Search for 500 vectors and return the accuracy in %.
| Params (M/efConstruction/ef)  | topk=1  | topk=10  | topk=100  | topk=500  |
|-------------------------------|---------|:--------:|:---------:|:---------:|
| 16/500/1                      | 32.0    |  NA      |  NA       |  NA       |
| 16/500/10                     | 83.2    | 79.0     |  NA       |  NA       |
| 16/500/100                    | 99.4    | 99.0     | 97.6      |  NA       |
| 16/500/500                    | 100.0   | 100.0    | 99.9      | 99.7      |


### Search performance

Unit: seconds

| Search Param (ef)      | nq    | CPU                |
|------------------------|-------|--------------------|
|      1 (topk=1)        | 1     | 0.015              |
|                        | 10    | 0.018              |
|                        | 100   | 0.026              |
|                        | 500   | 0.047              |
|                        | 1000  | 0.067              |
|    10 (topk=1/10)      | 1     | 0.021/0.018        |
|                        | 10    | 0.022/0.022        |
|                        | 100   | 0.045/0.047        |
|                        | 500   | 0.154/0.158        |
|                        | 1000  | 0.280/0.277        |
| 100 (topk=1/10/100)    | 1     | 0.023/0.023/0.022  |
|                        | 10    | 0.027/0.025/0.028  |
|                        | 100   | 0.059/0.059/0.069  |
|                        | 500   | 0.291/0.242/0.285  |
|                        | 1000  | 0.501/0.485/0.565  |
