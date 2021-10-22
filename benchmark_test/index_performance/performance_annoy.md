# Milvus performance with ANNOY
This topic displays the performance of create_index & search with different index types in Milvus 1.1 for reference purposes. The test data uses 100m data from an open source dataset, sift1B, with 1 billion 128-dimensional vectors.

The following table displays the terms used in this topic:

| Term   | Description                                    |
| ---------- | ---------------------------------------- |
| nq         | Number of vectors to query. The value is defined during search.    |
| topk       | The most similar topk search results.  |
| total_time | Total time for a batch search.                 |
| avg_time   | Average search time per vector.       |
| n_trees | The number of methods of space division. |
| search_k | The number of nodes to search. -1 means 5% of the whole data. |

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

ANNOY is a a tree-based index, currently only available for CPU version. The larger the n_trees, the larger the index file size and the more time it takes to create index. With similar recall rate, it takes less time to create index than graph-based indexes but more time-consuming than IVF indexes.

Parameters：`index_file_size` = 2048

| Index Param  (n_trees)  | CPU         |
|-------------------------|-------------|
| 1                       | 638.354s    |
| 8                       | 2988.229s   |
| 16                      | 5314.657s   |
| 64                      | 20142.021s  |

## Search

Increasing `n_trees`/`search_k` improves accuracy. However, larger `search_k` means slower search speed. With similar recall rate, search using ANNOY is slower than using graph-based indexes while faster than using IVF_FLAT.


### Recall

Search for 500 vectors and return the accuracy in %.
| Search Param (n_trees/search_k)  | topk=1  | topk=10  | topk=100  | topk=500  |
|----------------------------------|---------|:--------:|-----------|-----------|
| 1/-1                             | 99.6    | 98.9     | 97.8      | 96        |
| 1/10000                          | 86.2    | 80.2     | 72.6      | 65        |
| 1/100000                         | 98      | 97.1     | 94.4      | 91.4      |
| 8/-1                             | 100     | 100      | 99.9      | 99.8      |
| 8/10000                          | 97      | 96.3     | 92.1      | 87        |
| 8/100000                         | 100     | 99.8     | 99.5      | 99        |


### Search performance

Unit: seconds

| Search Param (n_trees/search_k)  | nq    | CPU (topk = 1/10/100/500)    |
|----------------------------------|-------|------------------------------|
|               1/-1               | 1     | 7.802/2.843/2.438/2.662      |
|                                  | 10    | 4.470/3.463/3.628/3.970      |
|                                  | 100   | 12.591/14.106/15.727/9.025   |
|                                  | 500   | 32.485/33.857/33.036/34.018  |
|                                  | 1000  | 53.271/53.156/57.063/57.081  |
|             1/10000              | 1     | 0.606/0.670/0.670/0.675      |
|                                  | 10    | 0.098/0.119/0.128/0.180      |
|                                  | 100   | 0.640/0.742/0.904/1.316      |
|                                  | 500   | 2.395/2.311/2.630/4.445      |
|                                  | 1000  | 3.893/3.214/4.799/7.933      |
|             1/100000             | 1     | 0.683/1.007/0.910/0.920      |
|                                  | 10    | 0.856/0.848/0.977/1.214      |
|                                  | 100   | 3.178/2.941/3.072/3.337      |
|                                  | 500   | 14.736/13.126/5.030/6.218    |
|                                  | 1000  | 23.812/24.402/27.876/27.884  |
|               8/-1               | 1     | 4.919/.571/1.423/1.733       |
|                                  | 10    | 1.792/1.818/1.733/1.856      |
|                                  | 100   | 6.099/6.058/6.224/6.496      |
|                                  | 500   | 23.830/23.091/23.409/25.032  |
|                                  | 1000  | 42.576/44.455/45.267/49.368  |
|             8/10000              | 1     | 0.108/0.107/0.114/0.124      |
|                                  | 10    | 0.115/0.108/0.125/0.132      |
|                                  | 100   | 0.288/0.382/0.533/0.838      |
|                                  | 500   | 1.295/1.502/2.249/3.816      |
|                                  | 1000  | 2.755/2.834/3.535/6.573      |
|             8/100000             | 1     | 0.608/0.772/0.776/0.768      |
|                                  | 10    | 0.849/0.784/0.806/0.887      |
|                                  | 100   | 2.895/2.513/2.781/2.979      |
|                                  | 500   | 12.299/10.737/10.408/13.581  |
|                                  | 1000  | 21.614/21.548/21.001/23.793  |
