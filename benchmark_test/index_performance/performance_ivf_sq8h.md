# Milvus performance with IVF_SQ8H
This topic displays the performance of create_index & search with different index types in Milvus 1.1 for reference purposes. The test data uses 100m data from an open source dataset, sift1B, with 1 billion 128-dimensional vectors.

The following table displays the terms used in this topic:

| Term   | Description                                    |
| ---------- | ---------------------------------------- |
| nq         | Number of vectors to query. The value is defined during search.    |
| topk       | The most similar topk search results.  |
| total_time | Total time for a batch search.                 |
| avg_time   | Average search time per vector.       |
| nprobe     | Number of buckets to search during a query. The value is defined during search.  |

Performance is correlated with nprobe. The greater the value of nprobe, the lower the performance, but the higher the accuracy. You can set nprobe per specific scenarios.

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

IVF_SQ8H is the optimized version of IVF_SQ8 that requires both CPU and GPU to work. Unlike IVF_SQ8, IVF_SQ8H uses a GPU-based coarse quantizer, which greatly reduces time to quantize. IVF_SQ8H is only available with Milvus-GPU version & GPU enabled.

Parameters：`nlist` = 4096, `index_file_size` = 2048

| Index Param (nlist) | GPU |
|---------------------|-----|
| 4096 | 182.694s |

## Search

When nq >= gpu_search_threshold, it is same as IVF_SQ8 using GPU. When nq < gpu_search_threshold, it uses both GPU & CPU to search. When nq is small, search is faster than IVF_SQ8 fully using GPU.

### Recall

Search for 500 vectors and return the accuracy in %.
| Param (nprobe) | topk=1 | topk=10 | topk=100 | topk=500 |
| -------------- | ------ | ------- | -------- | -------- |
| 1              | 45.8   | 42.3    | 34.8     | 30.3     |
| 64             | 96.8   | 97.4    | 97.3     | 96.7     |
| 128            | 96.8   | 98.1    | 98.4     | 98.4     |


### Search performance

- gpu_search_threshold: 1001
- Unit: seconds

| Search Param  (nprobe)  | nq    | CPU & GPU  (topk = 1/10/100/500)  |
|-------------------------|-------|-----------------------------------|
|          1              | 1     | 0.040/0.036/0.038/0.050           |
|                         | 10    | 0.058/0.064/0.093/0.138           |
|                         | 100   | 0.126/0.150/0.234/0.597           |
|                         | 500   | 0.305/0.383/0.720/1.952           |
|                         | 1000  | 0.598/0.782/1.390/3.991           |
|           64            | 1     | 0.057/0.065/0.066/0.126           |
|                         | 10    | 0.172/0.193/0.288/0.379           |
|                         | 100   | 0.491/0.533/0.756/1.080           |
|                         | 500   | 1.809/1.898/2.199/3.431           |
|                         | 1000  | 3.054/3.191/3.924/6.113           |
|          128            | 1     | 0.062/0.061/0.074/0.138           |
|                         | 10    | 0.256/0.307/0.314/0.494           |
|                         | 100   | 0.739/0.783/1.021/1.338           |
|                         | 500   | 2.694/2.817/3.217/4.436           |
|                         | 1000  | 5.095/5.064/5.717/8.275           |
