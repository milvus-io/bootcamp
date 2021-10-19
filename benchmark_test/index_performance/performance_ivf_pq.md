# Milvus performance with IVF_PQ
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

IVF_PQ is a quantization-based index. The index file is very small (about 4G), and the indexing speed is relatively fast. GPU creates index faster than CPU.

Parameters：`index_file_size` = 2048

| Index Param (nlist/m/nbits) | CPU | GPU |
| ------------------- | --- | --- |
| 2048/32/8 | 739s  | 250s |
| 4096/32/8 | 1267s | 289s |

## Search

Search accuracy rate is very low (54% to 75%), the smaller the indexing parameter `m`, the lower the rate. The retrieval speed in the IVF index is the fastest (CPU) with the same parameters with lower accuracy. No matter how large the `nprobe`/`nq` is, the time is about 2s for GPU retrieval.


### Recall

Search for 500 vectors and return the accuracy in %.
| Param (nlist/m/nbits/nprobe)  | topk=1    | topk=10  | topk=100  | topk=500  |
|-------------------------------|-----------|:--------:|:---------:|:---------:|
| 2048/32/8/1                   | 32.6%     | 40.2%    | 38.5%     | 35.1%     |
| 2048/32/8/16                  | 54.6%     | 64.7%    | 70.3%     | 72.4%     |
| 2048/32/8/64                  | 53.4%     | 65.9%    | 72.0%     | 75.3%     |
| 2048/32/8/128                 | 54.0%     | 66.0%    | 71.9%     | 75.3%     |


### Search performance

- nlist: 2048
- m: 32
- Unit: seconds

| Search Param (nprobe)  | nq    | CPU (topk = 1/10/100/500)  | GPU (topk = 1/10/100/500)  |
|------------------------|-------|----------------------------|----------------------------|
|         1              | 1     | 0.022/0.024/0.026/0.034/   | 1.835/1.676/1.781/1.879/   |
|                        | 10    | 0.048/0.056/0.062/0.093/   | 1.779/1.954/1.866/1.946/   |
|                        | 100   | 0.110/0.110/0.288/0.325/   | 1.927/1.836/2.031/2.042/   |
|                        | 500   | 0.280/0.287/0.364/0.640/   | 1.818/1.960/1.886/2.087/   |
|                        | 1000  | 0.546/0.738/0.967/1.210/   | 2.008/2.045/2.086/2.098/   |
|           16           | 1     | 0.040/0.043/0.046/0.068/   | 1.902/1.806/1.888/2.007/   |
|                        | 10    | 0.151/0.152/0.148/0.159/   | 1.809/1.953/1.844/1.913/   |
|                        | 100   | 0.253/0.249/0.288/0.303/   | 1.904/1.811/1.930/1.860/   |
|                        | 500   | 0.727/0.692/0.819/1.069/   | 1.860/1.917/1.858/2.057/   |
|                        | 1000  | 1.196/1.248/1.363/1.737/   | 1.914/1.844/2.013/2.182/   |
|           32           | 1     | 0.037/0.042/0.046/0.087/   | 1.804/1.900/1.823/1.896/   |
|                        | 10    | 0.225/0.199/0.216/0.267/   | 1.938/1.820/1.895/1.843/   |
|                        | 100   | 0.366/0.376/0.394/0.448/   | 1.841/1.868/1.857/1.932/   |
|                        | 500   | 1.060/1.078/1.108/1.332/   | 1.932/1.841/1.880/1.993/   |
|                        | 1000  | 1.930/1.955/2.092/2.526/   | 1.880/1.945/1.943/2.322/   |
