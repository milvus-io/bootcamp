# Milvus performance with IVF_FLAT
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

During this test, the data was imported into Milvus in batches of 100,000, and it took about 1.2s to import 100,000 vectors of 128 dimensions.

## Create Index

IVF_FLAT is a quantization-based index, index file size of sift100m data is about 50GB. GPU takes much less time to create an index than CPU does.

Parameters：`nlist` = 4096, `index_file_size` = 2048

| Index Param (nlist) | CPU | GPU |
| ------------------- | --- | --- |
| 4096 | 1180.32s | 285.28s |

## Search

Search accuracy is relatively high when `nprobe` is large enough. With same parameters, search speed is slowest among IVF indexes (but faster than FLAT). Query time increases when `nq` or `nprobe` increases.

### Recall

Search for 500 vectors and return the accuracy in %.
| Param (nprobe) | topk=1 | topk=10 | topk=100 | topk=500 |
| -------------- | ------ | ------- | -------- | -------- |
| 1              | 44.2   | 44.2    | 36.2     | 31.5     |
| 128            | 100    | 99.8    | 99.5     | 99.1     |
| 256            | 100    | 100     | 99.9     | 99.9     |


### Search performance

Unit: seconds

| Search Param (nprobe) | nq   | CPU (topk = 1/10/100/500) | GPU (topk = 1/10/100/500) |
| --------------------- | ---- | ------------------------- | ------------------------- |
| 1                     | 1    | 0.022/0.024/0.023/0.050   | 2.787/2.752/2.818/2.908   |
| 1                     | 10   | 0.043/0.044/0.059/0.060   | 2.802/2.853/3.117/3.430   |
| 1                     | 100  | 0.116/0.095/0.104/0.183   | 2.922/3.125/3.476/3.511   |
| 1                     | 500  | 0.332/0.319/0.366/0.605   | 2.958/3.362/2.771/2.902   |
| 1                     | 1000 | 0.579/0.606/0.782/0.967   | 3.065/3.240/2.820/3.080   |
| 128                   | 1    | 0.040/0.042/0.044/0.075   | 2.758/2.787/2.804/2.956   |
| 128                   | 10   | 0.513/0.522/0.505/0.570   | 2.834/2.789/3.181/3.327   |
| 128                   | 100  | 1.942/1.920/2.084/2.305   | 2.992/3.331/3.567/3.602   |
| 128                   | 500  | 7.874/7.866/8.014/8.002   | 3.713/4.114/3.537/3.606   |
| 128                   | 1000 | 14.958/14.989/15.065/15.456 | 4.567/4.888/4.394/4.672 |
| 256                   | 1    | 0.053/0.053/0.055/0.078   | 2.790/2.781/2.827/2.885   |
| 256                   | 10   | 0.911/0.886/0.894/0.978   | 2.802/2.885/3.136/3.385   |
| 256                   | 100  | 3.634/3.691/3.775/3.620   | 3.226/3.445/3.723/3.824   |
| 256                   | 500  | 14.754/14.810/14.828/15.094 | 4.322/4.897/4.206/4.466 |
| 256                   | 1000 | 28.305/28.294/28.480/28.914 | 6.063/6.399/6.004/6.119 |
