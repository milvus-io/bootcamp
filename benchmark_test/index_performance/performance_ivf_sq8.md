# Milvus performance with IVF_SQ8
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

IVF_SQ8 is a quantization-based index, index file size of sift100m data is much less than that of IVF_FLAT. GPU takes much less time to create an index than CPU does.

Parameters：`nlist` = 4096, `index_file_size` = 2048

| Index Param (nlist) | CPU | GPU |
| ------------------- | --- | --- |
| 4096 | 1409.197s | 198.952s |

## Search

Search accuracy is relatively high when `nprobe` is large enough. With same parameters, search speed is faster than that of IVF_FLAT. Query time increases when `nq` or `nprobe` increases. When both `nprobe` and `nq` are large, search with GPU is faster than search with CPU.

### Recall

Search for 500 vectors and return the accuracy in %.
| Param (nprobe) | topk=1 | topk=10 | topk=100 | topk=500 |
| -------------- | ------ | ------- | -------- | -------- |
| 1              | 48.2   | 41.7    | 35.9     | 31.2     |
| 64             | 97     | 97.3    | 97.3     | 96.6     |
| 128            | 98     | 98      | 98.4     | 98.4     |


### Search performance

Unit: seconds

| Search Param  (nprobe)  | nq    | CPU  (topk = 1/10/100/500)   | GPU  (topk = 1/10/100/500)   |
|-------------------------|-------|------------------------------|------------------------------|
|          1              | 1     | 0.0169/0.0150/0.0183/0.0314  | 0.7895/0.7825/0.7958/0.8298  |
|                         | 10    | 0.0381/0.0384/0.0638/0.1600  | 0.8058/0.8155/0.9103/1.0220  |
|                         | 100   | 0.1482/0.1675/0.4387/0.8686  | 0.8568/0.9404/1.1035/1.3566  |
|                         | 500   | 0.4140/0.6520/1.1842/3.2874  | 0.9819/1.1539/1.2960/2.5181  |
|                         | 1000  | 0.9671/1.3610/2.2811/5.1424  | 1.15661.3024/1.7897/4.1139   |
|           64            | 1     | 0.0362/0.0382/0.0433/0.0752  | 0.7739/0.7704/0.7878/0/8245  |
|                         | 10    | 0.2296/0.1981/0.2765/0.3603  | 0.8058/0/8214/0.8813/1.0132  |
|                         | 100   | 0.5608/0.5196/0.7518/0.1621  | 0.9348/1.0211/1.1608/1.4401  |
|                         | 500   | 1.9919/2.0679/2.6038/2.6790  | 1.3918/1.5553/1.6984/2.9047  |
|                         | 1000  | 3.4634/3.5184/3.9688/6.5398  | 1.9493/2.1058/2.3981/5.0772  |
|           128           | 1     | 0.0355/0.0413/0.0445/0.0835  | 0.7775/0.7823/0.7785/0.8350  |
|                         | 10    | 0.3028/0.2735/0.3374/0.4149  | 0.8170/0.8228/0.9204/1.0171  |
|                         | 100   | 0.7516/0.8597/1.3901/1.3657  | 1.0214/1.0992/1.2455/1.5132  |
|                         | 500   | 2.9589/3.0997/3.5877/4.6188  | 1.8101/1.9524/2.0786/3.3630  |
|                         | 1000  | 5.6767/5.7210/6.4196/7.8935  | 2.8010/2.9392/3.3580/5.8616  |
