# Performance Benchmark

This topic displays the performance of different index types in Milvus 0.11.0 for reference purposes. The test data uses an open source dataset, sift1B, with 1,000 million 128-dimensional vectors.

The following table displays the terms used in this topic:

| Term   | Description                                    |
| ---------- | ---------------------------------------- |
| nq         | Number of vectors to query. The value is defined during search.    |
| topk       | The most similar topk search results.  |
| total_time | Total time for a batch search.                 |
| avg_time   | Average search time per vector.       |
| nprobe     | Number of buckets to search during a query. The value is defined during search.  |

Performance is correlated with nprobe. The greater the value of nprobe, the lower the performance, but the higher the accuracy. You can set nprobe per specific scenarios. In this topic, nprobe is set to 32.

Test parameter Settings: Nlist =4096, Sift1m, segment_row_limit =1000000

Sift10m, SiFT100m, Segment_row_limit =2000000

Refer to [https://medium.com/@milvusio/how-to-choose-an-index-in-milvus-4f3d15259212](https://medium.com/@milvusio/how-to-choose-an-index-in-milvus-4f3d15259212) to learn how to choose indexes.

[SiFT1m test result](Performance_bench.md# sift1m): In this test, the first 1,000,000 vectors from the sift1B data set are selected.

[SiFT10m test results](Performance_bench.md# sift10m): In this test, the first 10,000,000 vectors from the sift1B dataset are selected.

[Results of the SIFT100M test](Performance_bench.md# sift100m): In this test, the first 100,000,000 vectors from the sift1B dataset are selected.

In this test, the data was imported into Milvus in batches of 100,000, and the import time of 100,000 128-dimensional vectors was approximately  : 2S

## sift1m

##### index：ivf_sq8

(Index creation time：8s)

(nprobe:256)

**accuracy：**

The department randomly took out 500 vector queries from the vector set to be queried, and the average value of the 500 results was taken out for recall rate, which was tested for a total of three times. The 500 vectors taken out each time were not completely the same

| accuracy | 第一次 | 第二次 | 第三次 |
| -------- | ------ | ------ | ------ |
| topk=1   | 98.2%  | 98.0%  | 98.8%  |
| topk=10  | 98.9%  | 97.8%  | 97.5%  |
| topk=100 | 98.2%  | 97.8%  | 97.7%  |
| topk=500 | 98.4%  | 96.9%  | 98.4%  |

**performance：**

Query (CPU)

| Time(s) | Topk =1 | Topk=10 | Topk=100 | Topk =500 |
| ------- | ------- | ------- | -------- | --------- |
| nq =1   | 0.0086  | 0.0063  | 0.0065   | 0.0082    |
| nq =10  | 0.0159  | 0.0138  | 0.0153   | 0.0187    |
| nq =100 | 0.0285  | 0.0318  | 0.0336   | 0.0422    |
| nq =500 | 0.0976  | 0.0988  | 0.1154   | 0.1590    |

##### 索引ivf_sq8h

(Index creation time：7.8s)

(nprobe:256)

**accuracy：**

The department randomly took out 500 vector queries from the vector set to be queried, and the average value of the 500 results was taken out for recall rate, which was tested for a total of three times. The 500 vectors taken out each time were not completely the same

| accuracy  | 第一次 | 第二次 | 第三次 |
| --------- | ------ | ------ | ------ |
| Topk =1   | 99.4%  | 98.8%  | 99.0%  |
| Topk =10  | 98.7%  | 98.7%  | 98.4%  |
| Topk =10  | 98.7%  | 98.7%  | 98.4%  |
| Topk =500 | 98.0%  | 98.1%  | 98.0%  |

**performance**：

Query (CPU)

| time(S) | topk=1 | topk=10 | topk=100 | topk=500 |
| ------- | ------ | ------- | -------- | -------- |
| nq=1    | 0.0163 | 0.0062  | 0.0065   | 0.0076   |
| nq=10   | 0.0141 | 0.0142  | 0.0152   | 0.0168   |
| nq=100  | 0.0332 | 0.0334  | 0.0369   | 0.0495   |
| nq=500  | 0.1093 | 0.1054  | 0.1235   | 0.1713   |




## sift10m

##### index：ivf_sq8

(Index creation time：64s)

(nprobe:128)

**accuracy：**

The department randomly took out 500 vector queries from the vector set to be queried, and the average value of the 500 results was taken out for recall rate, which was tested for a total of three times. The 500 vectors taken out each time were not completely the same

| accuracy | 第一次 | 第二次 | 第三次 |
| -------- | ------ | ------ | ------ |
| topk=1   | 98.0%  | 98.4%  | 98.6%  |
| topk=10  | 98.2%  | 97.7%  | 98.2%  |
| topk=100 | 98.3%  | 98.0%  | 98.1%  |
| topk=500 | 97.8%  | 97.5%  | 97.4%  |

**performance**：

Query (CPU)

| time(S) | topk=1 | topk=10 | topk=100 | topk=500 |
| ------- | ------ | ------- | -------- | -------- |
| nq=1    | 0.0189 | 0.0176  | 0.0183   | 0.0253   |
| nq=10   | 0.0594 | 0.0570  | 0.0598   | 0.0638   |
| nq=100  | 0.0982 | 0.1011  | 0.1056   | 0.1234   |
| nq=500  | 0.3526 | 0.3371  | 0.3712   | 0.4370   |



##### index：ivf_sq8h

(nprobe:128)

**accuracy：**

The department randomly took out 500 vector queries from the vector set to be queried, and the average value of the 500 results was taken out for recall rate, which was tested for a total of three times. The 500 vectors taken out each time were not completely the same

| accuracy | 第一次 | 第二次 | 第三次 |
| -------- | ------ | ------ | ------ |
| topk=1   | 97.0%  | 98.6%  | 97.6%  |
| topk=10  | 98.1%  | 98.0%  | 97.7%  |
| topk=100 | 98.1%  | 98.2%  | 98.0%  |
| topk=500 | 97.6%  | 97.7%  | 97.6%  |

**performance**：

Query (CPU)

| time(S) | topk=1 | topk=10 | topk=100 | topk=500 |
| ------- | ------ | ------- | -------- | -------- |
| nq=1    | 0.0180 | 0.0187  | 0.0189   | 0.0369   |
| nq=10   | 0.0584 | 0.0684  | 0.0603   | 0.0692   |
| nq=100  | 0.0971 | 0.1038  | 0.1033   | 0.1203   |
| nq=500  | 0.3494 | 0.3496  | 0.0382   | 0.4585   |
| nq=1000 | 0.7034 | 0.6702  | 0.7381   | 0.8805   |



## sift100m

##### index：ivf_sq8

(Index creation time：300s)

(nprobe:64)

**accuracy：**

The department randomly took out 500 vector queries from the vector set to be queried, and the average value of the 500 results was taken out for recall rate, which was tested for a total of three times. The 500 vectors taken out each time were not completely the same

| accuracy | 第一次 | 第二次 | 第三次 |
| -------- | ------ | ------ | ------ |
| topk=1   | 97.2%  | 97.2%  | 97.6%  |
| topk=10  | 97.3%  | 97.5%  | 97.4%  |
| topk=100 | 97.2%  | 97.5%  | 97.5%  |
| topk=500 | 96.6%  | 96.8%  | 96.9%  |



**performance**：

Query (CPU)

| time(S) | topk=1 | topk=10 | topk=100 | topk=500 |
| ------- | ------ | ------- | -------- | -------- |
| nq=1    | 0.1293 | 0.1388  | 0.148    | 0.2495   |
| nq=10   | 0.3315 | 0.3185  | 0.3366   | 0.3716   |
| nq=100  | 0.5495 | 0.5313  | 0.5587   | 0.6257   |
| nq=500  | 1.7314 | 1.8109  | 2.119    | 2.1481   |
| nq=1000 | 3.3412 | 3.3696  | 3.5066   | 4.1944   |

##### index：ivf_sq8h

(nprobe:64)

**accuracy：**

The department randomly took out 500 vector queries from the vector set to be queried, and the average value of the 500 results was taken out for recall rate, which was tested for a total of three times. The 500 vectors taken out each time were not completely the same

| accuracy | 第一次 | 第二次 | 第三次 |
| -------- | ------ | ------ | ------ |
| topk=1   | 98.2%  | 96.0%  | 98.2%  |
| topk=10  | 97.4%  | 97.2%  | 97.3%  |
| topk=100 | 97.7%  | 97.3%  | 97.5%  |
| topk=500 | 97.1%  | 96.7%  | 97.0%  |

**performance**：

| time(S) | topk=1 | topk=10 | topk=100 | topk=500 |
| ------- | ------ | ------- | -------- | -------- |
| nq=1    | 0.1499 | 0.1477  | 0.1415   | 0.2358   |
| nq=10   | 0.3340 | 0.3572  | 0.3330   | 0.4376   |
| nq=100  | 0.5142 | 0.5120  | 0.6066   | 0.6223   |
| nq=500  | 1.6685 | 1.6852  | 1.7850   | 2.0676   |
| nq=1000 | 3.2482 | 3.2930  | 3.2112   | 3.8729   |

### 
