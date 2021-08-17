# Milvus performance
This topic displays the performance of different index types in Milvus1.1 for reference purposes. The test data uses an open source dataset, sift1B, with 1 billion 128-dimensional vectors.

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



## Performance result

[sift1m test result](performance_benchmark.md#sift1m): In this test, the first 1,000,000 vectors in the sift1B dataset are selected.

[sift10m test result](performance_benchmark.md#sift10m): In this test, the first 10,000,000 vectors in the sift1B data set are selected.

[sift100m test result](performance_benchmark.md#sift100m): In this test, the first 100,000,000 vectors in the sift1B data set were selected.

During this test, the data was imported into milvus in batches of 100,000, and it took about 0.7S to import 100,000 vectors of 128 dimensions.

## Hardware Configuration:
| component  | config                             |
| ---------- | ---------------------------------- |
| OS         | Ubuntu LTS 18.04                   |
| CPU        | Intel Core i7-8700                 |
| GPU0       | Nvidia GeForce GTX 1060, 6GB GDDR5 |
| GPU1       | Nvidia GeForce GTX 1660, 6GB GDDR5 |
| GPU driver | CUDA 10.1, Driver 418.74           |
| Memory     | 16 GB DDR4 ( 2400 Mhz ) x 4        |
| Storage    | SATA 3.0 SSD 256 GB                |

## Sift1m

### Ivf_sq8 index

**Time spent building index**: 27.8 S, and `nlist` = 4096.

**Search parameter**: {nprobe: 256}

**Accuracy**:

In this section, 500 vectors are randomly removed from the set of vectors to be queried. The recall is taken as the average of these 500 results and tested three times in total, each time the 500 vectors taken out are not exactly the same.

| accuracy(%) | first | second | third |
| ----------- | ----- | ------ | ----- |
| topk=1      | 98.6  | 99.2   | 98.4  |
| topk=10     | 98.2  | 98.3   | 98.7  |
| topk=100    | 98.6  | 98.5   | 98.6  |
| topk=500    | 97.9  | 97.8   | 98.0  |

**Performance：p99 latency**
(use cpu search)

| time(s) | topk=1  | topk=10 | topk=100 | topk=500 |
| ------- | ------- | ------- | -------- | -------- |
| nq=1    | 0.00370 | 0.00233 | 0.00244  | 0.00273  |
| nq=10   | 0.0137  | 0.0151  | 0.0131   | 0.0151   |
| nq=100  | 0.0847  | 0.0846  | 0.0887   | 0.0997   |
| nq=500  | 0.366   | 0.366   | 0.375    | 0.419    |
| nq=1000 | 0.689   | 0.688   | 0.699    | 0.794    |

### ivf_sq8h index

**Time spent building index**: 27.0 S, and `nlist` = 4096.

**Search parameter**: {nprobe: 256}

**Accuracy**:

In this section, 500 vectors are randomly removed from the set of vectors to be queried. The recall is taken as the average of these 500 results and tested three times in total, each time the 500 vectors taken out are not exactly the same.

| accuracy(%) | first | second | third |
| ----------- | ----- | ------ | ----- |
| topk=1      | 99.2  | 98.2   | 98.8  |
| topk=10     | 98.5  | 98.7   | 98.4  |
| topk=100    | 98.7  | 98.7   | 98.6  |
| topk=500    | 98.2  | 98.2   | 98.1  |

**Performance：p99 latency**
(use cpu search)

| time(s) | topk=1  | topk=10 | topk=100 | topk=500 |
| ------- | ------- | ------- | -------- | -------- |
| nq=1    | 0.00268 | 0.00269 | 0.00277  | 0.00308  |
| nq=10   | 0.0175  | 0.0136  | 0.0132   | 0.0155   |
| nq=100  | 0.0836  | 0.0839  | 0.0876   | 0.0981   |
| nq=500  | 0.362   | 0.355   | 0.371    | 0.409    |
| nq=1000 | 0.679   | 0.693   | 0.693    | 0.771    |

## sift10m

### ivf_sq8 index

**Time spent building index**: 62.8 S, and `nlist` = 4096.

**Search parameter**: {nprobe:128}

**Accuracy：**

In this section, 500 vectors are randomly removed from the set of vectors to be queried. The recall is taken as the average of these 500 results and tested three times in total, each time the 500 vectors taken out are not exactly the same.

| accuracy(%) | first | second | third |
| ----------- | ----- | ------ | ----- |
| topk=1      | 97.4  | 97.0   | 97.8  |
| topk=10     | 98.2  | 98.0   | 98.1  |
| topk=100    | 98.1  | 98.1   | 98.3  |
| topk=500    | 97.3  | 97.4   | 97.7  |

**Performance：p99 latency**
(use cpu search)

| time(s) | topk=1(s) | topk=10(s) | topk=100(s) | topk=500(s) |
| ------- | --------- | ---------- | ----------- | ----------- |
| nq=1    | 0.00547   | 0.0074     | 0.0057      | 0.00626     |
| nq=10   | 0.0569    | 0.0575     | 0.0571      | 0.0593      |
| nq=100  | 0.398     | 0.480      | 0.464       | 0.423       |
| nq=500  | 1.69      | 1.69       | 1.71        | 1.74        |
| nq=1000 | 3.24      | 3.28       | 3.26        | 3.45        |

### ivf_sq8h index

**Time spent building index**: 59.6 S, and `nlist` = 4096.

**Search parameter**: {nprobe:128}

**Accuracy：**

In this section, 500 vectors are randomly removed from the set of vectors to be queried. The recall is taken as the average of these 500 results and tested three times in total, each time the 500 vectors taken out are not exactly the same.

| accuracy(%) | first | second | third |
| ----------- | ----- | ------ | ----- |
| topk=1      | 97.4  | 97.4   | 99.0  |
| topk=10     | 98.3  | 97.8   | 98.1  |
| topk=100    | 98.1  | 98.0   | 98.0  |
| topk=500    | 97.5  | 97.5   | 97.5  |

**Performance：p99 latency**

| time(s) | topk=1  | topk=10 | topk=100 | topk=500 |
| ------- | ------- | ------- | -------- | -------- |
| nq=1    | 0.00574 | 0.00741 | 0.00569  | 0.00626  |
| nq=10   | 0.0556  | 0.0555  | 0.0567   | 0.0586   |
| nq=100  | 0.411   | 0.423   | 0.421    | 0.45     |
| nq=500  | 1.71    | 1.74    | 1.74     | 1.77     |
| nq=1000 | 3.25    | 3.28    | 3.26     | 3.45     |

## sift100m

### ivf_sq8 index

**Time spent building index**: 440.1 S, and `nlist` = 4096.

**Search parameter**: {nprobe:64}

**Accuracy：**

In this section, 500 vectors are randomly removed from the set of vectors to be queried. The recall is taken as the average of these 500 results and tested three times in total, each time the 500 vectors taken out are not exactly the same.

| accuracy(%) | first | second | third |
| ----------- | ----- | ------ | ----- |
| topk=1      | 97.0  | 95.8   | 97.6  |
| topk=10     | 97.4  | 97.3   | 97.4  |
| topk=100    | 97.6  | 97.2   | 97.3  |
| topk=500    | 97.2  | 96.4   | 96.6  |

**Performance：p99 latency**
(use cpu search)

| time(s) | topk=1 | topk=10 | topk=100 | topk=500 |
| ------- | ------ | ------- | -------- | -------- |
| nq=1    | 0.0274 | 0.0274  | 0.0306   | 0.0334   |
| nq=10   | 0.365  | 0.305   | 0.359    | 0.374    |
| nq=100  | 2.20   | 2.21    | 2.23     | 2.28     |
| nq=500  | 8.57   | 8.63    | 8.69     | 8.93     |
| nq=1000 | 16.1   | 16.0    | 16.0     | 16.7     |

### ivf_sq8h index

**Time spent building index**: 468.9 S, and `nlist` = 4096.

**Search parameter**: {nprobe:64}

**Accuracy：**

In this section, 500 vectors are randomly removed from the set of vectors to be queried. The recall is taken as the average of these 500 results and tested three times in total, each time the 500 vectors taken out are not exactly the same.

| accuracy(%) | first | second | third |
| ----------- | ----- | ------ | ----- |
| topk=1      | 96.6  | 97.0   | 97.4  |
| topk=10     | 97.6  | 97.1   | 97.2  |
| topk=100    | 97.4  | 97.4   | 97.3  |
| topk=500    | 96.8  | 96.7   | 96.6  |


**Performance：p99 latency**
(use cpu search)

| time(s) | topk=1 | topk=10 | topk=100 | topk=500 |
| ------- | ------ | ------- | -------- | -------- |
| nq=1    | 0.0320 | 0.0335  | 0.0321   | 0.0375   |
| nq=10   | 0.299  | 0.302   | 0.305    | 0.309    |
| nq=100  | 2.25   | 2.27    | 2.29     | 2.45     |
| nq=500  | 9.27   | 9.18    | 9.36     | 9.91     |
| nq=1000 | 17.5   | 17.5    | 17.8     | 18.0     |
