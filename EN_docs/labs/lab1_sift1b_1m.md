# Lab Test 1: One Million Vector Search

## 1. Prepare test data and scripts

The one million vectors used in this test are extracted from the dataset [SIFT1B](http://corpus-texmex.irisa.fr/).

#### Hardware requirements

| Component           | Minimum Config                |
| ------------------ | -------------------------- |
| OS            | Ubuntu LTS 18.04 |
| CPU           | Intel Core i5-8250U           |
| GPU           | Nvidia GeForce MX150, 2GB GDDR5  |
| GPU Driver    | CUDA 10.1, Driver 418.74 |
| Memory        | 8 GB DDR4          |
| Storage       | NVMe SSD 256 GB             |

#### Download test tools

Download the following data and scripts, and save them to a file named *milvus_sift1m*.

- [One million vectors dataset](https://pan.baidu.com/s/1nVIIxO8MnOle339iYs2dUw)
- [Query vector dataset](https://pan.baidu.com/s/1mBRM1cJZ6QWehDuddOYl4A)
- [Search ground truth](https://pan.baidu.com/s/1-95nJvW3vx2Cq9wqBWOFaA) 
- [Test scripts](/bootcamp/scripts/)

When it is done, there should be the following files in *milvus_sift1m*:

1. The *bvecs_data* file containing one million vectors
2. The *query.npy* file that has 10,000 query vectors
3. The ground_truth.txt file with the top 1000 most similar results for each query vector
4. The test script files : *milvus_bootcamp.py* and *get_id.sh*

> **Note：** Please go through the README carefully before testing with script *milvus_bootcamp.py*. Make changes to the parameters in the script to match your scenario.

## 2. Configure Milvus parameters

To optimize Milvus's performance, you can change system parameters to suit your requirements. In this test, 90% recall rate can be achieved by using the recommended values in below table. 

Configuration file: **/home/$USER/milvus/conf/server_config.yaml**

|         Parameter         | Recommended value |
| ---------------------- | ---- |
| index_building_threshold |   64   |
|    cpu_cache_capacity    |   4    |
|    use_blas_threshold    |  801   |
|          nprobe          |   32   |

After the parameter configuration, restart Milvus Docker apply them.

```bash
$ docker restart <container id>
```

## 3. Import data

#### Before the data import

- Make sure the files *bvecs_data* and *milvus_bootcamp.py* are both placed under the directory *milvus_sift1m*. 
- Make sure Milvus is already installed and started. (For details of Milvus installation, please read [Milvus Quick Start](../milvus101/quickstart.md) )

#### Import data

Go to *milvus_sift1m*, and run the following command:

```bash
$ python3 milvus_bootcamp.py --table=ann_1m_sq8 --index=ivfsq8 -t
```

You will see vectors inserted into a table named *ann_1m_sq8*, with the index_type of `IVF_SQ8`. 

![1m_import](pic/1m_import.png)

To show the available tables and number of vectors in each table, use the following command:

```bash
$ python3 milvus_bootcamp.py --show
$ python3 milvus_bootcamp.py --table=ann_1m_sq8 --rows
```

When the import is completed, a file *ann_1m_sq8_idmap.txt* will be created under *milvus_sift1m*. The file stores the vector ids and the metadata such as from which .npy file each vector comes from.   

To ensure that index is built for all imported data, go to directory  **/home/$USER/milvus/db**, and run the following statement:

```bash
$ sqlite3 meta.sqlite
```

Use below command to verify if index is built for all data:

```sqlite
sqlite> select * from TableFiles where table_id='ann_1m_sq8';
30|ann_1m_sq8|3|1565599487052367000|3|102400000|1565599495009366|1565599487052372|1190712
31|ann_1m_sq8|3|1565599495107862000|3|102400000|1565599502559292|1565599495107863|1190712
32|ann_1m_sq8|3|1565599502656466000|3|102400000|1565599510079453|1565599502656467|1190712
33|ann_1m_sq8|3|1565599510129972000|3|51200000|1565599513555987|1565599510129973|1190712
34|ann_1m_sq8|3|1565599513650120000|3|102400000|1565599521067974|1565599513650121|1190712
35|ann_1m_sq8|3|1565599521132604000|3|51200000|1565599524843984|1565599521132605|1190712
```

When you examine the verification results, you will notice that multiple records are returned with the status verification. That's due to the fact that the data in the table will be automatically divided into multiple ranges to optimize the query performance. 

The 3rd column represents the index type built for the table (`3` represents index type `IVF_SQ8`), while the 5th column shows if index is built for a particular range (`3` represents that index is already built for the range). If there are any ranges for which the index is not yet built, you can build index manually by running below statement under directory *milvus_sift1m*:

```bash
$ python3 milvus_bootcamp.py --table=ann_1m_sq8 --build
```

Go to `sqlite` interface to check that index is built for all ranges. To learn the meaning of remaining columns in table status verification results, use `.schema` under directory **/home/$USER/milvus/db**:

```sqlite
sqlite>.schema
```

## 4. Accuracy test

SIFT1B provides not only the vector dataset to search 10,000 vectors, but also the top 1000 ground truth for each vector, which allows convenient calculation of precision rate. The vector search precision of Milvus can be represented as follows:

Accuracy = Number of shared vectors (between Milvus search results and Ground truth) / (query_records * top_k)

#### Step 1: Run accuracy test scripts

To test the search precision for top 20 results of 10 vectors randomly chosen from the 10,000 query vectors, go to directory *milvus_sift1m*, and run this script:

```bash
$ python3 milvus_bootcamp.py --table=ann_1m_sq8 -q 10 -k 20 -s
```

#### Step 2: Verify test results

When the test script is completed, you will see the following test results in the file *10_20_result.csv* in *accuracy_results*. 

![1m_accu_10_20](pic/1m_accu_10_20.png)

- nq - the ordinal number of query vectors
- topk - the top k most similar result vectors for the query vectors 
- total_time - the total query elapsed time (in seconds)
- avg_time - the average time to query one vector (in seconds)
- recall - the accuracy calculated by comparing Milvus search results and ground truth

The accuracy rate has a positive correlation with search parameter `nprobe` (number of sub-spaces searched). In this test, when the `nprobe` = 32, the accuracy can reach > 90%.  However, as the `nprobe` gets bigger, the search time will be longer. 

Therefore, based on your data distribution and business scenario, you need to edit `nprobe` to optimize the trade-off between accuracy and search time. 

## 5. Performance test

To test search performance, go to directory *milvus_sift1m*, and run the following script: 

```bash
$ python3 milvus_bootcamp.py --table=ann_1m_sq8 -s
```

When the execution is completed, you will see the test results in the file *xxx_results.csv* ('xxx' represents the execution time) in *performance_results*. Below is a partial display of the results:

![1m_per_10_20](pic/1m_per_10_20.png)

- nq - the number of query vectors
- topk - the top k most similar vectors for the query vectors 
- total_time - the total query elapsed time (in seconds)
- avg_time - the average time to query one vector (in seconds)

> **Note:** 1. In milvus_bootcamp.py, `nq` is set to be 1, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800 respectively, and `topk` is set to be 1, 20, 50, 100, 300, 500, 800, 1000. 
>
> 2. To run the first vector search, some extra time is needed to load the data (from the disk) to the memory.

> **Tip:** It is recommended to run several performance tests continuously, and use the search time of the second run. If the tests are executed intermittently, Intel CPU may downgrade to base clock speed.
