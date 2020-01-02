# Lab Test 1: One Million Vector Search

## 1. Prepare test data and scripts

The one million vectors used in this test are extracted from the dataset [SIFT1B](http://corpus-texmex.irisa.fr/).

#### Hardware requirements

The following configuration has been tested:

| Component           |  Configuration                |
| ------------------ | -------------------------- |
| Operating System           | Ubuntu LTS 18.04 |
| CPU           | Intel Core i5-8250U           |
| GPU           | NVIDIA GeForce MX150, 2GB GDDR5  |
| GPU Driver    | Driver 418.74 |
| Memory        | 8 GB DDR4          |
| Storage       | NVMe SSD 256 GB             |
| Milvus       | 0.6.0            |
| pymilvus       | 0.2.6            |

#### Download test tools

Download the following data and scripts:

- 1 million test data: [https://pan.baidu.com/s/19fj1FUHfYZwn9huhgX4rQQ](https://pan.baidu.com/s/19fj1FUHfYZwn9huhgX4rQQ)
- Query data: [https://pan.baidu.com/s/1nVAFi5_DBZS2eazA7SN0VQ](https://pan.baidu.com/s/1nVAFi5_DBZS2eazA7SN0VQ)
- Ground truth: [https://pan.baidu.com/s/1KGlBiJvuGpqjbZOIpobPUg](https://pan.baidu.com/s/1KGlBiJvuGpqjbZOIpobPUg)
- Test scripts: [/bootcamp/benchmark_test/scripts/](/benchmark_test/scripts/)

Create a folder named `milvus_sift1m` and move all downloaded files to the folder:

- Unzip the 1 million test data to get the `bvecs_data` folder that contains 10 `npy` files. Each `npy` file contains 100,000 vectors.
- Unzip the query data to get the `query_data` folder that contains `query.npy`, which contains 10,000 vectors to query.
- Unzip the ground truth data to get the gnd folder with `ground_truth_1M.txt`, which contains the locations of top 1000 similar vectors in the query data.
- The test script files contain the following Python scripts: `milvus_load.py`, `milvus_toolkit.py`, `milvus_search.py`, and `milvus_compare.py`.

> Note: Make sure that `bvecs_data`, `query_data`, `gnd`, and test scripts are in the same folder level.

## 2. Configure Milvus parameters

To optimize the performance of Milvus, you can change Milvus parameters based on data distribution, performance, and accuracy requirements. In this test, 90% or higher recall rate can be achieved by using the recommended values in the following table.

Configuration file: `/home/$USER/milvus/conf/server_config.yaml`

|         Parameter         | Recommended value |
| ---------------------- | ---- |
|       `cpu_cache_capacity`   |   4   |
|         `gpu_resource_config`.`cache_capacity`      |  1    |
|         `use_blas_threshold`	                |   801     |
|         `gpu_search_threshold`	                |   1001     |
|         `search_resources`	                |   gpu0     |

Refer to [Milvus Configuration](https://github.com/milvus-io/docs/blob/0.6.0/reference/milvus_config.md) for more information.

Use default values for other parameters. After setting parameter values, restart Milvus Docker to apply all changes.

```bash
$ docker restart <container id>
```

## 3. Create a table and build indexes

Make sure Milvus is already installed and started. (For details of Milvus installation, please read [Milvus Quick Start](../EN_getting_started/basics/quickstart.md)).


Go to `milvus_sift1m`, and run the following command to create a table and build indexes:

```bash
$ python3 milvus_toolkit.py --table ann_1m_sq8h --dim 128 -c
$ python3 milvus_toolkit.py --table ann_1m_sq8h --index sq8h --build
```

Vectors are then inserted into a table named `ann_1m_sq8h`, with the index_type of `IVF_SQ8H`. 

To show the available tables and number of vectors in each table, use the following command:

```bash
#Show available tables
$ python3 milvus_toolkit.py --show
#Show the number of rows in ann_1m_sq8h
$ python3 milvus_toolkit.py --table ann_1m_sq8h --rows
#Show the index type of ann_1m_sq8h
$ python3 milvus_toolkit.py --table ann_1m_sq8h --desc_index
```

## 4.  Import data

Make sure table ann_1m_sq8 is successfully created.

Run the following command to import 1,000,000 rows of data:

```bash
$ python3 milvus_load.py --table=ann_1m_sq8h -n
```

You can see that all data is imported from the file for once.

Run the following command to check the number of rows in the table:

```bash
$ python3 milvus_toolkit.py --table=ann_1m_sq8h --rows
```

To make sure that all data imported to Milvus has indexes built. Navigate to `/home/$USER/milvus/db` and enter the following command:

```bash
$ sqlite3 meta.sqlite
```

In sqlite3 CLI, enter the following command to check the current status:

```sql
sqlite> select * from TableFiles where table_id='ann_1m_sq8h';
```

Milvus divides a vector table into shards for storage. So, a query returns multiple records. The third column specifies the index type and 5 stands for IVF_SQ8H. The fifth column specifies the build status of the index and 3 indicates that index building is complete for the shard. If index building is not complete for a specific shard, you can manually build indexes for the shard.

Exit sqlite CLI:

```sql
sqlite> .quit
```

Enter `milvus_sift1m` and run the following command:

```bash
$ python3 milvus_toolkit.py --table=ann_1m_sq8h --index=sq8h --build 
```

After manually building indexes, enter sqlite CLI again and make sure that index building has been completed for all shards. To understand the meanings of other columns, navigate to `/home/$USER/milvus/db` and enter the following command in the sqlite CLI:

```bash
$ sqlite3 meta.sqlite
sqlite>.schema
```

## 5. Accuracy test

SIFT1B provides not only the vector dataset to search 10,000 vectors, but also the top 1000 ground truth for each vector, which allows convenient calculation of precision rate. The vector search accuracy of Milvus can be represented as follows:

Accuracy = Number of shared vectors (between Milvus search results and Ground truth) / (query_records * top_k)

#### Step 1: Run query script

To test the search precision for top 20 results of 10 vectors randomly chosen from the 10,000 query vectors, go to directory `milvus_sift1m`, and run this command:

```bash
$ python3 milvus_search.py --table ann_1m_sq8h --nq 10 --topk 20 --nprobe 64 -s
```

> Note: nprobe affects search accuracy and performance. The greater the value, the higher the accuracy, but the lower the performance. In this experiment, you are recommended to use 32 for nprobe.

After running the command above, a `search_output` folder is created and includes `ann_1m_sq8h_32_output.txt`, which records top 20 similar vectors for the 10 vectors. In the text file, each 20 rows are formatted as one group that corresponds to the result of one query. The first column specifies the location of the vector to query in `query.npy`; the second column specifies the vectors correspond to the query result in `bvecs_data` (e.g. in 80006099349, the first `8` does not have a meaning, but `0006` after the first `8` stands for the 6th file in `bvecs_data`, the last `099349` indicates that the 099349th vector in the 6th file is the vector corresponding to the query result); the third column specifies the vector to query and the Euclidean distance.


#### Step 2: Run accuracy test script

Use the following command to compare the query result with ground truth and calculate the search accuracy of Milvus:

```bash
$ python3 milvus_compare.py --table ann_1m_sq8h --nprobe 64 -p
```

#### Step 3: Verify test results

When the test script is completed, a `compare` folder is generated and includes `64_ann_1m_sq8h_10_20_output.csv`.

- nq - the ordinal number of query vectors
- topk - the top k most similar result vectors for the query vectors 
- total_time - the total query elapsed time (in seconds)
- avg_time - the average time to query one vector (in seconds)
- recall - the accuracy calculated by comparing Milvus search results and ground truth

The accuracy rate has a positive correlation with search parameter `nprobe` (number of sub-spaces searched). In this test, when the `nprobe` = 64, the accuracy can reach > 90%.  However, as the `nprobe` gets bigger, the search time will be longer. 

Therefore, based on your data distribution and business scenario, you need to edit `nprobe` to optimize the trade-off between accuracy and search time. 

## 5. Performance test

To test search performance, go to directory *milvus_sift1m*, and run the following script: 

```bash
$ python3 milvus_toolkit.py --table=ann_1m_sq8h --nprobe 64 -s
```

When the execution is completed, a `performance` folder is generated and includes `ann_1m_sq8h_32_output.csv`, which includes the running time for topk values with different nq values.

- nq - the number of query vectors
- topk - the top k most similar vectors for the query vectors 
- total_time - the total query elapsed time (in seconds)
- avg_time - the average time to query one vector (in seconds)

**Note:**

> 1. In milvus_toolkit.py, `nq` is set to be 1, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800, respectively, and `topk` is set to be 1, 20, 50, 100, 300, 500, 800, 1000, respectively.
>
> 2. To run the first vector search, some extra time is needed to load the data (from the disk) to the memory.
>
> 3. It is recommended to run several performance tests continuously, and use the search time of the second run. If the tests are executed intermittently, Intel CPU may downgrade to base clock speed.
