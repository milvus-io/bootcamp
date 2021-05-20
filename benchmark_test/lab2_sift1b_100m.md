# Lab Test 2: 100-Million-Scale Vector Similarity Search

## 1. Prepare test data and scripts

The 100 million vectors used in this test are extracted from the dataset [SIFT1B](http://corpus-texmex.irisa.fr/).

#### Hardware requirements

| Component           | Minimum Config                |
| ------------------ | -------------------------- |
| OS            | Ubuntu LTS 18.04 |
| CPU           | Intel Core i7-8700        |
| GPU           | Nvidia GeForce GTX 1060, 6GB GDDR5 |
| GPU Driver    | CUDA 10.1, Driver 418.74 |
| Memory        | 16 GB DDR4 ( 2400 Mhz ) x 2                |
| Storage       | SATA 3.0 SSD 256 GB                  |

Download the following data and scripts, and save them to a file named **milvlus_sift100m**. 
- [100 million vector dataset](https://pan.baidu.com/s/1N5jGKHYTGchye3qR31aNnA)

  Download the dataset and extract the data to **milvus_sift100M/bvecs_data/**. When the extraction is completed, there will be 1000 .npy files, each containing 100,000 vectors, in file **bvecs_data**.

- [Query vector dataset](https://pan.baidu.com/s/1l9_lDItU2dPBPIYZ7oV0NQ)

- [Search ground truth](https://pan.baidu.com/s/1Raqs_1NkGMkPboENFlkPsw )     Extraction code  ：yvdr

- [Test scripts](/benchmark_test/scripts/)

When it is done, there should be the following files in **milvus_sift100m**:

1. The **bvecs_data** file containing 100 million vectors
2. The **query.npy** file that has 10,000 query vectors
3. The **ground_truth.txt** file with the top 1000 most similar results for each query vector
4. The test script files : `main.py`、`milvus_toolkit.py`、`milvus_load.py`、`config.py`。

> **Note:** Please go through the README carefully before testing with script . Make changes to the parameters in the script to match your scenario.

## 2. Configure Milvus parameters

To optimize Milvus's performance, you can change system parameters to suit your requirements. In this test, 90% recall rate can be achieved by using the recommended values in below table. 

Configuration file: **/home/$USER/milvus/conf/server_config.yaml**

|         Parameter         | Recommended value |
| ----------------------   | ---- |
| index_building_threshold |  1024  |
|    cpu_cache_capacity    |   25   |
|    use_blas_threshold    |  801   |
|          nprobe          |   32   |

Refer to [Milvus Configuration](https://milvus.io/docs/v1.1.0/milvus_config.md) for more information.

Use default values for other parameters. After setting parameter values, restart Milvus Docker to apply all changes.

```bash
$ docker restart <container id>
```

## 3. Create a table and build indexes

Make sure Milvus is already installed and started. (For details of Milvus installation, please read [Milvus Quick Start](https://milvus.io/docs/v1.0.0/milvus_docker-cpu.md)).

> Before testing, please modify the corresponding parameters according to the [script instructions](/benchmark_test/scripts/README.md)

Go to `milvus_sift1m`, and run the following command to create a table and build indexes:

```bash
$ python3 main.py --collection ann_100m_sq8 --dim 128 -c
$ python3 main.py --collectio ann_100m_sq8q8 --index sq8 --build 
```

Vectors are then inserted into a table named `ann_100m_sq8h`, with the index_type of `IVF_SQ8H`. 

To show the available tables and number of vectors in each table, use the following command:

```bash
#查看库中有哪些表
$ python3 main.py --show
#查看表ann_100m_sq8h的行数
$ python3 main.py --collectio ann_100m_sq8q8 --rows
ann_100m_sq8_sq8h的索引类型
$ python3 main.py --collection ann_100m_sq81m_sq8 --describe_index
```



## 4. Import data

Make sure table ann_100m_sq8 is successfully created.

> In this project, due to the large amount of data, the downloaded data sets are in uint8 format, so you need to modify the parameter IS_UINT8 in config.py to True before running

Run the following command to import 100m rows of data:

```bash
$ python3 main.py --collection=ann_100m_sq8 --load
```

You can see that all data is imported from the file for once.

Run the following command to check the number of rows in the table:

```bash
$ python3 main.py --collection=ann_100m_sq8 --rows
```

To make sure that all data imported to Milvus has indexes built. Navigate to `/home/$USER/milvus/db` and enter the following command:

```bash
$ sqlite3 meta.sqlite
```

In sqlite3 CLI, enter the following command to check the current status:

```sql
sqlite> select * from TableFiles where table_id='ann_100m_sq8h';
```

Milvus divides a vector table into shards for storage. So, a query returns multiple records. The third column specifies the index type and 5 stands for IVF_SQ8H. The fifth column specifies the build status of the index and 3 indicates that index building is complete for the shard. If index building is not complete for a specific shard, you can manually build indexes for the shard.

Exit sqlite CLI:

```sql
sqlite> .quit
```

Enter `milvus_sift1m` and run the following command:

```bash
$ python3 main.py --collection=ann_100m_sq8 --index=sq8 --build 
```

After manually building indexes, enter sqlite CLI again and make sure that index building has been completed for all shards. To understand the meanings of other columns, navigate to `/home/$USER/milvus/db` and enter the following command in the sqlite CLI:

```bash
$ sqlite3 meta.sqlite
sqlite>.schema
```

## 5. Accuracy test

SIFT1B provides not only the vector dataset to search 10,000 vectors, but also the top 1000 ground truth for each vector, which allows convenient calculation of precision rate. The vector search accuracy of Milvus can be represented as follows:

Accuracy = Number of shared vectors (between Milvus search results and Ground truth) / (query_records * top_k)

####  Run query script

Before the accuracy test, you need to manually create the directory `recall_result / recall_compare_out` to save the test results. To test the search precision for  top1(top10, top100, top200) results of 500 vectors randomly chosen from the 10,000 query vectors, go to directory `milvus_sift1m`, and run this command:

```bash
$ python3 main.py --collection=ann_100m_sq8 --search_param 128 --recall
```

> Note: search_param is nprobe value. nprobe affects search accuracy and performance. The greater the value, the higher the accuracy, but the lower the performance. In this experiment.

After executing the above command, an `ann_sift1m_sq8_128_500_recall.txt` text file will be generated in the` recall_result` folder. The text file records the id and distance of the most similar first 200 vectors corresponding to 500 vectors,Every 200 lines in the text file correspond to a query result of a query. At the same time, multiple texts will be generated under the `recall_compare_out` file. Taking ` ann_sift1m_sq8_128_500_100` as an example, this text records the respective corresponding accuracy rates and the total average accuracy rate of the 500 vectors queried when topk = 100.

The accuracy rate has a positive correlation with search parameter `nprobe` (number of sub-spaces searched). In this test, when the `nprobe` = 64, the accuracy can reach > 90%.  However, as the `nprobe` gets bigger, the search time will be longer. 

Therefore, based on your data distribution and business scenario, you need to edit `nprobe` to optimize the trade-off between accuracy and search time. 

## 6. Performance test

To test search performance, go to directory *milvus_sift1m*, and run the following script: 

```bash
$ python3 main.py --collection=ann_100m_sq8 --search_param 128 --performance
```

When the execution is completed, a `performance` folder is generated and includes `ann_100m_sq8h_32_output.csv`, which includes the running time for topk values with different nq values.

- nq - the number of query vectors
- topk - the top k most similar vectors for the query vectors 
- total_time - the total query elapsed time (in seconds)
- avg_time - the average time to query one vector (in seconds)

**Note:**

> 1. In milvus_toolkit.py, `nq` is set to be 1, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800, respectively, and `topk` is set to be 1, 20, 50, 100, 300, 500, 800, 1000, respectively.
> 2. To run the first vector search, some extra time is needed to load the data (from the disk) to the memory.
> 3. It is recommended to run several performance tests continuously, and use the search time of the second run. If the tests are executed intermittently, Intel CPU may downgrade to base clock speed.
