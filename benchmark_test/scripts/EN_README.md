# EN_README

## premise

Before starting this project, you need to install milvus0.1.0.

Install python packages

```shell
pip install -r requirements.txt
```

## Script description

| Parameter               | Description                                                         |
| ------------------ | ------------------------------------------------------------ |
| --collection       | Specify collection name                                       |
| --dim              | Specify vector size                           |
| --index            | Specify index type <flat, ivf_flat, sq8, sq8h, pq, nsg, hnsw> |
| --search_param     | Specify query parameters. |
| --partition_tag    | Specify partition label                                                 |
| --create           | Perform the operation of creating a collection. This operation needs to specify the collection and dim parameters. |
| --load             | Perform the operation of writing data. This operation needs to specify the parameter collection.             |
| --build            | Perform indexing operations. This operation needs to specify the parameters collection and index.      |
| --performance      | Perform performance testing operations. This operation needs to specify the parameters collection and search_param. |
| --recall           | Perform recall test operations. This operation needs to specify the parameters collection and search_param. |
| --create_partition | Perform the operation of creating a partition. This operation needs to specify the parameters collection and partition.  |
| --info             | View the data information of a certain collection. This operation needs to specify the parameter collection.         |
| --describe         | View the basic information of a collection. This operation needs to specify the parameter collection.         |
| --show             | Show the collections that exist in Milvus. No other parameters are required for this operation.                      |
| --has              | Determine whether a collection exists. This operation needs to specify the parameter collection.           |
| --rows             | View the number of vectors in a collection. This operation needs to specify the parameter collection.         |
| --describe_index   | Display index information for a collection. This operation needs to specify the parameter collection.         |
| --flush            | Manual data placement operation. This operation needs to specify the parameter collection.               |
| --drop             | Delete the specified collection. This operation needs to specify the parameter collection.                   |
| --drop_index       | Delete the index of the specified collection. This operation needs to specify the parameter collection.             |
| --version          | Check the version of milvus server and pymilvus. No other parameters are required for this operation.      |



## Configuration File

| Parameter        | Description                    | Defaults    |
| ----------- | ----------------------- | --------- |
| MILVUS_HOST | ilvus server ip   | 127.0.0.1 |
| MILVUS_PORT | Milvus server port | 19530     |

lParameters required when creating a collection：

| Parameter            | Description                             | Defaults        |
| --------------- | -------------------------------- | ------------- |
| INDEX_FILE_SIZE | The data file size specified when the collection was created.     | 2048          |
| METRIC_TYPE     | Specify the vector similarity calculation method when creating a collection. | MetricType.L2 |

Parameters required for indexing：

| Parameter           | Description                          | Defaults |
| -------------- | ----------------------------- | ------ |
| NLIST          | Parameter `nlist` when creating ivf series index            | 16384  |
| PQ_M           | Parameter `m` when creating ivf_pq index               | 12     |
| SEARCH_LENGTH  | Parameter `SEARCH_LENGTH` when creating RNSG index  | 45     |
| OUT_DEGREE     | Parameter `OUT_DEGREE` when creating RNSG index     | 50     |
| CANDIDATE_POOL | Parameter `CANDIDATE_POOL` when creating RNSG index | 300    |
| KNNG           | Parameter `KNNG` when creating RNSG index           | 100    |
| HNSW_M         | Parameter `M` when creating HNSW index               | 16     |
| EFCONSTRUCTION | Parameter `EFCONSTRUCTION` when creating HNSW index  | 500    |

Configuration parameters required when writing data：

| Parameter            | Description                                                 | Defaults  |
| --------------- | ---------------------------------------------------- | ------- |
| FILE_TYPE       | File format for writing data<npy,csv,bvecs,fvecs>              | bvecs   |
| FILE_NPY_PATH   | When the data format is npy, the path of the directory where the file is located            | ' '     |
| FILE_CSV_PATH   | When the data format is csv, the path of the directory where the file is located	            | ' '     |
| FILE_FVECS_PATH | When the data format is fvecs, the path where the file is located              | ' '     |
| FILE_BVECS_PATH | When the data format is bvecs, the path where the file is located              | ' '     |
| VECS_VEC_NUM    | When the data format is bvecs or fvecs, the amount of data to be written            | 1000000 |
| VECS_BASE_LEN   | When the data format is bvecs or fvecs, the amount of data written into milvus each time | 500000  |
| if_normaliz     | Do you need to normalize the data before importing it                     | False   |

Parameters required for performance test

| Parameter                  | Description                                             | Defaults             |
| --------------------- | ------------------------------------------------ | ------------------ |
| NQ_FOLDER_NAME        | The directory where the vector to be queried is located                   | None                |
| PERFORMANCE_FILE_NAME | Performance results will be saved in this folder                       | 'performance '     |
| nq_scope              | The nq value to be tested (Here means testing multiple nq values)             | [1,10,100,200]     |
| topk_scope            | The topk value to be tested in each np (Here means testing multiple topk values) | [1,1, 10, 100,500] |
| IS_CSV                | Whether the vector to be queried exists in a csv format file                | False              |
| IS_UINT8              | Whether the vector to be queried is the value of uint8                      | False              |

Parameters required for recall test：

| Parameter                | Description                                                | Defaults           |
| ------------------- | --------------------------------------------------- | ---------------- |
| recall_topk         | Topk value queried when testing recall                            | 200              |
| compute_recall_topk | Multiple topk values to be calculated when calculating the recall rate, less than or equal to recall_topk | [1, 10, 100,200] |
| recall_nq           | The average recall of nq vectors that need to be calculated when testing the recall          | 500              |
| recall_vec_fname    | The path of the file where the vector to be queried is located when testing the recall rate              | recall_vec_fname |
| GT_FNAME_NAME       | Standard result set for comparison with test results                          | GT_FNAME_NAME    |
| recall_res_fname    | The test results are saved in this directory                              | recall_res_fname |
| recall_out_fname    | The recall rate calculation result exists under this path                          | recall_out_fname |



## Instructions

1. Create a Collection

```shell
python main.py --collection <collection_name> -c
```

2. Create an index

```shell
python main.py --collection <collection_name> --index <index_type> --build
```

3. Data load

```shell
python main.py --collection <collection_name> --load
```

4. Performance Test

```shell
python main.py --collection <collection_name> --search_param <search_param> --performance
```

5. Recall test

```shell
python main.py --collection <collection_name> --search_param <search_param> --recall
```

6. Create partition

```shell
python main.py --collection <collection_name> --partition_tag --create_partition
```

7. View collection information

```shell
python main.py --collection <collection_name> --describe
```

8. View collections

```shell
python main.py --show
```

9. Determine whether the collection exists

```shell
python main.py --collection <collection_name> --has
```

10. View the number of vectors in the collection

```shell
python main.py --collection <collection_name> --rows
```

11. View the index type of the collection

```shell
python main.py --collection <collection_name> --describe_index
```

12. Delete the collection

```shell
python main.py --collection <collection_name> --drop
```

13. Delete the index

```shell
python main.py --collection <collection_name> --drop_index
```

14. View milvus server and pymilvus version

```shell
python main.py --version
```


