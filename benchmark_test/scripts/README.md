# README

## Preparation
This project is the benchmark test based Milvus 2.0.0-rc5.

Before running this project script, you need to start the service of milvus 2.0.

Install python package

```
pip install -r requirements.txt
```

## Scripts

| Parameter          | Description                                                  |
| ------------------ | ------------------------------------------------------------ |
| --collection       | Specify the name of the collection to be operated on         |
|                    |                                                              |
| --index_type       | When creating an index, you need to specify the index type<FLAT, IVF_FLAT, IVF_SQ8, IVF_SQ8 , IVF_PQ, RNSG, HNSW, ANNOY> |
| --search_param     | When querying, specify the parameter value when querying (When the index is of type Ivf, this parameter refers to Nprobe. When indexing by Rnsg, this parameter refers to Search_Length. When the index is HNSW, this parameter refers to EF) |
| --create           | Perform the operation of creating a collection. This operation needs to specify two parameters: Collection and Dim |
| --insert           | Perform the operation of writing data. This operation needs to specify the parameter Collection |
| --create_index     | Perform indexing operations. This operation needs to specify the parameters Collection and Index |
| --performance      | Perform performance testing operations. This operation needs to specify the parameters Collection and Search_param |
| --percentile_test  | Testing the performance of Milvus p99  |
| --recall           | Perform recall test operations. This operation needs to specify the parameters Collection and Search_param |
| --partition_name   | Specify the partition label                                  |
| --create_partition | Perform the operation of creating a partition. This operation needs to specify the parameters Collection and Partition |
| --index_info       | View the index information of a certain collection. This operation needs to specify the parameter Collection |
| --has              | Determine whether a collection exists. This operation needs to specify the parameter Collection |
| --rows             | View the number of vectors in a collection. This operation needs to specify the parameter Collection |
| --drop             | Delete the specified collection. This operation needs to specify the parameter Collection |
| --drop_index       | Delete the index of the specified collection. This operation needs to specify the parameter Collection |
| --load             | Load the specified collection data to memory |
| --list             | List all collections.|
| --release          | Release the specified collection data from memory|


## Configuration File

| Parameter   | Description                               | Defaults  |
| ----------- | ----------------------------------------- | --------- |
| MILVUS_HOST | The IP where the Milvus server is located | 127.0.0.1 |
| MILVUS_PORT | Port provided by Milvus server            | 19530     |

**Parameters required when creating a collection**：

| Parameter        | Description                                                  | Defaults |
| ---------------- | ------------------------------------------------------------ | -------- |
| METRIC_TYPE      | Specify the vector similarity calculation method when creating a collection | L2       |
| VECTOR_DIMENSION | Specify the vector dimension when creating a collection      | 128      |

**Parameters required for indexing**：

| Parameter      | Description                              | Defaults |
| -------------- | ---------------------------------------- | -------- |
| NLIST          | Nlist value when indexing                | 2000     |
| PQ_M           | M value when indexing PQ                 | 12       |
| SEARCH_LENGTH  | SEARCH_LENGTH value when indexing NSG    | 45       |
| OUT_DEGREE     | OUT_DEGREE value when building index NSG | 50       |
| CANDIDATE_POOL | CANDIDATE_POOL value when indexing NSG   | 300      |
| KNNG           | KNNG value when indexing NSG             | 100      |
| HNSW_M         | M value for indexing HNSW                | 16       |
| EFCONSTRUCTION | Build the EFCONSTRUCTION value of HNSW   | 500      |
| N_TREE         | N_TREE value when indexing ANNOY         | 8        |

**Configuration parameters required when writing data**：

| Parameter          | Description                                                  | Defaults |
| ------------------ | ------------------------------------------------------------ | -------- |
| FILE_TYPE          | File format for writing data<npy,csv,bvecs,fvecs>            | Npy      |
| BASE_FILE_PATH     | The directory of data to be loaded in milvus                 | ''       |
| IF_NORMALIZE       | Does the data need to be normalized before insertion         | False    |
| TOTAL_VECTOR_COUNT | When the data format is bvecs or fvecs, the amount of data to be written | 20000    |
| IMPORT_CHUNK_SIZE  | When the data format is bvecs or fvecs, the amount of data written into milvus each time(<=256MB) | 20000    |

**Parameters required for performance** **test**

| Parameter                | Description                                                  | Defaults            |
| ------------------------ | ------------------------------------------------------------ | ------------------- |
| QUERY_FILE_PATH          | The directory where the vector to be queried is located      | ''                  |
| PERFORMANCE_RESULTS_PATH | Performance results will be saved in this folder             | 'performance '      |
| NQ_SCOPE                 | The nq value to be tested (Here means testing multiple nq values) | [1,10,100,500,1000] |
| TOPK_SCOPE               | The topk value to be tested in each np (Here means testing multiple topk values) | [1,1, 10, 100,500]  |

**Parameters required for recall test**：

| Parameter         | Description                                                  | Defaults                           |
| ----------------- | ------------------------------------------------------------ | ---------------------------------- |
| RECALL_NQ         | The average recall results of the nq vectors to be calculated when testing recall | 500                                |
| RECALL_TOPK       | Topk value queried when testing recall                       | 500                                |
| RECALL_CALC_SCOPE | Multiple topk values to be calculated when calculating the recall rate, less than or equal to RECALL_TOPK | [1, 10, 100,500]                   |
| RECALL_QUERY_FILE | The path of the file where the vector to be queried is located when testing the recall rate | ''                                 |
| IS_CSV            | Whether the vector to be queried exists in a csv format file | False                              |
| IS_UINT8          | Whether the vector to be queried is the value of uint8       | False                              |
| GROUNDTRUTH_FILE  | Standard result set for comparison with test results         | ''                                 |
| RECALL_RES  | The recall results are saved in this directory               | 'recall_result'                    |
| RECALL_RES_TOPK  | The average recall rate for each topk are saved in this directory | 'recall_result/recall_compare_out' |

## Instructions

**1.Create a Collection**

```shell
python main.py --collection <collection_name> -c
```

**2. Create an index**

```shell
python main.py --collection <collection_name> --index_type <index_type> --create_index
```

**3. Data insert**

```
python main.py --collection <collection_name> --insert
```

**4. Load data to memory**
```
python main.py --collection <collection_name> --load
```

**5. Performance Test**

```
python main.py --collection <collection_name> --search_param <search_param> --performance
```

**6. Recall test**

```
python main.py --collection <collection_name> --search_param <search_param> --recall
```

**7.Create partition**

```
python main.py --collection <collection_name> --partition_name --create_partition
```

**8. View collection index information**

```
python main.py --collection <collection_name> --index_info
```

**9.Determine whether the collection exists**

```
python main.py --collection <collection_name> --has
```

**10.View the number of vectors in the collection**

```
python main.py --collection <collection_name> --rows
```

**11.Delete the collection**

```
python main.py --collection <collection_name> --drop
```

**12.Delete the index**

```
python main.py --collection <collection_name> --drop_index
```

**13. List collection**

```
python main.py --list
```

**14. p99 performance test**

```
python main.py --collection <collection_name> --search_param <search_param> --percentile 99 --percentile_test
```
