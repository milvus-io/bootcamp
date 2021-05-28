# README

## Preparation

Before running this project script, you need to start the service of milvus 1.1.

Install python package

```
pip install -r requirements.txt
```

## Scripts

| Parameter          | Description                                                  |
| ------------------ | ------------------------------------------------------------ |
| --collection       | Specify the name of the collection to be operated on         |
| --dim              | When creating a new collection, you need to specify the dimensions of the vectors in the collection |
| --index            | When creating an index, you need to specify the index type<flat, ivf_flat, sq8, sq8h, pq, nsg, hnsw> |
| --search_param     | When querying, specify the parameter value when querying (When the index is of type Ivf, this parameter refers to Nprobe. When indexing by Rnsg, this parameter refers to Search_Length. When the index is HNSW, this parameter refers to EF) |
| --partition_tag    | Specify the partition label:                                 |
| --create           | Perform the operation of creating a collection. This operation needs to specify two parameters: Collection and Dim |
| --load             | Perform the operation of writing data. This operation needs to specify the parameter Collection |
| --build            | Perform indexing operations. This operation needs to specify the parameters Collection and Index |
| --performance      | Perform performance testing operations. This operation needs to specify the parameters Collection and Search_param |
| --recall           | Perform recall test operations. This operation needs to specify the parameters Collection and Search_param |
| --create_partition | Perform the operation of creating a partition. This operation needs to specify the parameters Collection and Partition |
| --info             | View the data information of a certain collection. This operation needs to specify the parameter Collection |
| --describe         | View the basic information of a collection. This operation needs to specify the parameter Collection |
| --show             | Display the collections that exist in the library. No other parameters are required for this operation |
| --has              | Determine whether a collection exists. This operation needs to specify the parameter Collection |
| --rows             | View the number of vectors in a collection. This operation needs to specify the parameter Collection |
| --flush            | Manual data placement operation. This operation needs to specify the parameter Collection |
| --drop             | Delete the specified collection. This operation needs to specify the parameter Collection |
| --drop_index       | Delete the index of the specified collection. This operation needs to specify the parameter Collection |
| --version          | Check the version of milvus server and pymilvus. No other parameters are required for this operation |

## Configuration File

| Parameter   | Description                               | Defaults  |
| ----------- | ----------------------------------------- | --------- |
| MILVUS_HOST | The IP where the Milvus server is located | 127.0.0.1 |
| MILVUS_PORT | Port provided by Milvus server            | 19530     |

**Parameters required when creating a collection**：

| Parameter       | Description                                                  | Defaults      |
| --------------- | ------------------------------------------------------------ | ------------- |
| INDEX_FILE_SIZE | Specify the size of each segment.                            | 2048          |
| METRIC_TYPE     | Specify the vector similarity calculation method when creating a collection | MetricType.L2 |

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

```
python main.py --collection <collection_name> -c
```

**2. Create an index**

```
python main.py --collection <collection_name> --index <index_type> --build
```

**3. Data load**

```
python main.py --collection <collection_name> --load
```

**4. Performance** **Test**

```
python main.py --collection <collection_name> --search_param <search_param> --performance
```

**5. Recall test**

```
python main.py --collection <collection_name> --search_param <search_param> --recall
```

**6.Create partition**

```
python main.py --collection <collection_name> --partition_tag --create_partition
```

**7. View collection information**

```
python main.py --collection <collection_name> --describe
```

**8.View collections**

```
python main.py --show
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

**13.View** **milvus** **server and** **pymilvus** **version**

```
python main.py --version
```