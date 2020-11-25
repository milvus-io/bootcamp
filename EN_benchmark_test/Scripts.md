# README

## Preparation

Before running this project script, you need to start the service of milvus 0.11.

Install python package

```shell
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
| MILVUS_PORT | Milvus server提供的端口                   | 19530     |

l**Parameters required when creating a collection**：

| Parameter         | Description                                                  | Defaults      |
| ----------------- | ------------------------------------------------------------ | ------------- |
| SEGMENT_ROW_LIMIT | The number of rows to store entities when creating a collection | 4096          |
| METRIC_TYPE       | Specify the vector similarity calculation method when creating a collection | MetricType.L2 |
| AUTO_ID           | Whether to automatically generate ID                         | True          |

l**Parameters required for indexing**：

| Parameter      | Description                              | Defaults |
| -------------- | ---------------------------------------- | -------- |
| NLIST          | Nlist value when indexing                | 16384    |
| PQ_M           | M value when indexing PQ                 | 12       |
| SEARCH_LENGTH  | SEARCH_LENGTH value when indexing NSG    | 45       |
| OUT_DEGREE     | OUT_DEGREE value when building index NSG | 50       |
| CANDIDATE_POOL | CANDIDATE_POOL value when indexing NSG   | 300      |
| KNNG           | KNNG value when indexing NSG             | 100      |
| HNSW_M         | M value for indexing HNSW                | 16       |
| EFCONSTRUCTION | Build the EFCONSTRUCTION value of HNSW   | 500      |

l**Configuration parameters required when writing data**：

| Parameter       | Description                                                  | Defaults |
| --------------- | ------------------------------------------------------------ | -------- |
| FILE_TYPE       | File format for writing data<npy,csv,bvecs,fvecs>            | bvecs    |
| FILE_NPY_PATH   | When the data format is npy, the path of the directory where the file is located | ' '      |
| FILE_CSV_PATH   | When the data format is csv, the path of the directory where the file is located | ' '      |
| FILE_FVECS_PATH | When the data format is fvecs, the path where the file is located | ' '      |
| FILE_BVECS_PATH | When the data format is bvecs, the path where the file is located | ' '      |
| VECS_VEC_NUM    | When the data format is bvecs or fvecs, the amount of data to be written | 1000000  |
| VECS_BASE_LEN   | When the data format is bvecs or fvecs, the amount of data written into milvus each time | 500000   |
| if_normaliz     | Do you need to normalize the data before importing it        | False    |

l**Parameters required for performance** **test**

| Parameter             | Description                                                  | Defaults           |
| --------------------- | ------------------------------------------------------------ | ------------------ |
| NQ_FOLDER_NAME        | The directory where the vector to be queried is located      | ' '                |
| PERFORMANCE_FILE_NAME | Performance results will be saved in this folder             | 'performance '     |
| nq_scope              | The nq value to be tested (Here means testing multiple nq values) | [1,10,100,200]     |
| topk_scope            | The topk value to be tested in each np (Here means testing multiple topk values) | [1,1, 10, 100,500] |
| IS_CSV                | Whether the vector to be queried exists in a csv format file | False              |
| IS_UINT8              | Whether the vector to be queried is the value of uint8       | False              |

l**Parameters required for recall test**：

| Parameter           | Description                                                  | Defaults         |
| ------------------- | ------------------------------------------------------------ | ---------------- |
| recall_topk         | Topk value queried when testing recall                       | 200              |
| compute_recall_topk | Multiple topk values to be calculated when calculating the recall rate, less than or equal to recall_topk | [1, 10, 100,200] |
| recall_nq           | The average recall of nq vectors that need to be calculated when testing the recall | 500              |
| recall_vec_fname    | The path of the file where the vector to be queried is located when testing the recall rate | recall_vec_fname |
| GT_FNAME_NAME       | Standard result set for comparison with test results         | GT_FNAME_NAME    |
| recall_res_fname    | The test results are saved in this directory                 | recall_res_fname |
| recall_out_fname    | The recall rate calculation result exists under this path    | recall_out_fname |

## Instructions

**1.Create a Collection**

```shell
python main.py --collection <collection_name> -c
```

**2. Create an index**

```shell
python main.py --collection <collection_name> --index <index_type> --build
```

**3. Data load**

```shell
python main.py --collection <collection_name> --load
```

**4. Performance** **Test**

```shell
python main.py --collection <collection_name> --search_param <search_param> --performance
```

**5. Recall test**

```shell
python main.py --collection <collection_name> --search_param <search_param> --recall
```

**6.Create partition**

```shell
python main.py --collection <collection_name> --partition_tag --create_partition
```

**7. View collection information**

```shell
python main.py --collection <collection_name> --describe
```

**8.View collections** 

```shell
python main.py --show
```

**9.Determine whether the collection exists**

```shell
python main.py --collection <collection_name> --has
```

**10.View the number of vectors in the collection**

```shell
python main.py --collection <collection_name> --rows
```

**11.Delete the collection**

```shell
python main.py --collection <collection_name> --drop
```

**12.Delete the index**

```shell
python main.py --collection <collection_name> --drop_index
```

**13.View** **milvus** **server and** **pymilvus** **version**

```shell
python main.py --version
```

