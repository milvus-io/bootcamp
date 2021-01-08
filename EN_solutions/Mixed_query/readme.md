# Milvus -Based Mixed Query Solution

This solution is an example of hybrid query combined with the vector database Milvus.

In the following example, feature vectors and structured data are used to simulate face attributes, showing how to perform a mixed query of structured data and unstructured data. In the example, for a given vector (can be regarded as a given face picture), and its attributes (gender, time, whether to wear glasses), combine Milvus to query the top ten most similar vectors and their European styles distance.

## Preparations

1. [Install Milvus 0.11](https://www.milvus.io/cn/docs/v0.11.0/milvus_docker-gpu.md)
3. pip install pymilvus==0.3.0
4. pip install numpy

## Data source

The data used in this test is ANN_SIFT1B <http://corpus-texmex.irisa.fr/>

- Base vectors: ANN_SIFT1B Base_set
- Query vectors: ANN_SIFIT1B Query_set

> Note: You can also use data in `bvecs` format. 

## Test scripts

The following test scripts are used in this example:

- `mixed_import.py` for importing data into Milvus 
- `mixed_query.py` for executing customized hybrid search.

### mixed_import.py

Before executing the script, you need to look at some of the variables in the script and make changes based on the runtime environment and data to ensure that the code is running correctly.

##### Parameters

| Parameter      | Description                                                  |
| -------------- | ------------------------------------------------------------ |
| `MILVUS_TABLE` | Name of the table to create in Milvus.                       |
| `FILE_PATH`    | Path of local storage of base vectors.                       |
| `VEC_NUM`      | Total number of vectors to import into Milvus.               |
| `BASE_LEN`     | Number of vectors batch imported into the table.             |
| `VEC_DIM`      | Dimension set in the table in Milvus. It should be set to the dimension of the data to be imported |
| `SERVER_ADDR`  | Address of Milvus server.                                    |
| `SERVER_PORT`  | Port of Milvus server.                                       |

##### Execute the script

When you have completed configuring the above parameter, you can import data by below command:

```shell
python3 mixed_import.py
```

In this script, vector and vector genera (including gender, time, glasses or not) are stored in the Milvus vector search engine

### mixed_query.py

Before searching vectors, edit the following parameters in the script to match your runtime environment. 

##### Parameters

| Parameter            | Description                                                  |
| -------------------- | ------------------------------------------------------------ |
| `QUERY_PATH`         | Path for the local storage of query vectors.                 |
| `MILVUS_TABLE`       | Name of the table to create in Milvus. Use the same table name set for Milvus in `mixed_import.py`. |
| `SERVER_ADDR`        | Address of Milvus server.                                    |
| `SERVER_PORT`        | Port of Milvus server.                                       |
| `TOP_K`              | The top k most similar result vectors.                       |
| `DISTANCE_THRESHOLD` | Threshold to filter the top k result vectors. Default value is 1. Vectors with a Euclidean distances smaller than this threshold will be selected out. |

##### Variables

| Variable           | Description                                                  |
| ------------------ | ------------------------------------------------------------ |
| `-n` / `--num`     | Defines the ordinal rank of the query vector in the vector base set. |
| `-s` / `--sex`     | Define the gender of the human face: 0 means male and 1 means famale |
| `-t` / `--time`    | Specifies the query time range, e.g. 2018                    |
| `-g` / `--glasses` | Defines if the human face wears glasses: 10 means glasses, or  11 means no glasses |
| `-q` / `--query`   | Starts the query execution.                                  |
| `-v` / `--vector`  | The vectors corresponding with the ids entered.              |



##### Run the example

Query the vector similar to the 0 vector in the vector set, and the gender is male, and the date is 2018:

```shell
python3 mixed_query.py -n 0 -s 0 -t 2018, -q
```

Query the vector similar to the 20th vector in the vector set, and the gender is female, without glasses:

```shell
python3 mixed_query.py -n 20 -s 1 -g 11
```

Query the vector similar to the 100th vector in the vector set, and the gender is female, wearing glasses, and the date is 2019:

```shell
python3 mixed_query.py -n 100 -s 1 -g 10 -t 2019  -q
```

The original vector corresponding to the ID obtained by the query:

```shell
python3 mixed_query.py -v 4011
```

This solution shows an example of a milvus-based hybrid query, which can also be mixed with a relational database for various solution.

