# Milvus-Based Hybrid Search of Vectors and Structured Data

This solution provides an example of consolidating Milvus (vector database) and PostgreSQL (relational database) to carry out the hybrid search of vectors and structured data.

In below example, feature vectors and structured data are used to represent human face attributes. Here is how hybrid search works out: First, search the top 10 most similar vectors (and their Euclidean distances) of a defined vector (could be a specified human face image). Then by comparing the Euclidean distance, find out (among the top 10  result vectors) vectors which have Euclidean distance < 1, and which at the same time meet the filtering conditions (gender, time, and if with glasses) in PostgreSQL. 

## Prerequisites:

Before executing the hybrid search, make sure you have completed the following steps:

1. [Install Milvus](https://github.com/milvus-io/docs/blob/master/install_milvus.md)
2. [Install PostgreSQL](https://www.postgresql.org/download/)
3. Use `pip install numpy` to download numpy.
4. Use `pip install psycopg2` to download psycopg2.
5. Use `pip install faker` to download Faker. 

## Data source

The data used in this test are from [ANN_SIFT1B](http://corpus-texmex.irisa.fr/) .

- Base vectors: ANN_SIFT1B Base_set
- Query vectors: ANN_SIFIT1B Query_set

> Note: You can also use data in `bvecs` format. 

## Test scripts

The following test scripts are used in this example:

- [mixed_import.py](https://github.com/milvus-io/bootcamp/blob/master/EN_solutions/hybrid_search/mixed_import.py) for importing data into Milvus and PostgreSQL.
- [mixed_query.py](https://github.com/milvus-io/bootcamp/blob/master/EN_solutions/hybrid_search/mixed_query.py) for executing customized hybrid search.

### mixed_import.py

Before executing this script, edit the following parameters in the script to match your runtime environment and data. 

##### Parameters

| Parameter | Description |
| --- | --- |
| `MILVUS_TABLE` |Name of the table to create in Milvus.|
| `PG_TABLE_NAME` |Name of the table to create in PostgreSQL.|
| `FILE_PATH` |Path of local storage of base vectors.|
| `VEC_NUM` |Total number of vectors to import into Milvus.|
| `BASE_LEN` |Number of vectors batch imported into the table.|
| `VEC_DIM` |Dimension set in the table in Milvus. It should be set to the dimension of the data to be imported|
| `SERVER_ADDR` |Address of Milvus server.|
| `SERVER_PORT` |Port of Milvus server.|
| `PG_HOST` |Address of PostgreSQL server.|
| `PG_PORT` |Port of PostgreSQL server.|
| `PG_USER` |Username to use in PostgreSQL.|
| `PG_PASSWORD` |Password to use in PostgreSQL.|
| `PG_DATABASE` |Database to use in PostgreSQL. |

##### Execute the script

When you have completed configuring the above parameter, you can import data by below command:

```shell
python3 mixed_import.py
```

After the execution, not only initial vectors are imported into Milvus, corresponding vector ids and vector attributes (such as gender, time the vector is generated, and if the human face wears glasses) are at the same time stored in PostgreSQL database. 

### mixed_query.py

Before searching vectors, edit the following parameters in the script to match your runtime environment. 

##### Parameters

| Parameter | Description |
| --- | --- |
|`QUERY_PATH` |Path for the local storage of query vectors.|
|`MILVUS_TABLE` |Name of the table to create in Milvus. Use the same table name set for Milvus in `mixed_import.py`.|
|`PG_TABLE_NAME` |Name of the table to create in PostgreSQL. Use the same table name set for PostgreSQL in `mixed_import.py`.|
|`SERVER_ADDR` |Address of Milvus server.|
|`SERVER_PORT` |Port of Milvus server.|
|`PG_HOST` |Address of PostgreSQL server.|
|`PG_PORT` |Port of PostgreSQL server.|
|`PG_USER` |Username to use in PostgreSQL.|
|`PG_PASSWORD` |Password to use in PostgreSQL.|
|`PG_DATABASE` |Database to use in PostgreSQL.|
|`TOP_K` |The top k most similar result vectors.|
|`DISTANCE_THRESHOLD` |Threshold to filter the top k result vectors. Default value is 1. Vectors with a Euclidean distances smaller than this threshold will be selected out.|


##### Variables

| Variable       | Description                                                  |
| -------------- | ------------------------------------------------------------ |
| `-n` / `--num`     | Defines the ordinal rank of the query vector in the vector base set. |
| `-s` / `--sex`     | Define the gender of the human face: `male` or `female`.     |
| `-t` / `--time`    | Specifies the query time range, e.g. `[2019-04-05 00:10:21, 2019-05-20 10:54:12]` |
| `-g` / `--glasses` | Defines if the human face wears glasses: `True` or `False`.  |
| `-q` / `--query`   | Starts the query execution.                                  |
| `-v` / `--vector`  | The vectors corresponding with the ids entered.              |

To search the top k most similar vectors of the vector which ranks `0` in the query vector set, meanwhile, the result vectors must match conditions that the gender is `male`, and that the vectors were generated during the time range of `[2019-05-01,  2019-07-12]`:

```shell
python3 mixed_query.py -n 0 -s male -t '[2019-05-01 00:00:00, 2019-07-12 00:00:00]' -q
```

To search the top k most similar vectors of the 20th vector in the query vector set, meanwhile, the result vectors must match conditions that the gender is `female` who wears no glasses:

```shell
python3 mixed_query.py -n 20 -s female -g False
```

To search the top k most similar vectors of the 100th vector in the query vector set, with gender `female` who wears glasses, and during the time range of `[2019-05-01 15:15:05, 2019-07-30 11:00:00]`:

```shell
python3 mixed_query.py -n 100 -s female -g True -t '[2019-05-01 15:15:05, 2019-07-30 11:00:00]' -q
```

To search the vector based on the vector id:

```shell
python3 mixed_query.py -v 237434787867
```

In conclusion, this solution demonstrates an example of hybrid search of structured and unstructured data using Milvus and PostgreSQL. Milvus supports easy integration with other relational databases to achieve hybrid search to match various scenarios.

