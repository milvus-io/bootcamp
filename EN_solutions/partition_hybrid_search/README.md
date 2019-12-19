# Hybrid search of vectors and structured data based on Milvus Partitions

This solution implements hybrid search of vectors and structured data based on Milvus Partitions.

The following example uses feature vectors and structured data to simulate human faces. For a specific vector (can be regarded as a specific feature vector retrieved from a human face) and corresponding properties, this example moves this vector into a specific partition based on its properties. Vectors in a partition share the same properties. During a query, this example searches for top 10 similar vectors in the corresponding partition based on specified properties (time when the face image is acquired, gender, with/without glasses).

## Prerequisite

- [Install Milvus](https://github.com/milvus-io/docs/blob/0.6.0/zh-CN/userguide/install_milvus.md)
- pip3 install numpy
- pip3 install faker

## Data source

This example uses the ANN_SIFT1B dataset.

- Download location: [http://corpus-texmex.irisa.fr/](http://corpus-texmex.irisa.fr/)
- Base dataset: ANN_SIFT1B Base_set
- Query dataset: ANN_SIFIT1B Query_set

> Note: You can also use other data files with `.bvecs` format.

## How to use the scripts

This example contains two scripts: `partition_import.py` and `partition_query.py`. `partition_import.py` defines the data properties and imports data to Milvus in partitions. `partition_query.py` performs hybrid search based on customizable conditions.

### partition_import.py

The raw data contains feature vectors from 1,000,000 face images. This example code randomly generates properties (time when the image is acquired, gender, with/without glasses) for these vectors. Every 100,000 vectors have the same properties. Thus, there will be 10 partitions in the table imported to Milvus. Each partition contains 100,000 vectors. (Each partition can contain different numbers of vectors. This is just a simple example.) Each partition contains its corresponding partition name and partition tag. A partition tag is a string composed of properties of the corresponding partition.

Before running this script, you need to check the variables and change their value per your environment and data to ensure that the script runs correctly.

**Variable description**

| Variable         | Description                                           |
| -------------- | ---------------------------------------------- |
| `MILVUS_TABLE` | Table name created in Milvus.              |
| `FILE_PATH`    | Path of the dataset to be imported to Milvus.            |
| `VEC_NUM`      | Number of vectors in the table. Should be smaller than the sum of the local dataset.  |
| `BASE_LEN`     | Number of vectors to be imported to the table each time. Equals to the number of vectors in a partition.  |
| `VEC_DIM`      | Dimension of the table. You must set this value per the dimension of imported vectors.|
| `SERVER_ADDR`  | Address of Milvus server.                   |
| `SERVER_PORT`  | Port of Milvus server.                       |

**Run**

```shell
$ python3 partition_import.py
```

![import](pic/import.PNG)

As is displayed in the previous screenshot, 1,000,000 vectors that are imported to Milvus are divided into 10 partitions. `partition_tag` specifies strings such as '2019-11-20'. `partition0` to `partition9` specify the `partition_name` of the corresponding partition. (`partition_tag` and `partition_name` are both self-defined.)

### partition_query.py

**Variable description**

| Variable         | Description                                                         |
| -------------- | ------------------------------------------------------------ |
| `MILVUS_TABLE` | Name of the table in Milvus to be queried. Must be the same as the table name created in `partition_import.py`. |
| `QUERY_PATH`   | Location of the dataset to be queried.                               |
| `SERVER_ADDR`  | Location of the Milvus server.                                    |
| `SERVER_PORT`  | Port of the Milvus server.                                        |
| `TOP_K`        | Number of vectors that are most similar to the raw data.               |

**Variables**

| Parameter |             | Description                                      |
| ---- | ----------- | ----------------------------------------- |
| `-n` | `--num`     | Sequence of the vector to query in the query dataset.   |
| `-s` | `--sex`     | Gender of the face: `male` or `female`.|
| `-t` | `--time`    | Time when the image is acquired, such as `2019-04-05`. |
| `-g` | `--glasses` | Whether the face is with glasses: `True` or `False`.|
| `-q` | `--query`   | Runs the query. No value needed for this parameter.                       |

**Example**

The following command queries top 10 vectors that are most similar to the 0th vector in the dataset. The gender is male and there are no glasses. The image is acquired on 2019-11-24:

```shell
$ python3 partition_query.py -n 0 -s male -g False -t 2019-11-24 -q
```

![search](pic/search.PNG)

In the previous query result, `id` specifies the sequence of the order to be inserted to the table. `distance` specifies the Euclidean distance between the vector to query and the queried vector.

The following command queries top 10 vectors that are most similar to the 10th vector in the dataset. The gender is male and the image is acquired on 2019-12-07:

```shell
$ python3 partition_query.py -n 10 -s male -t 2019-12-07 -q
```

The following command queries top 10 vectors that are most similar to the 50th vector in the dataset. There are no glasses and the image is acquired on 2019-11-29:

```shell
$ python3 partition_query.py -n 50 -g False -t 2019-11-29 -q
```

The following command queries top 10 vectors that are most similar to the 306th vector in the dataset. The gender is female and there are glasses.

```shell
$ python3 partition_query.py -n 306 -s female -g True -q
```

The following command queries top 10 vectors that are most similar to the 255th vector in the dataset. The gender is male.

```shell
$ python3 partition_query.py -n 255 -s male -q
```
The following command queries top 10 vectors that are most similar to the 3450th vector in the dataset.

```shell
$ python3 partition_query.py -n 3450 -q
```

