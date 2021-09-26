# Lab Test 1: One Million Vector Search

This experiment uses one million data from the SIFT1B dataset to test the performance and accuracy of Milvus 2.0.

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

#### Download test tools

Download the following data and scripts:

- 1 million test data: https://drive.google.com/file/d/1lwS6iAN-ic3s6LLMlUhyxzPlIk7yV7et/view?usp=sharing
- Query data: https://drive.google.com/file/d/17jPDk93PQsB5yGh1J1YD9N7X8jvPEUQL/view?usp=sharing
- Ground truth: https://drive.google.com/file/d/1vBP9mKu5oxyognHtOBBRtLvyPvo8cCp0/view?usp=sharing
- Test scripts: [/bootcamp/benchmark_test/scripts/](/benchmark_test/scripts/)

Create a folder named `milvus_sift1m` and move all downloaded files to the folder:

- Unzip the 1 million test data to get the `bvecs_data` folder that contains 10 `npy` files. Each `npy` file contains 100,000 vectors.
- Unzip the query data to get the `query_data` folder that contains `query.npy`, which contains 10,000 vectors to query.
- Unzip the ground truth data to get the gnd folder with `ground_truth_1M.txt`, which contains the locations of top 1000 similar vectors in the query data.
- The test script files contain the following Python scripts: `main.py`、`milvus_toolkit.py`、`milvus_load.py`、`config.py`。

> **Note:** Please go through the README carefully before testing with script . Make changes to the parameters in the script to match your scenario.



## 2. Create a table

Make sure Milvus is already installed and started. (For details of Milvus installation, please read [Milvus Quick Start](https://milvus.io/docs/preview/install_standalone-docker.md)).

>  Before testing, please modify the corresponding parameters according to the [script instructions](/benchmark_test/scripts/README.md)


Go to `milvus_sift1m`, and run the following command to create a table:

```bash
$ python3 main.py --collection ann_1m_sq8 --create
```

Vectors are then inserted into a table named `ann_1m_sq8h`, with the index_type of `IVF_SQ8H`. 

To show the available tables and number of vectors in each table, use the following command:

```bash
#Show if has collections
$ python3 main.py --collection ann_1m_sq8 --has
#Show the number of the entities in the collection ann_1m_sq8h
$ python3 main.py --collection ann_1m_sq8 --rows
```

## 3.  Insert data and build indexes

Run the following command to import 1,000,000 rows of data:

```bash
$ python3 main.py --collection=ann_1m_sq8 --insert
```

You can see that all data is imported from the file for once.

Run the following command to check the number of rows in the table:

```bash
$ python3 main.py --collection=ann_1m_sq8 --rows
```

Run the following command to create index and load data to memory:

```bash
$ python3 main.py --collection ann_1m_sq8 --index_type IVF_SQ8 --create_index

$ python3 main.py --collection ann_1m_sq8 --load
```

## 4. Accuracy test

SIFT1B provides not only the vector dataset to search 10,000 vectors, but also the top 1000 ground truth for each vector, which allows convenient calculation of precision rate. The vector search accuracy of Milvus can be represented as follows:

Accuracy = Number of shared vectors (between Milvus search results and Ground truth) / (query_records * top_k)

####  Run query script

Before the accuracy test, you need to manually create the directory `recall_result / recall_compare_out` to save the test results. To test the search precision for  top1(top10, top100, top200) results of 500 vectors randomly chosen from the 10,000 query vectors, go to directory `milvus_sift1m`, and run this command:

```bash
$ python3 main.py --collection=ann_1m_sq8 --search_param 128 --recall
```

> Note: search_param is nprobe value. nprobe affects search accuracy and performance. The greater the value, the higher the accuracy, but the lower the performance. In this experiment.

After executing the above command, an `ann_sift1m_sq8_128_500_recall.txt` text file will be generated in the` recall_result` folder. The text file records the id and distance of the most similar first 200 vectors corresponding to 500 vectors,Every 200 lines in the text file correspond to a query result of a query. At the same time, multiple texts will be generated under the `recall_compare_out` file. Taking ` ann_sift1m_sq8_128_500_100` as an example, this text records the respective corresponding accuracy rates and the total average accuracy rate of the 500 vectors queried when topk = 100.

The accuracy rate has a positive correlation with search parameter `nprobe` (number of sub-spaces searched). In this test, when the `nprobe` = 64, the accuracy can reach > 90%.  However, as the `nprobe` gets bigger, the search time will be longer. 

Therefore, based on your data distribution and business scenario, you need to edit `nprobe` to optimize the trade-off between accuracy and search time. 

## 5. Performance test

To test search performance, go to directory *milvus_sift1m*, and run the following script: 

```bash
$ python3 main.py --collection=ann_1m_sq8 --search_param 128 --performance
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

