# Prepare Data Files

- [Data Files](https://github.com/milvus-io/bootcamp/blob/master/docs/data_preparation/data_file_consideration.md#DataFiles)
- [Import Data](https://github.com/milvus-io/bootcamp/blob/master/docs/data_preparation/data_file_consideration.md#ImportData)
- [Save Vector ID](https://github.com/milvus-io/bootcamp/blob/master/docs/data_preparation/data_file_consideration.md#SaveVectorID)

## Data Files

Prepare vectors in CSV file format . It is recommended to set the file size limit of 0.1 million vectors. To speed up data insertion, try converting the CSV file into binary files in .npy format in advance.

1. Read the CSV file through `pandas.read_csv`, and generate `pandas.DataFrame` data structure.
2. Through `numpy.array`, convert `pandas.DataFrame` to `numpy.array` data structure.
3. Through `numpy.save`, save the array to a binary file in .npy format.

Converting the CSV file into .npy file could largely reduce the file size. Take the single-precision 512-dimensional vectors as an example, saving 0.1 million such vectors in CSV file takes about 800 MB, while in .npy file, the file size is reduced to < 400 MB.

Don't remove the original CSV file after the conversion, as it might be used later to check the vector query results. 

## Import Data

Currently, Milvus provides Python SDK. Follow below steps to import vector data through Python scripts: 

1. Read the CSV file through `pandas.read_csv`, and generate `pandas.DataFrame` data structure.
2. Through `numpy.array`, convert `pandas.DataFrame` to `numpy.array` data structure.
3. Through `numpy.array.tolist`, convert the `numpy.array` to a 2-dimensional list (in the form of [[],[]...[]]).
4. Import the 2-dimensional list into Milvus through the Python scripts. **A list of vector IDs** will be returned instantly. 

> **Note**: To verify the search precision, please prepare the ground truth set yourself.

## Save Vector ID

To reduce memory usage, in vector querying, Milvus returns only the vector IDs. As the current Milvus version does not support the storage of initial vector data, you need to manually store your vectors and their corresponding vector IDs.

If you want vector-based mixed queries, you can import vector IDs, initial vectors and their related attributes into a relational database. For detailed example, you may refer to:

***How to Realize Vector-based Mixed Query Through Combined Application of Milvus and PG***

