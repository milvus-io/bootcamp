# Prepare Data Files

- [Data Files](https://github.com/milvus-io/bootcamp/blob/master/docs/data_preparation/data_file_consideration.md#DataFiles)
- [Import Data](https://github.com/milvus-io/bootcamp/blob/master/docs/data_preparation/data_file_consideration.md#ImportData)

## Data Files

In the lab tests, data sets from [SIFT1B](http://corpus-texmex.irisa.fr/) will be provided.

If you already have data for the tests, it is recommended to prepare vectors in .npy format, with each file containing <= 100,000 vectors. 

Saving data in .npy file could largely reduce the file size. Take the single-precision 512-dimensional vectors as an example, saving 100,000 million such vectors in CSV file takes about 800 MB, while in .npy file, the file size is reduced to < 400 MB.

If you have only the CSV files, follow below steps to convert them into binary files in .npy format:

1. Read the CSV file through `pandas.read_csv`, and generate `pandas.DataFrame` data structure.
2. Through `numpy.array`, convert `pandas.DataFrame` to `numpy.array` data structure.
3. Through `numpy.save`, save the array to a binary file in .npy format.

## Import Data

Currently, Milvus provides Python SDK. Follow below steps to import vector data through Python scripts.

#### Import .npy files

1. Read the .npy file through `numpy.load` , and generate `numpy.array` data structure.
2. Through `numpy.array.tolist`, convert the `numpy.array` to a 2-dimensional list (in the form of [[],[]...[]]).
3. Import the 2-dimensional list into Milvus through the Python scripts. **A list of vector IDs** will be returned instantly. 

#### Import .csv files

1. Read the CSV file through `pandas.read_csv`, and generate `pandas.DataFrame` data structure.
2. Through `numpy.array`, convert `pandas.DataFrame` to `numpy.array` data structure.
3. Through `numpy.array.tolist`, convert the `numpy.array` to a 2-dimensional list (in the form of [[],[]...[]]).
4. Import the 2-dimensional list into Milvus through the Python scripts. **A list of vector IDs** will be returned instantly. 



