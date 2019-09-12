# 准备数据文件

  * [数据文件](#数据文件)
  * [数据导入](#数据导入)

## 数据文件
在后面的实验中，我们会提供基于 SIFT1B 的测试数据。

如果您已有测试数据，建议以
npy 格式准备向量数据，每个文件建议不超过 10 万条向量。以 512 维单精度向量为例，10 万条向量的 npy 文件大小不足 400 MB。而 csv 文件大小超过 800 MB。因此，npy 文件在数据导入时，有明显的速度优势。

如果只有 csv 文件，可以通过以下步骤生成相应的 npy 二进制文件：

1. 通过 pandas.read_csv 方法读入一个 csv 文件，生成 pandas.dataframe 数据类型
2. 通过 numpy.array 方法，将上述 pandas.dataframe 转换成 numpy.array 数据类型
3. 将 numpy.array 数据类型，通过 numpy.save 方法存为一个 npy 二进制文件

## 数据导入

目前 Milvus 数据库提供了一个 Python 客户端。通过 Python 脚本导入向量数据的过程：
### 针对 npy 文件
1. 通过 numpy.load 方法读入一个 npy 文件，生成 numpy.array 类型的数据
2. 通过 numpy.array.tolist 方法将 numpy.array 数据转换成 2 维列表（形如，[[],[]...[]]）
3. 通过 Milvus 提供的 Python API 将 2 维列表数据导入 Milvus 数据库，同时返回向量 ID 列表

### 针对 csv 文件
1. 通过 pandas.read_csv 方法读入一个 csv 文件，生成 pandas.dataframe 类型的数据
2. 通过 numpy.array 方法，将上述 pandas.dataframe 转换成 numpy.array 类型的数据
3. 通过 numpy.array.tolist 方法将 numpy.array 数据导转换成 2 维列表（形如，[[],[]...[]]）
4. 通过 Milvus 提供的 Python API 将 2 维列表数据导入 Milvus 数据库，同时返回向量 ID 列表
