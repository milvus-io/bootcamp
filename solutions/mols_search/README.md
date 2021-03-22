# Milvus 化学式检索

本项目后续将不再维护更新，最新内容将更新在https://github.com/zilliz-bootcamp/chemical_similarity_search。

## 环境准备

| 组件     | 推荐配置                                                     |
| -------- | ------------------------------------------------------------ |
| CPU      | Intel(R) Core(TM) i7-7700K CPU @ 4.20GHz                     |
| Memory   | 32GB                                                         |
| OS       | Ubuntu 18.04                                                 |
| Software | [Milvus 0.10.0](https://milvus.io/cn/docs/v0.10.0/install_milvus.md) <br />mols-search-webserver 0.7.0 <br />mols-search-webclient 0.3.0 |

以上配置已经通过测试，并且 Windows 系统也可以运行本次实验，以下步骤 Windows 系统通用。



## 数据准备

本次实验数据来源：[ftp://ftp.ncbi.nlm.nih.gov/pubchem/Compound/CURRENT-Full/SDF](ftp://ftp.ncbi.nlm.nih.gov/pubchem/Compound/CURRENT-Full/SDF)，该数据集是压缩的 SDF 文件，需要使用工具将其转换为 SMILES 文件，我们准备了转换后的一万条 SMILES 化学式文件 [test_1w.smi](./smiles-data)，下载该文件到本地：

```bash
$ wget https://raw.githubusercontent.com/milvus-io/bootcamp/0.10.0/solutions/mols_search/smiles-data/test_1w.smi
```



## 部署流程

#### 1. 启动 Milvus Docker

本次实验使用 Milvus 0.10.0CPU 版，安装启动方法参考https://milvus.io/cn/docs/v0.10.0/install_milvus.md 。



#### 2. 启动 mols-search-webserver docker

```bash
$ docker run -d -v <DATAPATH>:/tmp/data -p 35001:5000 -e "MILVUS_HOST=192.168.1.25" -e "MILVUS_PORT=19530" milvusbootcamp/mols-search-webserver:0.7.0
```

上述启动命令相关参数说明：

| 参数                          | 说明                                                         |
| ----------------------------- | ------------------------------------------------------------ |
| -v DATAPATH:/tmp/data         | -v 表示宿主机和 image 之间的目录映射<br />请将DATAPATH修改为你存放 test_1w.smi 数据的目录。 |
| -p 35001:5000                 | -p 表示宿主机和 image 之间的端口映射                         |
| -e "MILVUS_HOST=192.168.1.25" | -e 表示宿主机和 image 之间的系统参数映射<br />请修改`192.168.1.25`为启动 Milvus docker 的服务器 IP 地址 |
| -e "MILVUS_PORT=19530"        | 请修改`19530`为启动 Milvus docker 的服务器端口号             |



#### 3. 启动 mols-search-webclient docker

```bash
$ docker run -d -p 8001:80 -e API_URL=http://192.168.1.25:35001 milvusbootcamp/mols-search-webclient:0.3.0
```

> 参数 -e API_URL=http://192.168.1.25:35001 与本节第二部分相对应，请修改`192.168.1.25`为启动 Milvus docker 的服务器 IP 地址。



#### 4. 打开浏览器

```bash
# 请根据以上步骤修改 192.168.1.25 地址和 8001 端口
http://192.168.1.25:8001
```



## 界面展示

- 初始界面

![](./assert/init_status.PNG)

- 加载化学式数据
  1. 在 path/to/your/data 中输入 smi 文件位置：/tmp/data/test_1w.smi
  2. 点击 `+` 加载按钮
  3. 可以观察到化学式数量的变化：10000 Molecular Formula in this set

![](./assert/load_data.PNG)

- 化学式检索
  1. 输入待检索的**化学式**并按**回车**，如：Cc1ccc(cc1)S(=O)(=O)N
  2. 选择 TopK 值，将在右侧返回相似度最高的前 TopK 个化学式

![](./assert/search_data.PNG)

- 清除化学式数据

  点击`CLEAR ALL`按钮，将清除所有化学式数据

![](./assert/delete_data.PNG)



## 结论

本次实验实现了 Milvus 化学式检索，在数据准备时也可以使用自带 SMILES 数据。经测试，Milvus 在三千七百万化学式库/九千万化学式库（特征向量512维）性能表现如下：

|                                                              | 性能（三千七百万化学式库） | 性能（九千万化学式库） |
| :----------------------------------------------------------- | :------------------------- | :--------------------- |
| Milvus 单个化学式检索                                        | 190ms                      | 480ms                  |
| Milvus 批量 50 个化学式检索 性能结果为每个化学式检索的平均时间 | 6ms                        | 12ms                   |

Milvus 内存占用情况如下：

|                 | 三千七百万化学式库 | 九千万化学式库 |
| :-------------- | :----------------- | :------------- |
| Milvus 内存占用 | 2.6G               | 6G             |

可以看出，Milvus 在大规模化学式检索时有较快的检索性能和较低的内存占用。

基于 Milvus 我们搭建了九千多万的化学式检索系统 http://40.73.24.85 ，欢迎访问并检索您指定的化合物！
