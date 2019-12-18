# Milvus 快速上手

该指南主要包含 Milvus Docker 版的快速安装，以及相关 Python 示例代码的运行。如果想进一步了解 Milvus 的使用，请访问 [Milvus 用户指南](https://github.com/milvus-io/docs/blob/0.5.3/zh-CN/userguide/install_milvus.md)。

## 安装前提
1. Milvus Docker 版目前仅在 Linux 系统上运行，请确保您的 Linux 系统符合以下版本：

   | Linux        | Version        |
   | :----------------------- | :---------- |
   | CentOS                   | 7.5 或以上 |
   | Ubuntu LTS               | 18.04 或以上 |

2. 硬件配置要求：

   | Component |   Minimum Config         |
   | -------- | ---------------- |
   | CPU      | Intel CPU Haswell 或以上         |
   | GPU      | Nvidia Pascal series 或以上 |
   | GPU Driver    | CUDA 10.1, Driver 418.74 或以上 |
   | Memory     | 8 GB或以上 (取决于具体向量数据规模) |
   | Storage | SATA 3.0 SSD 或以上    |

3. 客户端浏览器要求：

   Milvus 提供了基于 Prometheus 监控和 Grafana 的展示平台，可以对数据库的各项指标进行可视化展示，兼容目前主流的 Web 浏览器如：微软 IE、Google Chrome、Mozilla Firefox 和 Safari 等。

4. 请确保您已经安装以下软件包，以便 Milvus Docker 版能正常运行：

   - [NVIDIA driver](https://www.nvidia.com/Download/index.aspx)(418 或以上)
   - [Docker](https://docs.docker.com/install/)（19.03或以上）


## 安装 Milvus Docker 版

1. 下载 Milvus Docker 镜像文件

   ```shell
   # Download Milvus Docker image
   $ docker pull milvusdb/milvus:0.5.3-d111319-75998b
   ```

2. 创建 Milvus 文件夹，并添加 server_config.yaml 和 log_config.conf。

   ```shell
   # Create Milvus file
   $ mkdir /home/$USER/milvus
   $ cd /home/$USER/milvus
   $ mkdir conf
   $ cd conf
   $ wget https://github.com/milvus-io/docs/blob/0.5.3/assets/server_config.yaml
   $ wget https://github.com/milvus-io/docs/blob/0.5.3/assets/log_config.conf
   ```

3. 启动 Milvus server

   ```shell
   # Start Milvus
   $ nvidia-docker run -td --runtime=nvidia -p 19530:19530 -p 8080:8080 -v /home/$USER/milvus/db:/opt/milvus/db -v /home/$USER/milvus/conf:/opt/milvus/conf -v /home/$USER/milvus/logs:/opt/milvus/logs milvusdb/milvus:0.5.3-d111319-75998b
   ```

4. 获得 Milvus container id

   ```shell
   # Get Milvus container id
   $ docker ps -a
   ```

5. 确认 Milvus 运行状态

   ```shell
   # Make sure Milvus is up and running
   $ docker logs <milvus container id>
   ```

## 运行 Python 示例代码

接下来，让我们来运行一个 Python 程序示例。您将创建一个向量数据表，向其中插入 10 条向量，然后运行一条向量相似度查询。

1. 请确保系统已经安装了 [Python3](https://www.python.org/downloads/)

2. 安装 Milvus Python SDK

   ```shell
   # Install Milvus Python SDK
   $ pip install pymilvus==0.2.5
   ```

   > 提示：如果需要进一步了解 Milvus Python SDK，请阅读 [Milvus Python SDK 使用手册](https://pypi.org/project/pymilvus)。

3. 创建 *example.py* 文件，并向文件中加入 [Python 示例代码](https://github.com/milvus-io/pymilvus/blob/0.5.3/examples/example.py)。

4. 运行示例代码。

   ```shell
   # Run Milvus Python example
   $ python3 example.py
   ```

5. 确认程序正确运行。

   恭喜您！您已经成功完成了在 Milvus 上的第一次向量相似度查询。
