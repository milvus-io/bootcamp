# Mishards - Milvus 集群分片中间件

Milvus 旨在帮助用户实现海量非结构化数据的近似检索和分析。单个 Milvus 实例可处理十亿级数据规模，而对于百亿或者千亿级数据，则需要一个 Milvus 集群实例。该实例对于上层应用可以像单机实例一样使用，同时满足海量数据低延迟、高并发业务需求。

本文主要展示如何使用 Mishards 分片中间件来搭建 Milvus 集群。

关于Mishards-Milvus更多详解请参考（https://github.com/milvus-io/milvus/blob/0.6.0/shards/README_CN.md）

本文默认你已经会在单机上安装使用milvus了，在此基础上可参考下文搭建一个Milvus集群。

## 环境准备

搭建Milvus集群至少需要两台设备和一个共享存储设备。

1.安装[Nvidia-driver](https://www.nvidia.com/Download/index.aspx)418 或更高版本。

2.安装[docker](https://docs.docker.com/install/linux/docker-ce/ubuntu/)。

2.安装[docker-compose](https://docs.docker.com/compose/install/)。

3.安装[nvidia-docker2](https://github.com/nvidia/nvidia-docker/wiki/Installation-(version-2.0))。

在本示例中，将用两台设备搭建一个小的milvus集群。其中一台设置为可写，另一台设置为只读。

## 搭建步骤

### 1.安装mysql

mysql服务只需要在集群中**任意一台**设备上启动即可。

此处将通过docker启动mysql。可以看到，本项目目录下包含了mysqld.cnf、mysql_dc.yml两个脚本。通过如下命令运行脚本将启动mysql服务。

```shell
$ docker-compose -f mysql_dc.yml up -d
```

查看mysql服务是否启动成功

```shell
$ docker ps
```

### 2.启动Milvus

集群中的每一台设备均需要安装milvus,不同的设备可给milvus配置不同的读写权限。（这里建议给集群中的一台设备配置为可写，其他均为只读）

#### 可写

本项目目录下有cluster_wr_server.yml、cluster_wr.yml两个脚本。根据实际环境，修改相应配置。

cluster_wr_server.yml：

![1577780602167](pic\1577780602167.png)

在该配置文件中，参数deploy_mode决定了milvus是只读还是可写。此处选择cluster_writable表示为该milvus可写。参数backend_url应修改为mysql所安装的设备的地址。其余配置参照milvus单机版时的配置。

cluster_wr.yml：

![1577931601864](pic\1577931601864.png)

此处volumes下的路径/test/solution/milvus/db是数据存储位置，该路径需指向一个共享存储，集群中的所有设备数据存储位置均设置为同一个共享存储。其余参数使用默认设置即可。

通过如下命令启动milvus服务：

```shell
$ docker-compose -f cluster_wr.yml up -d
```

#### 只读

本项目目录下有cluster_ro_server.yml、cluster_ro.yml两个脚本。根据实际环境，修改相应配置。

cluster_ro_server.yml：

![1577782332404](pic\1577782332404.png)

此处deploy_mode设置为cluster_readonly表示为该milvus只可读（即只有在search时才会启用该服务）。参数backend_url应修改为mysql所安装的设备的地址。其余配置参照milvus单机版时的配置。

cluster_ro.yml:

![1577931719030](pic\1577931719030.png)

该处路径同上。

通过如下命令启动milvus服务：

```shell
$ docker-compose -f cluster_ro.yml up -d
```

（注意：集群中每台设备的Milvus安装与启动也可参考[milvus官网](https://milvus.io/cn/docs/v0.6.0/guides/get_started/install_milvus/gpu_milvus_docker.md)的安装步骤执行。但是需要修改conf文件夹下的配置文件server_config.yml，可写设备按照cluster_wr_server.yml修改参数deploy_mode和backend_url，只读设备按照cluster_ro_server.yml修改参数deploy_mode和backend_url。且启动时，所有设备数据存储路径均需要映射到同一个共享存储。）

### 3.启动mishards

mishards服务只需在集群中**任意一台**设备上启动即可。

本项目目录中有一个cluster_mishards.yml文件，如图：

![1577783243935](pic\1577783243935.png)

在脚本中需要注意修改的参数：

SQLALCHEMY_DATABASE_URI：将192.168.1.85修改为mysql所在的ip地址。

WOSERVER：修改为 Milvus 可写实例的IP地址。参考格式： `tcp://127.0.0.1:19530`。

DISCOVERY_STATIC_HOSTS：集群中的所有ip地址。

(SERVER_PORT定义了Mishards的服务端口)

通过如下命令启动mishards服务：

```
$ docker-compose -f cluster_mishards.yml up
```



## 使用

完成上述步骤后，成功搭建Milvus集群。

通过连接Mishards所在的ip地址，以及Mishards服务端口19531就可以连接该milvus集群server。其余操作与Milvus 单机版一致。

在查询过程中观察Mishards打印的日志，可以看见集群中每个设备所分配到的任务。