# Distributed Solution Based on Mishards, a Sharding Middleware for Milvus Clusters


Milvus aims to achieve efficient similarity search and analytics for massive-scale vectors. A standalone Milvus instance can easily handle vector search among billion-scale vectors. However, for 10 billion, 100 billion or even larger datasets, a Milvus cluster is needed.

This topic displays how to use Mishards to build a Milvus cluster. Refer to https://www.milvus.io/docs/v1.0.0/milvus_docker-cpu.md for more information.

This topic assumes you can install and use Milvus in a standalone server. Refer to the following content to learn how to build a Milvus cluster.

## Install dependencies

A Milvus cluster needs at least two servers and one shared storage device.

1. Install [NVIDIA driver](https://www.nvidia.com/Download/index.aspx) 418 or higher.

2. Install [docker](https://docs.docker.com/install/linux/docker-ce/ubuntu/).

3. Install [docker-compose](https://docs.docker.com/compose/install/).

4. Install [nvidia-docker2](https://github.com/nvidia/nvidia-docker/wiki/Installation-(version-2.0)).

In this solution, we use two servers to build a small Milvus cluster, with one writable and the other read-only for Milvus.

## How to build

### 1. Install MySQL

You only need to run MySQL in either of the two servers.

Run MySQL using docker with `mysqld.cnf` and `mysql_dc.yml`.

```shell
$ docker-compose -f mysql_dc.yml up -d
```

Check whether MySQL service is successfully started:

```shell
$ docker ps
```

### 2. Run Milvus

Milvus must be installed in both servers. Different servers can have different access privileges for Milvus. It is recommended that Milvus has write access to one server and read-only access to the other.

#### Configure Milvus with write access

Use `cluster_wr_server.yml` and `cluster_wr.yml` for Milvus with write access. Update the configurations per the actual environment.

`cluster_wr_server.yml`:

![1577780602167](pic/image-1.png)

In this config file, the `enable` parameter determines whether to set to distributed mode, the `role` parameter determines whether Milvus has read-only access or write access and the value `rw` indicates that the Milvus has write access. You must update `meta_uri` to the IP address of the server that installs MySQL. You can set other parameters per the requirements in a standalone server.

`cluster_wr.yml`:

![1577931601864](pic/1577931601864.png)

`/test/solution/milvus/db` in `volumes` indicates the data storage location, which must direct to the shared storage device. All data storage locations in the cluster must use the same shared storage device. You can use default values for other parameters.

Run the following command to start Milvus:

```shell
$ docker-compose -f cluster_wr.yml up -d
```

#### Configure Milvus with read-only access

Use `cluster_ro_server.yml` and `cluster_ro.yml` for Milvus with read-only access. Update the configurations per the actual environment.

`cluster_ro_server.yml`:

![1577782332404](pic/image-2.png)

`role` is `ro`, which indicates that Milvus has read-only access and runs only during search. You must update `meta_uri` to the IP address of the server that installs MySQL. You can set other parameters per the requirements in a standalone server.

`cluster_ro.yml`:

![1577931719030](pic/1577931719030.png)

`/test/solution/milvus/db` in `volumes` indicates the data storage location, which must direct to the shared storage device. All data storage locations in the cluster must use the same shared storage device. You can use default values for other parameters.

Run the following command to start Milvus:

```shell
$ docker-compose -f cluster_ro.yml up -d
```

> Note: You can also refer to the [installation guide](https://milvus.io/docs/v0.7.1/guides/get_started/install_milvus/gpu_milvus_docker.md) to learn how to install and run Milvus. However, you must edit `server_config.yml` per the config files for Milvus with write access or read-only access. Also, all data storage locations in the cluster must map to the same shared storage device.

### 3.Run Mishards

You can run Mishards in either of the two servers. In the server to run Mishards, configure `cluster_mishards.yml`. 

![image](https://user-images.githubusercontent.com/53458891/113251640-3643fc00-92f5-11eb-8036-3946c9431734.png)


You need to modify the following parameters:

`SQLALCHEMY_DATABASE_URI`: change `192.168.1.85` to the IP address of the server that runs MySQL.

`WOSERVER`: change the value to the IP address of the server that grants write access to Milvus, such as `tcp://127.0.0.1:19530`。

`DISCOVERY_STATIC_HOSTS`: all IP addresses in the cluster.

`SERVER_PORT` defines the server ports of Mishards.

Use the following command to run Mishards:

```shell
$ docker-compose -f cluster_mishards.yml up
```

## Use Mishards

You have successfully built a Milvus cluster after completing the steps above.

Connect to the IP address of the server with Mishards and server port 19531 to connect to the Milvus cluster server. Other operations are the same as Milvus in a standalone server.

During a query, you can inspect the logs printed by Mishards to see tasks assigned to each server in a cluster.
