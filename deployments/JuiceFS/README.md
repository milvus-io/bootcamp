### Build a Milvus distributed cluster based on JuiceFS

This tutorial uses JuiceFS as shared storage to build Mishards. JuiceFS is an open source POSIX file system built on top of object stores such as Redis and S3, and is equivalent to a stateless middleware that helps various applications share data through a standard file system interface. As shown in the diagram below

![](1.png)

### **Environment Preparation**

To build a Milvus cluster you need at least two devices and a shared storage device, i.e. **JuiceFS**.

1. Install [Nvidia-driver](https://www.nvidia.com/Download/index.aspx)418 or higher.

2. Install [docker](https://docs.docker.com/install/linux/docker-ce/ubuntu/).

2. Install [docker-compose](https://docs.docker.com/compose/install/).

3. Install [nvidia-docker2](https://github.com/nvidia/nvidia-docker/wiki/Installation-(version-2.0)).

### **Building steps**

This project is a distributed build solution based on Milvus 1.0

1.**Install MySQL**

MySQL services can be started on any of the **devices** in the cluster, for MySQL installation see [Managing Metadata with MySQL](https://milvus.io/cn/docs/v1.0.0/data_manage.md)

2.**Install JuiceFS**

The [pre-compiled version](https://github.com/JuiceFSicedata/JuiceFSicefs/releases) selected for this tutorial can be downloaded directly, and the detailed installation process can be found on the [JuiceFS official website](https://github.com/ JuiceFSicedata/JuiceFSicefs/blob/main/README_CN.md) for the installation tutorial.

After downloading you will need to install the dependencies, JuiceFS requires a Redis (2.8 and above) server to store the metadata, see [Redis Quick Start](https://redis.io/topics/quickstart).

JuiceFS needs to be configured with object storage, the object storage used in the tutorial is Azure Blob Storage, users need to choose their own suitable object storage, refer to [Text Block](https://github.com/juicedata/juicefs/blob/main/docs/en/how_to_setup_object_storage.md) . Once the object storage has been formatted and completed, it can be mounted as a directory.

Assuming that you have a locally running Redis service, use it to format a filesystem called `test`.

```
$  export AZURE_STORAGE_CONNECTION_STRING="DefaultEndpointsProtocol=https;AccountName=XXX;AccountKey=XXX;EndpointSuffix=core.windows.net"
$ ./juicefs format \
    --storage wasb \
    --bucket https://<container> \
    ... \
    localhost test #Formatting
```

If the Redis service is not local, the localhost needs to be replaced with a full address like this: redis://user:password@host:6379/1

Once the filesystem has been formatted, it can be mounted as a directory.

```
$ ./juicefs mount -d localhost ~/jfs
```

3.**Start Milvus**

Each device in the cluster requires Milvus to be installed, and different devices can be configured with different read and write permissions to Milvus. One device in the cluster is configured as writable, the others are read-only

**write-only**/**read-only**

In the Milvus system configuration file **server_config.yaml**, the following parameters need to be configured

###### Section `cluster` 

| **Parameter** | **Description**                 | Parameter Setting |
| :------------ | :------------------------------ | :---------------- |
| enable        | Whether to enable cluster mode. | ture              |
| role          | Milvus deployment role          | rw                |

##### Section `general`

| **Parameter** | **Description**                                              | Parameter Setting                        |
| :------------ | :----------------------------------------------------------- | :--------------------------------------- |
| `meta_uri`    | URI for metadata storage, using  MySQL (for distributed cluster Milvus). Format: `dialect://username:password@host:port/database`. `dialect` | mysql://root:milvusroot@host:3306/milvus |

***Read-only requires the parameter role to be set to ro, the rest of the parameters are the same as write-only***

**Milvus start-up configuration**

```
sudo docker run -d --name milvus_gpu_1.0.0 --gpus all \
-p 19530:19530 \
-p 19121:19121 \
-v /root/jfs/milvus/db:/var/lib/milvus/db \    #/root/jfs/milvus/db is the path to shared storage
-v /home/$USER/milvus/conf:/var/lib/milvus/conf \
-v /home/$USER/milvus/logs:/var/lib/milvus/logs \
-v /home/$USER/milvus/wal:/var/lib/milvus/wal \
milvusdb/milvus:1.0.0-gpu-d030521-1ea92e
```

4.**Starting Mishards**

The Mishards service can simply be started on any of the **devices** in the cluster, here we use the `cluster_mishards.yml` file from the project 

```
version: "2.3"
services:
    mishards:
        restart: always
        image: milvusdb/mishards
        ports:
            - "0.0.0.0:19531:19531"
            - "0.0.0.0:19532:19532"
        volumes:
            - /tmp/milvus/db:/tmp/milvus/db
- /tmp/mishards_env:/source/mishards/.env
        command: ["python", "mishards/main.py"]
        environment:
            FROM_EXAMPLE: 'true'
            SQLALCHEMY_DATABASE_URI: mysql+pymysql://root:milvusroot@192.168.1.85:3306/milvus?charset=utf8mb4
            DEBUG: 'true'
            SERVER_PORT: 19531
            WOSERVER: tcp://192.168.1.85:19530
            DISCOVERY_PLUGIN_PATH: static
            DISCOVERY_STATIC_HOSTS: 192.168.1.85, 192.168.1.38
            DISCOVERY_STATIC_PORT: 19530
```

Parameters to note in the script that need to be changed.

`SQLALCHEMY_DATABASE_URI`: change `192.168.1.85` to the IP address where MySQL is located.

`WOSERVER`: change to the IP address of the Milvus writeable instance. Reference format: `tcp://127.0.0.1:19530`.

`DISCOVERY_STATIC_HOSTS`: all IP addresses in the cluster.

`SERVER_PORT` defines the service port for Mishards

Start the Mishards service with the following command.

```
$ docker-compose -f cluster_mishards.yml up
```

### **Caution**

1.Can I replace the root user in JuiceFS?

JuiceFS can be mounted by any user. The default cache directory is `$HOME/.Juicefs/cache` (macOS) or `/var/jfsCache` (Linux), make sure the user has write access to this directory, or switch to another directory with permissions

If you do not use a privileged user, you may get a `docker: Error response from daemon: error while creating mount source path 'XXX': mkdir XXX: file exists` error.

2.When installing the JuiceFS dependency on Redis, you need to configure the redis.conf file and set the protection node to no