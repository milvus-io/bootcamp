# Build a Milvus distributed cluster based on JuiceFS

This tutorial uses [JuiceFS](https://github.com/juicedata/juicefs) as shared storage to build Mishards. JuiceFS is an open source POSIX file system built on top of Redis and object storage (e.g. S3), and is equivalent to a stateless middleware that helps various applications share data through a standard file system interface. As shown in the diagram below:

<img src="2.png" alt="2" style="zoom:60%;" />

## Environment preparation

To build a Milvus cluster you need at least two servers and a shared storage device, i.e. **JuiceFS**.

1. Install [NVIDIA driver](https://www.nvidia.com/Download/index.aspx) 418 or higher.

2. Install [Docker](https://docs.docker.com/install/linux/docker-ce/ubuntu/).

2. Install [Docker Compose](https://docs.docker.com/compose/install/).

3. Install [nvidia-docker 2.0](https://github.com/nvidia/nvidia-docker/wiki/Installation-(version-2.0)).

## Building steps

This project is a distributed build solution based on Milvus 1.0.

### 1. Install MySQL

MySQL service can be started on any of the **server** in the cluster, for MySQL installation see [Managing Metadata with MySQL](https://milvus.io/docs/v1.0.0/data_manage.md).

### 2. Install and configure JuiceFS

The [precompiled binaries](https://github.com/juicedata/juicefs/releases) selected for this tutorial can be downloaded directly, and the detailed installation process can be found on the [JuiceFS website](https://github.com/juicedata/juicefs) for the installation tutorial.

After downloading you will need to install the dependencies, JuiceFS requires a Redis (2.8 and above) server to store the metadata, see [Redis Quick Start](https://redis.io/topics/quickstart). **It's highly recommended use Redis service managed by public cloud provider if possible.**

JuiceFS needs to be configured with object storage, i.e. create a new volume through `juicefs format` command. The object storage used in the tutorial is Azure Blob Storage, you need to choose your own suitable object storage, refer to [the guide](https://github.com/juicedata/juicefs/blob/main/docs/en/how_to_setup_object_storage.md). Once the volume has been formatted, it can be mounted as a directory.

Assuming that you have a locally running Redis service, use it to format a volume called `test`:

```sh
$ export AZURE_STORAGE_CONNECTION_STRING="DefaultEndpointsProtocol=https;AccountName=XXX;AccountKey=XXX;EndpointSuffix=core.windows.net"
# Formatting a volume
$ ./juicefs format \
    --storage wasb \
    --bucket https://<container> \
    ... \
    localhost test
```

If the Redis service is not running locally, the `localhost` in the above command needs to be replaced with a full address like this: `redis://username:password@host:6379/1`.

Once the volume has been formatted, it can be mounted as a directory (e.g. `~/jfs`):

```sh
$ ./juicefs mount -d localhost ~/jfs
```

For more information, please refer to [JuiceFS website](https://github.com/juicedata/juicefs).

### 3. Starting Milvus

Each server in the cluster requires Milvus to be installed, and different servers can be configured with different read and write permissions to Milvus. One server in the cluster is configured as write-only, the others are read-only.

#### Write-only/Read-only configuration

In the Milvus system configuration file `server_config.yaml`, the following parameters need to be configured.

##### Section `cluster`

| Parameter     | Description                    | Parameter Setting |
| :------------ | :----------------------------- | :---------------- |
| `enable`      | Whether to enable cluster mode | `ture`            |
| `role`        | Milvus deployment role         | `rw`              |

##### Section `general`

| Parameter     | Description                                                                                                                        | Parameter Setting                          |
| :------------ | :-----------------------------------------------------------                                                                       | :---------------------------------------   |
| `meta_uri`    | URI for metadata storage, using  MySQL (for distributed cluster Milvus). Format: `dialect://username:password@host:port/database`. | `mysql://root:milvusroot@host:3306/milvus` |

***Read-only requires the parameter `role` to be set to `ro`, the rest of the parameters are the same as write-only.***

#### Starting Milvus

```sh
sudo docker run -d --name milvus_gpu_1.0.0 --gpus all \
-p 19530:19530 \
-p 19121:19121 \
-v /root/jfs/milvus/db:/var/lib/milvus/db \    # /root/jfs/milvus/db is the path to JuiceFS
-v /home/$USER/milvus/conf:/var/lib/milvus/conf \
-v /home/$USER/milvus/logs:/var/lib/milvus/logs \
-v /home/$USER/milvus/wal:/var/lib/milvus/wal \
milvusdb/milvus:1.0.0-gpu-d030521-1ea92e
```

### 4. Starting Mishards

The Mishards service can simply be started on any of the **devices** in the cluster, here we use the `cluster_mishards.yml` file from the project:

```yaml
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

- `SQLALCHEMY_DATABASE_URI`: change `192.168.1.85` to the IP address where MySQL is located.
- `WOSERVER`: change to the IP address of the Milvus writeable instance. Reference format: `tcp://127.0.0.1:19530`.
- `DISCOVERY_STATIC_HOSTS`: all IP addresses in the cluster.
- `SERVER_PORT`: defines the service port for Mishards

Start the Mishards service with the following command.

```sh
$ docker-compose -f cluster_mishards.yml up
```

## **FAQ**

### 1. Can I mount JuiceFS volume with non-root user?

JuiceFS can be mounted by any user. The default cache directory is `$HOME/.juicefs/cache` (macOS) or `/var/jfsCache` (Linux), make sure the user has write access to this directory, or switch to another directory with sufficient permissions.

If you do not use a privileged user, you may get an error like `docker: Error response from daemon: error while creating mount source path 'XXX': mkdir XXX: file exists`. Refer to [JuiceFS FAQ](https://github.com/juicedata/juicefs/blob/main/docs/en/faq.md#docker-error-response-from-daemon-error-while-creating-mount-source-path-xxx-mkdir-xxx-file-exists) for more information.

### 2. Cannot connect to Redis

When Redis is executed with the default configuration (binding all the interfaces) and without any password in order to access it, it enters a special mode called **protected mode**. So you need to configure the `redis.conf` file and set the `protected-mode` to `no`.
