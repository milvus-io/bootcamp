# Distributed Solution Based on Mishards, a Sharding Middleware for Milvus Clusters


Milvus aims to achieve efficient similarity search and analytics for massive-scale vectors. A standalone Milvus instance can easily handle vector search among billion-scale vector datasets. However, for 10 billion, 100 billion or even larger datasets, a Milvus cluster is needed.

Here we will go over how to use Mishards to build a Milvus cluster. Refer to https://milvus.io/docs/v1.0.0/mishards.md for more information.

This topic assumes you can install and use Milvus in a standalone server. Refer to the following content to learn how to build a Milvus cluster.

## Requirements

| Packages   | Services    |
| -                  | -                 |   
| docker-compose    | Milvus-1.0.0      |
|                   | MySQL             |


## Up and Running

### Setup Docker Network

For this example we are going to create a docker network in order to simulate having multiple servers on different IPs. This is done so that anyone can recreate this example and have it working. Run the following command to create a network with the name `my-net` and IP range of `10.0.0.0/16`.

```shell
$ docker network create --subnet 10.0.0.0/16 my-net
```

### Install and run MySQL

Only one instance of MySQL needs to be run in the cluster. For the example, we have provided a file for starting up the MySQL server. To run the server, run the following command. 

```shell
$ docker-compose -f mysql_dc.yml up -d
```

In order to see which IP address it was assigned in my-net, run the following command:

```shell
$ docker network inspect my-net
```
This IP is important as we will need it for the rest of the nodes.

### Run Milvus Write Node

For now, only one write node is supported in the cluster. In order to launch this node we must first configure the cluster_wr_server.yml and cluster_wr.yml

First lets confgiure the `cluster_wr_server.yml`. There is one line that might need to be edited for this example:

```yaml
general:
  timezone: UTC+8
  meta_uri: mysql://root:milvusroot@10.0.0.2:3306/milvus
```

In the `meta_uri` you might need to change the IP address to the one found when running the previous "docker network inspect" command. If following this guide line by line, the docker engine should automatically assign the server to 10.0.0.2, but this needs to be checked just in case.

Next we will need to edit the `cluster_wr.yml`. There is one area that might need to be changed:

```yaml
volumes:
    - ./test:/var/lib/milvus/db
    - ./cluster_wr_server.yml:/var/lib/milvus/conf/server_config.yaml
```
The first volume specifies where the collections will be stored. This location needs to be shared by all the nodes, so we mount a location to the container. In this case the cluster expects the shared location to be at `./test`. If this folder does not exist in the current directory, the cluster will not work, so you must first create the folder test in the current directory to make sure everything works. The second line doesnt need to be edited for this example, it specifies which server_config.yml is used for the node being created, in this case the wr node. 

Now, in order to run the write node, we run the following line:

```shell
$ docker-compose -f cluster_wr.yml up -d
```

Now when we run the following line we should see both the MySQL server and the write node. 

```shell
$ docker network inspect my-net
```

### Run Milvus Read Nodes

In a cluster we are able to have as many read nodes as we want. In order to launch the read nodes, we first need to start off by doing the same edits as  the write nodes. In the `cluster_ro_server.yml` we must edit this line to the correct IP if it has changed:

```yaml
meta_uri: mysql://root:milvusroot@10.0.0.2:3306/milvus
```

As with the write server, in the `cluster_ro.yml` we must point to the shared storage location for the db, in this example we assume that you have created the `./test` folder in the working directory.

```yaml
ports:
    - 19530
volumes:
    - ./test:/var/lib/milvus/db
    - ./cluster_ro_server.yml:/var/lib/milvus/conf/server_config.yaml
networks:
```

Something that you might have noticed in the ro version is that we are letting docker assign the port maps for accessing from outside the network. It is not necessary to expose ports, but it does make debugging easier so in this example we will be exposing all of the containers to localhost. 

In order to run the ro nodes, we can run the following command:

```shell
$ docker-compose -f cluster_ro.yml up -d --scale milvus_ro=3
```
The scale argument just says how many nodes of to boot up.

To check which IP ports are exposed for each container we can use the following command:

```shell
$ docker ps
```

### Run Mishards

The last step is running the mishards. In order to do this we need to make a few changes to the `cluster_mishards.yml`,

You need to modify the following parameters:

```yaml
ports:
        - "0.0.0.0:19540:19530"
```
You need to select which port you want to be able to access the mishards cluster from, in this case port 19540 on the localhost. The internal IP, in this case 19530, should match up with the `SERVER_PORT` paramter that is lower in the file.

```yaml
volumes:
    - ./test_mi:/tmp/milvus/db
```
You must change the location that the shared storage is located at, if using the example test folder, no change is necessary.

```yaml
SQLALCHEMY_DATABASE_URI: mysql+pymysql://root:milvusroot@10.0.0.2:3306/milvus
```
Same situation as the other nodes, if MySQL is not on 10.0.0.2, you must change it in this line.

```yaml
WOSERVER: tcp://10.0.0.3:19530
```
This line should point to the IP address of the wo node, in this example it most likely will be 10.0.0.3.

```yaml
DISCOVERY_STATIC_HOSTS: 10.0.0.3, 10.0.0.4, 10.0.0.5, 10.0.0.6
DISCOVERY_STATIC_PORT: 19530
```
The `DISCOVERY_STATIC_HOSTS` parameter should list all the IPs of the milvus nodes, so the IP of the 1 wo node and the IPs of the 3 ro nodes. These can be found using the `$ docker network inspect my-net`.
The `DISCOVERY_STATIC_PORT` parameter signifies which ports the milvus nodes are listening to. The default is 19530, and if following this guide it should not be changed

Once set up, the following command can be used to startup the mishards cluster:

```shell
$ docker-compose -f cluster_mishards.yml up
```

## Use Mishards

Now that mishards is up and running we can begin to use it. In order to use it you should connect to Milvus how you normally do, with the only difference being that the IP and the PORT should be the mishards node. In this case, if running an application outside of docker, the IP adress should be `0.0.0.0` and the port should be `19540` as that is the port we exposed for the mishards node. 

During a query, you can inspect the logs printed by Mishards to see tasks assigned to each node in a cluster.
