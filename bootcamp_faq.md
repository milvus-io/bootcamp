# Bootcamp FAQ

## Deployment

**Q1:** **ERROR with Compose file './docker-compose.yaml' is invalid**

Please check the version of docker and docker-compose，[Milvus requires](https://milvus.io/docs/v2.1.x/prerequisite-docker.md#Software-requirements) Docker >= 19.03 and Docker Compose >= 1.25.1.

**Q2:** **Can I use my own Milvus or MySQL**

Of course, you only need to modify the parameters related to Milvus and MySQL in server/src/config.py. If it is a webserver deployed with docker-compose, you can also modify the `webserver.environment ` parameter in docker-compose.yaml.

**Q3:** **When start server and show "Error while attempting to bind on address(0.0.0.0,5000) address already in use"**

It means that port 5000 has been occupied, please close the process; you can also modify the port of the webserver in docker-compose.yaml (note that `API_URL` in `webclient.environment` needs to be modified to the correspond port).

## System Running

**Q1:** **How to check the system**

Run `docker-compose ps`, and you can see that the status of all containers is "Up", for example:

```Bash
CONTAINER ID   IMAGE                                         COMMAND                  CREATED              STATUS                             PORTS                               NAMES
25b4c8e13590   milvusbootcamp/img-search-server:towhee       "/bin/sh -c 'python3…"   59 seconds ago       Up 49 seconds                      0.0.0.0:5000->5000/tcp              img-search-webserver
ae9a9a783952   milvusdb/milvus:v2.0.0-rc8-20211104-d1f4106   "/tini -- milvus run…"   59 seconds ago       Up 58 seconds                      0.0.0.0:19530->19530/tcp            milvus-standalone
7e88bdf66d96   minio/minio:RELEASE.2020-12-03T00-03-10Z      "/usr/bin/docker-ent…"   About a minute ago   Up 59 seconds (healthy)            9000/tcp                            milvus-minio
4a3ea5fff0f9   mysql:5.7                                     "docker-entrypoint.s…"   About a minute ago   Up 59 seconds                      0.0.0.0:3306->3306/tcp, 33060/tcp   img-search-mysql
f3c7440d5dc4   milvusbootcamp/img-search-client:1.0          "/bin/bash -c '/usr/…"   About a minute ago   Up 59 seconds (health: starting)   0.0.0.0:8001->80/tcp                img-search-webclient
cc6b473d905d   quay.io/coreos/etcd:v3.5.0                    "etcd -advertise-cli…"   About a minute ago   Up 59 seconds                      2379-2380/tcp                       milvus-etcd
```

You can also try to run  [hello_milvus.py](https://milvus.io/docs/v2.1.x/example_code.md) to make sure Milvus has started successfully and the network port is available.

**Q2:** **Running load in reverse image search but it shows "No image in the set"**

\- First, open the developer mode in your own browser to view the status information of the network.

\- Then run `docker logs img-search-webserver` to get the logs. If the front-end page is refreshed, but the log is not updated, which means the webserver did not receive the request. Try changing "127.0.0.1" in `webclient.environment.API_URL` to the IP address.

**Q3:** **The log shows "FileNotFoundError:No such file or direcctory", but the folder has data** 

The location where the data is loaded in the container is not the local location, so you need to mount the data to the container (modify the `webserver.volumes` in docker-compose.yaml), and then use the path in the container to load the data.

**Q4:**  **The log shows EOF Error**

EOF error usually means that the model download failed, you can check the network and try again.

## Customize

**Q1:** **Can I modify the code that starts the service with docker-compose**

If you want to modify the code, we recommend using the source code method. But if you just want to modify a little code, you can get into the webserver container shell and modify it directly, and then restart the container.

**Q2:** **Updated the model and VECTOR_DIM in the system, but still logs "Dimension not match"**

After modifying the code, please confirm whether it will conflict with the previous Milvus Collection. To avoid it, you can update the collection/table name (`DEFAULT_TABLE` in server/src/config.py).