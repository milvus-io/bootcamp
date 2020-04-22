# Chemical Structure Similarity Search

## Environment

| Component     | Recommended Configuration                                                     |
| -------- | ------------------------------------------------------------ |
| CPU      | Intel(R) Core(TM) i7-7700K CPU @ 4.20GHz                     |
| Memory   | 32 GB                                                         |
| OS       | Ubuntu 18.04                                                 |
| Software | [Milvus 0.7.0](https://milvus.io/cn/docs/v0.7.0/guides/get_started/install_milvus/cpu_milvus_docker.md) <br />mols-search-webserver 0.4.0 <br />mols-search-webclient 0.3.0 |

The previous configuration has been tested and this scenario is also supported in Windows.

## Data preparation

Data source: [ftp://ftp.ncbi.nlm.nih.gov/pubchem/Compound/CURRENT-Full/SDF](ftp://ftp.ncbi.nlm.nih.gov/pubchem/Compound/CURRENT-Full/SDF). The data source contains compressed SDF files. You need to convert these files to SMILES files. We already prepared a SMILE file containing 10,000 chemical structures [test_1w.smi](../../solutions/mols_search/smiles-data). You can use wget to download the file:

```bash
$ wget https://raw.githubusercontent.com/milvus-io/bootcamp/0.7.0/solutions/mols_search/smiles-data/test_1w.smi
```

## Deploy

#### 1. Run Milvus Docker

This demo uses Milvus 0.7.0 CPU version. Refer to https://milvus.io/cn/docs/v0.7.0/guides/get_started/install_milvus/cpu_milvus_docker.md to learn how to install and run Milvus. 

**Note：Please use the following command to run Milvus:**

```bash
# Start Milvus
$ docker run -d --name milvus_cpu \
-p 19530:19530 \
-p 19121:19121 \
-p 9091:9091 \
-v /home/$USER/milvus/db:/var/lib/milvus/db \
-v /home/$USER/milvus/conf:/var/lib/milvus/conf \
-v /home/$USER/milvus/logs:/var/lib/milvus/logs \
-v /home/$USER/milvus/wal:/var/lib/milvus/wal \
milvusdb/milvus:0.7.0-cpu-d031120-de409b
```

#### 2. Run mols-search-webserver docker

```bash
$ docker run -d -v <DATAPATH>:/tmp/data -p 35001:5000 -e "MILVUS_HOST=192.168.1.25" -e "MILVUS_PORT=19530" milvusbootcamp/mols-search-webserver:0.4.0
```

Refer to the following table for detailed parameter description:

| Parameter                     | Description                                                      |
| ----------------------------- | ------------------------------------------------------------ |
| -v DATAPATH:/tmp/data       | -v specifies directory mapping between the host and the docker image. Please change DATAPATH to the location of test_1w.smi. |
| -p 35001:5000                 | -p specifies pot mapping between the host and the image.                        |
| -e "MILVUS_HOST=192.168.1.25" | -e specifies the system parameter mapping between the host and the image. Pease update `192.168.1.25` to the IP address of the Milvus docker.|
| -e "MILVUS_PORT=19530"        | Update `19530` to the port of Milvus docker.           |



#### 3. Run mols-search-webclient docker

```bash
$ docker run -d --rm -p 8001:80 -e API_URL=http://192.168.1.25:35001 milvusbootcamp/mols-search-webclient:0.3.0
```

> Note: Please update `192.168.1.25` to the IP address of the Milvus docker.


#### 4. Launch a browser

```bash
# Please update IP address and port per your previous configurations
http://192.168.1.25:8001
```

## How to use

- Initial interface

![](../../solutions/mols_search/assert/init_status.PNG)

- Load chemical structures
  1. In `path/to/your/data`, enter the location of the smi file. For example, `/tmp/data/test_1w.smi`.
  2. Click `+` to load.
  3. You can see the number of chemical structures have changed: 10000 Molecular Formula in this set

![](../../solutions/mols_search/assert/load_data.PNG)

- Search chemical structures
  1. Enter the chemical structure to search, such as `Cc1ccc(cc1)S(=O)(=O)N`, and press \<ENTER\>.
  2. Set the value of topk. This demo returns topk most similar chemical structures.

![](../../solutions/mols_search/assert/search_data.PNG)

- Clear chemical structure data

  Click `CLEAR ALL` to remove all chemical structure data.

![](../../solutions/mols_search/assert/delete_data.PNG)


## Conclusion

This demo implements chemical structure similarity search with Milvus. You can also use your own SMILES data. For 37 million chemical structures/90 million structures (512-dimensional feature vector), Milvus has the following performance:

|                                                              | Performance (37 million) | Performance (90 million) |
| :----------------------------------------------------------- | :------------------------- | :--------------------- |
| Search a single chemical structure                                       | 190 ms                      | 480 ms                  |
| Search 50 chemical structures. The performance is the average time of each structure.| 6 ms                        | 12 ms       |

The following table shows memory usage:

|                 | 37 million | 90 million |
| :-------------- | :----------------- | :------------- |
| Memory Usage | 2.6 G               | 6 G             |

We can see that Milvus has faster search speed and lower memory usage for massive-scale chemical structure search.

We have built a chemical structure similarity search system (http://40.73.24.85) based on Milvus. Welcome to search for your own chemical structure.
