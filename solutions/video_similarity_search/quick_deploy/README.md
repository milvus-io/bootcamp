# Building a video search system based on Milvus

## Overview

This demo uses ResNet50, an image feature extraction model, and Milvus to build a system that can perform reverse image search.

The system architecture is displayed as follows:



## Data source

This demo uses the PASCAL VOC image set, which contains 17125 images with 20 categories: human; animals (birds, cats, cows, dogs, horses, sheep); transportation (planes, bikes,boats, buses, cars, motorcycles, trains); household (bottles, chairs, tables, pot plants, sofas, TVs)

Dataset size: ~ 2 GB.

Download location: http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar

> Note: You can also use other images for testing. This system supports the following formats: .jpg and .png.

## How to deploy the system

### 1. Start Milvus and MySQL

As shown in the architecture diagram, the system will use Milvus to store and search the feature vector data, and Mysql is used to store the correspondence between the ids returned by Milvus and the image paths, then you need to start Milvus and Mysql first.

- **Start Milvus v1.1.0**

  First, you are supposed to refer to the Install Milvus v1.1.0 for how to run Milvus docker.

  ```
  $ wget -P /home/$USER/milvus/conf https://raw.githubusercontent.com/milvus-io/milvus/v1.1.0/core/conf/demo/server_config.yaml
  ```

  ```
  $ sudo docker run -d --name milvus_cpu_1.1.0 \
  ```

  ```
  -p 19530:19530 \
  ```

  ```
  -p 19121:19121 \
  ```

  ```
  -v /home/$USER/milvus/db:/var/lib/milvus/db \
  ```

  ```
  -v /home/$USER/milvus/conf:/var/lib/milvus/conf \
  ```

  ```
  -v /home/$USER/milvus/logs:/var/lib/milvus/logs \
  ```

  ```
  -v /home/$USER/milvus/wal:/var/lib/milvus/wal \
  ```

  ```
  milvusdb/milvus:1.1.0-cpu-d050721-5e559c
  ```

  > Note the version of Milvus.

  - **Start MySQL**

    ```
    $ docker run -p 3306:3306 -e MYSQL_ROOT_PASSWORD=123456 -d mysql:5.7
    ```

    ### 2. Start Server

    The next step is to start the system server. It provides HTTP backend services, and there are two ways to start, such as Docker and source code.

    #### 2.1 Run server with Docker

    - **Set parameters**

      Please modify the parameters according to your own environment. Here listing some parameters that need to be set, for more information please refer to [config.py](./server/src/config.py).

      | **Parameter**   | **Description**                                       | **example**      |
      | --------------- | ----------------------------------------------------- | ---------------- |
      | **DATAPATH1**   | The dictionary of the image path.                     | /data/image_path |
      | **MILVUS_HOST** | The IP address of Milvus, you can get it by ifconfig. | 192.168.1.85     |
      | **MILVUS_PORT** | The port of Milvus.                                   | 19530            |

      ```
      $ export DATAPATH1='/data/image_path'
      ```

      ```
      $ export Milvus_HOST='192.168.1.85'
      ```

      ```
      $ export Milvus_PORT='19530'
      ```

      - **Run Docker**

        ```
        $ docker run -d \
        ```

        ```
        -v ${DATAPATH1}:${DATAPATH1} \
        ```

        ```
        -p 5000:5000 \
        ```

        ```
        -e "MILVUS_HOST=${MILVUS_IP}" \
        ```

        ```
        -e "MILVUS_PORT=${MILVUS_PORT}" \
        ```

        ```
        milvusbootcamp/pic-search-webserver:2.0
        ```

        > **Note:** -v ${DATAPATH1}:${DATAPATH1} means that you can mount the directory into the container. If needed, you can load the parent directory or more directories.

        #### 2.2 Run source code

        - **Install the Python packages**

          ```
          $ cd server
          ```

          ```
          $ pip install -r requirements.txt
          ```

          - **Set configuration**

            ```
            $ vim server/src/config.py
            ```

            Please modify the parameters according to your own environment. Here listing some parameters that need to be set, for more information please refer to [config.py](./server/src/config.py).

            | **Parameter**    | **Description**                                       | **Default setting** |
            | ---------------- | ----------------------------------------------------- | ------------------- |
            | MILVUS_HOST      | The IP address of Milvus, you can get it by ifconfig. | 127.0.0.1           |
            | MILVUS_PORT      | Port of Milvus.                                       | 19530               |
            | VECTOR_DIMENSION | Dimension of the vectors.                             | 2048                |
            | MYSQL_HOST       | The IP address of Mysql.                              | 127.0.0.1           |
            | MYSQL_PORT       | Port of Milvus.                                       | 3306                |
            | DEFAULT_TABLE    | The milvus and mysql default collection name.         | milvus_img_search   |

            - **Run the code** 

            Then start the server with Fastapi. 

            ```
            $ cd src
            ```

            ```
            $ python main.py
            ```

            - **The API docs**

              Type 127.0.0.1:5000/docs in your browser to see all the APIs.

              [img] (API_imag.png)

              - **Code  structure**

                If you are interested in our code or would like to contribute code, feel free to learn more about our code structure.

                ```
                └───server
                ```

                ```
                │   │   Dockerfile
                ```

                ```
                │   │   requirements.txt
                ```

                ```
                │   │   main.py  # File for starting the program.
                ```

                ```
                │   │
                ```

                ```
                │   └───src
                ```

                ```
                │       │   config.py  # Configuration file.
                ```

                ```
                │       │   encode.py  # Covert image/video/questions/... to embeddings.
                ```

                ```
                │       │   milvus.py  # Connect to Milvus server and insert/drop/query vectors in Milvus.
                ```

                ```
                │       │   mysql.py   # Connect to MySQL server, and add/delete/query IDs and object information.
                ```

                ```
                │       │   
                ```

                ```
                │       └───operations # Call methods in milvus.py and mysql.py to insert/query/delete objects.
                ```

                ```
                │               │   insert.py
                ```

                ```
                │               │   query.py
                ```

                ```
                │               │   delete.py
                ```

                ```
                │               │   count.py
                ```

                ### 3. Start Client

                - **Start the front-end**

                  ```
                  # Please modify API_URL to the IP address and port of the server.
                  $ export API_URL='http://192.168.1.85:5000'
                  $ docker run -d -p 8001:80 \
                  -e API_URL=${API_URL} \
                  milvusbootcamp/pic-search-webclient:1.0
                  ```

                  - **How to use**

                    Enter `WEBCLIENT_IP:8001`  in the browser to open the interface for reverse image search. 

                    > `WEBCLIENT_IP`specifies the IP address that runs pic-search-webclient docker.

                    ![arch](file:///Users/chenshiyu/workspace/git/mia/bootcamp/solutions/image_similarity_search/quick_deploy/pic/web4.png?lastModify=1623839861)

                    Enter the path of an image folder in the pic_search_webserver docker container with `${DATAPATH1}`, then click `+` to load the pictures. The following screenshot shows the loading process:

                    ![arch](file:///Users/chenshiyu/workspace/git/mia/bootcamp/solutions/image_similarity_search/quick_deploy/pic/web2.png?lastModify=1623839861)

                    > Note: After clicking the Load button, it will take 1 to 2 seconds for the system to response. Please do not click again.

                    The loading process may take several minutes. The following screenshot shows the interface with images loaded.

                    ![arch](file:///Users/chenshiyu/workspace/git/mia/bootcamp/solutions/image_similarity_search/quick_deploy/pic/web3.png?lastModify=1623839861)

                    Select an image to search.

                    ![arch](file:///Users/chenshiyu/workspace/git/mia/bootcamp/solutions/image_similarity_search/quick_deploy/pic/web5.png?lastModify=1623839861)

                    