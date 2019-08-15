#  Milvus Quick Start

In this guide, we will walk you through installing Milvus and your very first vector search Python codes with it. To learn more about how to use Milvus, please visit [Milvus Docker User Guide](./userguide/preface.md).

## Prerequisites

1. Milvus installation is currently supported on Linux systems, make sure one of the following Linux distributions is used:

   | Linux operation system | Supported versions          |
   | :--------------------- | :--------------- |
   | CentOS                 | 7.5 and higher   |
   | Ubuntu LTS             | 16.04 and higher |

2. Make sure following software packages are installed so that Milvus can deployed through Docker:

   - [CUDA 9.0 and higher]( https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html)
   - [Docker CE]( https://docs.docker.com/install/)
   - [NVIDIA-Docker2](https://github.com/NVIDIA/nvidia-docker)

## Installing Milvus Docker

1. Download Milvus Docker image.

   ```shell
   # Download Milvus Docker image
   $ docker pull milvusdb/milvus:0.3.1
   ```

2. Create Milvus file, and add server_cofig and log_config to it.

   ```shell
   # Create Milvus file
   $ mkdir /home/$USER/milvus
   $ cd /home/$USER/milvus
   $ mkdir conf
   $ cd conf
   $ wget https://raw.githubusercontent.com/milvus-io/docs/branch-0.3.1/assets/server_config.yaml
   $ wget https://raw.githubusercontent.com/milvus-io/docs/branch-0.3.1/assets/log_config.conf
   
   ```

3. Start Milvus server.

   ```shell
   # Start Milvus
   $ nvidia-docker run -td --runtime=nvidia -p 19530:19530 -p 8080:8080 -v /home/$USER/milvus/db:/opt/milvus/db -v /home/$USER/milvus/conf:/opt/conf -v /home/$USER/milvus/logs:/opt/milvus/logs milvusdb/milvus:0.3.1

   ```

4. Get Milvus container id.

   ```shell
   # Get Milvus container id
   $ docker ps -a
   ```

5. Confirm Milvus running status.

   ```shell
   # Make sure Milvus is up and running
   $ docker logs <milvus container id>
   ```

## Running Python example program

Now, let's run a Python example program. You will need to create a vector data table, insert 10 vectors, and then run a vector similarity search.

1. Make sure [Python 3.4](https://www.python.org/downloads/) or higher is already installed and in use.

2. Install Milvus Python SDK.

   ```shell
   # Install Milvus Python SDK
   $ pip install pymilvus==0.1.24
   ```

   > Note: To learn more about Milvus Python SDK, go to [Milvus Python SDK Playbook](https://pypi.org/project/pymilvus). 

3. Create a new file *example.py*, and add [Python example code](https://github.com/milvus-io/pymilvus/blob/branch-0.3.1/examples/AdvancedExample.py) to it.

4. Run the example code.

   ```shell
   # Run Milvus Python example
   $ python3 example.py
   ```

5. Confirm the program is running correctly.


Congratulations! You have successfully completed your first vector similarity search with Milvus.

