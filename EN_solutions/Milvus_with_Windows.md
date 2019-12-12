# Build Milvus 0.6.0 from source in Windows

## Install Windows Docker

Milvus is developed in Ubuntu. Before 0.6.0, Milvus requires GPU acceleration. In 0.6.0, Milvus supports CPU-only mode, which supports running Milvus without a GPU. You can use docker to build Milvus 0.6.0 from source in Windows.

Refer to [https://www.runoob.com/docker/windows-docker-install.html](https://www.runoob.com/docker/windows-docker-install.html) to learn how to install Windows Docker. To ensure that Milvus can be successfully compiled and run in Docker, navigate to **Docker Settings** -> **Advanced** to edit the parameters for CPUs/Memory/Swap. The following values are recommended:

| CPUs       | 4          |
| ---------- | ---------- |
| **Memory** | **4096MB** |
| **Swap**   | **2048MB** |



## Compile and run Milvus in Windows Docker

After installing Windows Docker, you can compile and run Milvus in Docker. Refer to [https://milvus.io/blog/2019/11/25/docker-compilation/](https://milvus.io/blog/2019/11/25/docker-compilation/) for more information. However, because Windows Docker does not support nvidia-docker, you can only use the CPU-only version in Windows Docker.

## Run Milvus example program

After finish compiling Milvus, run the [example program](https://github.com/milvus-io/docs/blob/0.6.0/userguide/example_code.md). When `Query result is correct` is displayed, you can assume that the compilation is successful.
