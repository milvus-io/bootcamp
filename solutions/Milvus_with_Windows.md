# Windows 环境源码编译 Milvus 0.6.0

## Windows Docker 安装

Milvus 主要是在 Ubuntu 环境下进行开发的，在0.6.0之前的版本都是使用 GPU 加速的版本，Milvus 0.6.0提供纯 CPU 版本，可以让使用者在无 GPU 的机器上运行 Milvus，在 Windows 环境中可以利用 docker 容器源码编译运行 Milvus 0.6.0 。

Windows Docker 安装参考 https://www.runoob.com/docker/windows-docker-install.html , 为确保后续在 Docker 容器中成功编译运行 Milvus ，请前往 Docker Settings -> Advanced 中修改 CPUs/Memory/Swap 参数，建议参数值：

| CPUs       | 4          |
| ---------- | ---------- |
| **Memory** | **4096MB** |
| **Swap**   | **2048MB** |



## 在 Docker 容器中编译运行 Milvus

成功安装 Windows Docker 后，可以在 Docker 容器中编译运行 Milvus，参考 https://milvus.io/blog/2019/11/25/docker-compilation/ , 但是由于 Windows Docker 无法使用 nvidia-docker，即无法使用 GPU，所以在 Windows 下只能使用 Milvus 0.6.0 CPU 版本，在参考文章中应注意步骤一和步骤二。



## Milvus 运行示例程序

Milvus 编译完成后，可以运行示例程序 https://github.com/milvus-io/docs/blob/0.6.0/userguide/example_code.md , 当出现 `Query result is correct`，表示在 Windows 环境成功编译运行 Milvus ！