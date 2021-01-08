# 使用Nginx实现对Milvus的负载均衡

本示例主要展示如何利用Nginx实现对Milvus的负载均衡，Nginx最常见的功能是服务器的负载均衡配置，通过使用Nginx分发请求，将不同请求分发给多个Milvus中，从而实现负载均衡。

## 前期准备

本示例至少需要两台设备和一个共享存储设备，基于0.10.4版本的Milvus的搭建

1、Nginx 1.18.0

2、至少两个以上的Milvus读节点和一个Milvus的写节点

## 实现步骤

### 一、Nginx 安装

1、首先，从[Nginx官网](http://nginx.org/en/download.html)下载软件压缩包，并进行解压

```
wget http://nginx.org/download/nginx-1.18.0.tar.gz
tar -zxvf nginx-1.18.0.tar.gz
cd nginx-1.18.0
```

2、进入到解压软件目录之后，需要安装依赖库

```
更新源
sudo apt-get update
安装C++依赖库
sudo apt-get install build-essential
sudo apt-get install libtool
安装openssl依赖库
sudo apt-get install openssl
安装pcre依赖库
sudo apt-get install libpcre3 libpcre3-dev
安装zlib依赖库
sudo apt-get install zlib1g-dev 
```

3、编译前进行配置，将nginx安装到**/usr/local/ngnix**目录

```
./configure --prefix=/usr/local/nginx
 注：--prefix：配置安装目录
```

4、进行编译和安装，如果没有权限的话需要切换到Root用户

```
#su root
make  编译
make install 安装
```

5、使用下面命令启动nginx，然后测试ngnix是否安装成功

```
/usr/local/nginx/sbin/nginx 
```

查看是否有nginx进程启动，有进程启动，说明安装成功了

```
ps -ef | grep nginx
```

当然也可以再浏览器中输入*<localhost(本机的IP地址)：端口（默认端口是80）>*，看到welcome to nginx代表安装成功

![](1.png)

## 二、Milvus配置

1、本示例中Milvus的版本为0.10.4，详细安装方式参考[Milvus官网](https://www.milvus.io/cn/docs/v0.11.0/milvus_docker-gpu.md)，在安装时需要将所有设备数据存储位置都应设置为共享存储的路径，如下图所示。![](2.png)

其中**/cifs/test/nfs/milvus/db**为共享存储的路径

2、然后[使用Mysql管理元数据](https://www.milvus.io/cn/docs/v0.10.4/data_manage.md)，安装Mysql完成之后，需要在Milvus的配置文件server_config.yaml中修改参数meta_uri。 服务器1的ip地址为192.168.1.85，服务器2的ip地址为172.16.10.1，我们在服务器1上安装两个Milvus ，Milvus 1设置可读 ，IP地址为192.168.1.85:19537和Milvus3设置为可写192.168.1.85:19539，服务器2安装Milvus2设置为可读，ip地址为172.16.10.1:19538。在server_config.yaml 配置文件中修改参数enable和role，如下图所示。![](3.png)

其中参数 `enable` 表示是否设置为分布式模式，参数 `role` 决定了 Milvus 是只读还是可写，参数 `meta_uri` 应修改为 MySQL 所安装的设备的地址，其余配置参照 Milvus 单机版时的配置。

注：**修改完配置文件之后需要重启Milvus**

## 三、Nginx配置

完成milvus配置之后，需要修改Nginx的配置文件，配置文件的位置为 **/usr/local/nginx/conf/nginx.conf**，在配置文件的末尾添加如下代码，本示例中使用的负载均衡策略为**轮询**

```
stream {
    log_format proxy '$remote_addr [$time_local] '
                 '$protocol $status $bytes_sent $bytes_received '
                 '$session_time "$upstream_addr" '
                 '"$upstream_bytes_sent" "$upstream_bytes_received" "$upstream_connect_time"';
    # 日志格式配置
    access_log /var/log/nginx/access.log proxy ;
    open_log_file_cache off; #日志缓存设置，此处设置为禁用
    server {
       listen 19585; # 为监听的端口号不能和Milvus的端口号冲突
       proxy_pass milvus;

    }
    upstream milvus {
       server 192.168.1.85:19537;为milvus1的Ip地址
       server 172.16.70.1:19538;为Milvus2的Ip地址
    }

}

```

*注：代码不能写在**http{}**内部*

2、修改完Nginx配置文件之后，测试配置文件是否配置正确，先停止Nginx服务之后再重新启动

```
/usr/local/nginx/sbin/nginx -t          #测试配置文件是否正确
/usr/local/nginx/sbin/nginx  -s stop    #停止Nginx服务
/usr/local/nginx/sbin/nginx             #启动Nginx服务
```

3、建立一个虚拟环境，安装Milvus对应版本的pymilvus ，运行**python3 test.py**脚本进行测试，然后使用如下命令查看nginx的日志文件

```
tail /var/log/nginx/access.log
```

其中test.py脚本中连接Milvus的Ip地址和端口需要修改为nginx配置的IP地址和端口

```
client = Milvus(host='localhost', port='19585')
```

最后，得到ngnix日志查询结果如下图所示，可以看到nginx将请求分给不同的Milvus

![](\4.png)



