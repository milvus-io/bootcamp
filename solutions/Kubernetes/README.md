# 基于 Kubernetes 部署 Milvus 分布式集群

本示例主要展示如何安装共享存储，如何利用 Helm 和 Kubectl 两种方式搭建 Milvus 集群。

本示例不包含如何搭建 Kubernetes 集群，如何安装 Helm。

## 环境准备

- Kubernetes 1.10+
- Helm >= 2.12.0
- Docker >= 19.03

## 搭建共享存储

如果我们希望在 Kubernetes 集群中一个存储卷可以被多个 Pod 同时挂载，多个 Pod 同时修改相同数据，这时便需要共享存储。目前常见的共享资源协议有 NFS 和 CIFS 等。下面，我们将演示如何搭建 NFS 存储资源并在 Kubernetes 中部署 NFS Server。

* [**利用 docker 搭建 NFS 存储资源**](https://github.com/ehough/docker-nfs-server)

  **server端**

  1. 创建容器并运行

     ```bash
     $ docker run -d --privileged --restart=always \
     -v /data/usr/nfs_test:/nfs  \
     -e NFS_EXPORT_0='/nfs                  *(rw,no_subtree_check,no_root_squash,fsid=1)' \
     -p 111:111 \
     -p 111:111/udp \
     -p 2049:2049 \
     -p 2049:2049/udp \
     -p 32765:32765 \
     -p 32765:32765/udp \
     -p 32766:32766 \
     -p 32766:32766/udp \
     -p 32767:32767 \
     -p 32767:32767/udp \
     --cap-add SYS_ADMIN \
     erichough/nfs-server
     ```

     > `/data/nfs` 为 server 端的共享文件夹的目录
     
  2. 安装 NFS 模型

     ```bash
      $ sudo apt install nfs-kernel-server
     ```

  3. 查看容器状态

     ```bash
     $ sudo docker logs c8a4abde5401
     ......
     ==================================================================
           SERVER STARTUP COMPLETE
     ==================================================================
     ----> list of enabled NFS protocol versions: 4.2, 4.1, 4, 3
     ----> list of container exports:
     ---->   /nfs                  *(rw,no_subtree_check,no_root_squash,fsid=1)
     ----> list of container ports that should be exposed:
     ---->   111 (TCP and UDP)
     ---->   2049 (TCP and UDP)
     ---->   32765 (TCP and UDP)
     ---->   32767 (TCP and UDP)
     
     ==================================================================
           READY AND WAITING FOR NFS CLIENT CONNECTIONS
     ==================================================================
     ```

  **client端**

  下面我们配置并启动客户端，检查 NFS 是否搭建成功。

  1. 挂载

     ```bash
     $ mount -t nfs -o rw,nfsvers=3 192.168.1.31:/nfs /data/nfs
     ```

     > 192.168.1.31 为 server 端 IP
     >
     > `/data/nfs` 为 client 端挂载路径

  2. 查看挂载信息

     ```bash
     $ df -h
     ```

* [**利用 Helm 配置 StorageClass**](https://github.com/helm/charts/tree/master/stable/nfs-client-provisioner)

  1. 拉取源码

     ```bash
     $ git clone https://github.com/helm/charts.git
     $ cd charts/stable/nfs-client-provisioner
     ```

  2. 安装 NFS client chart

     > [chart](https://github.com/helm/charts) 为预先配置好的安装包资源，类似于 Ubuntu 的 APT 和 CentOS 中的 YUM。当 chart 安装到 Kubernetes 中后就会创建一个 release。
     >
     > NFS Client Provisioner 是用于自动创建 Kubernetes PV 的自动化插件。它可以根据已配置好的 NFS Server，自动创建 Kubernetes PV。

     ```bash
     # 修改 values.yaml 下 内容：
     # nfs:
     # server: 192.168.1.31
     # path: /nfs
     # mountOptions:
     #   - rw
     #   - nfsvers=3
     $ helm install nfs-client .
     ```
     
  3. 检查部署状态：

     ```bash
     $ helm list
     NAME      	NAMESPACE	REVISION	UPDATED                                	STATUS  	CHART                       	APP VERSION
     nfs-client	default  	1       	2020-07-16 16:33:11.528645222 +0800 CST	deployed	nfs-client-provisioner-1.2.8	3.1.0             
     ```


## 利用 Helm 部署 Milvus

1. 拉取源码

   ```bash
   $ git clone -b 0.10.0 https://github.com/milvus-io/milvus-helm.git
   $ cd milvus-helm
   ```
   
2. 部署 Milvus

   ```bash
   $ helm install --set cluster.enabled=true --set persistence.enabled=true --set mysql.enabled=true my-release  .
   ```

   > 关于 Milvus 服务器的详细参数，可参考 [Milvus Configuration](https://github.com/milvus-io/milvus-helm/tree/0.10.0#configuration)

3. 查看 Milvus release 是否安装成功：

   ```bash
   $ helm list
   NAME      	NAMESPACE	REVISION	UPDATED                                	STATUS  	CHART                       	APP VERSION
   my-release	default  	1       	2020-07-16 15:04:09.735039543 +0800 CST	deployed	milvus-0.10.0               	0.10.0     
   nfs-client	default  	1       	2020-07-16 14:20:16.652557193 +0800 CST	deployed	nfs-client-provisioner-1.2.8	3.1.0    
   ```

4. 查看 Pods 是否启动成功：

   ```bash
   $ kubectl get pods
   # You are expected to see the following output.
   NAME                                                READY   STATUS    RESTARTS   AGE
   my-release-milvus-mishards-8f97db7fb-qxxgn          1/1     Running   0          12m
   my-release-milvus-readonly-66784bccd6-67wcr         1/1     Running   0          12m
   my-release-milvus-writable-55d7ff788b-n4zc6         1/1     Running   1          12m
   my-release-mysql-8688668cd-2rj7k                    1/1     Running   1          12m
   nfs-client-nfs-client-provisioner-86cf7c4bc-hd7bq   1/1     Running   3          32m
   ```

   > 如果有 pods 未启动成功，请使用 `kubectl logs <NAME>` 或 `kubectl describe pod <NAME>` 进行错误排查
   >
   > 更多关于 Helm 的使用，请参考 [Helm 官方文档](https://helm.sh/docs/)。

## 利用 kubectl 部署 Milvus

利用 kubectl 部署应用的实质便是部署 yaml 或 json 文件中定义的内容。因此我们需要利用 Go 安装 schelm 插件。通过 schelm 插件获得 manifest 文件，它们即为 Kubernetes 可以识别的 yaml 格式的资源描述。

1. 拉取源码

   ```bash
   $ git clone -b 0.10.0 https://github.com/milvus-io/milvus-helm.git
   $ cd milvus-helm
   ```

2. 下载并解压 GO 

   ```bash
   $ wget https://dl.google.com/go/go1.14.6.linux-amd64.tar.gz
   $ sudo tar -C /usr/local -xzf go1.14.6.linux-amd64.tar.gz
   ```
   
3. 在 `/etc/profile` 或者 `$HOME/.profile` 添加环境变量

   ```bash
   export PATH=$PATH:/usr/local/go/bin
   ```

   > 其他系统的安装流程，请参考 [Install the Go tools](https://golang.org/doc/install) 。

4. 安装 schelm 插件

   ```bash
   $ go get -u github.com/databus23/schelm
   $ sh
   sh               sha224sum        sha384sum        shadowconfig     sh.distrib       shopt            showconsolefont  showrgb          shuf             
   sha1sum          sha256sum        sha512sum        shasum           shift            shotwell         showkey          shred            shutdown         
   ```

5. 获取 Milvus 的 manifest 文件

   ```bash
   $ helm install --dry-run --debug --set web.enabled=true --set cluster.enabled=true --set persistence.enabled=true --set mysql.enabled=true my-release  . | ~/go/bin/schelm output/
   # You are expected to see the following output.
   install.go:172: [debug] Original chart version: ""
   install.go:189: [debug] CHART PATH: /home/mia/milvus-helm/charts/milvus
   
   2020/07/17 14:51:33 Creating output/milvus/charts/mysql/templates/secrets.yaml
   2020/07/17 14:51:33 Creating output/milvus/charts/mysql/templates/configurationFiles-configmap.yaml
   2020/07/17 14:51:33 Creating output/milvus/charts/mysql/templates/initializationFiles-configmap.yaml
   2020/07/17 14:51:33 Creating output/milvus/templates/config.yaml
   2020/07/17 14:51:33 Creating output/milvus/charts/mysql/templates/pvc.yaml
   2020/07/17 14:51:33 Creating output/milvus/templates/pvc.yaml
   2020/07/17 14:51:33 Creating output/milvus/charts/mysql/templates/svc.yaml
   2020/07/17 14:51:33 Creating output/milvus/templates/mishards-svc.yaml
   2020/07/17 14:51:33 Creating output/milvus/templates/readonly-svc.yaml
   2020/07/17 14:51:33 Creating output/milvus/templates/writable-svc.yaml
   2020/07/17 14:51:33 Creating output/milvus/charts/mysql/templates/deployment.yaml
   2020/07/17 14:51:33 Creating output/milvus/templates/mishards-deployment.yaml
   2020/07/17 14:51:33 Creating output/milvus/templates/readonly-deployment.yaml
   2020/07/17 14:51:33 Creating output/milvus/templates/writable-deployment.yaml
   ```
   
6. 将配置文件应用到 Pod

   ```bash
   $ cd output/milvus/
   $ kubectl apply -f templates/
   # You are expected to see the following output.
   configmap/my-release-milvus created
   deployment.apps/my-release-milvus-mishards created
   service/my-release-milvus created
   persistentvolumeclaim/my-release-milvus created
   deployment.apps/my-release-milvus-readonly created
   service/my-release-milvus-readonly created
   deployment.apps/my-release-milvus-writable created
   service/my-release-milvus-writable created
   $ cd /charts/mysql/
   $ kubectl apply -f templates/
   # You are expected to see the following output.
   configmap/my-release-mysql-configuration created
   deployment.apps/my-release-mysql created
   configmap/my-release-mysql-initialization created
   persistentvolumeclaim/my-release-mysql created
   secret/my-release-mysql created
   service/my-release-mysql created
   ```

   > 如果出现格式转换错误，请修改对应 .yaml 文件

7. 查看 Pods 是否成功启动

   ```bash
   $ kubectl get pods
   ```

8. 查看 PVC

   ```bash
   $ kubectl get pvc -A
   NAMESPACE   NAME                                    STATUS   VOLUME                                     CAPACITY   ACCESS MODES   STORAGECLASS   AGE
   default     my-release-milvus                       Bound    pvc-8a5c6706-ccc3-44d0-a13b-e50632aafb01   50Gi       RWX            nfs-client     54s
   default     my-release-mysql                        Bound    pvc-a5599f51-06b9-4743-aacd-1d00f9fd9fe0   4Gi        RWO            nfs-client     29s
   default     pvc-nfs-client-nfs-client-provisioner   Bound    pv-nfs-client-nfs-client-provisioner       10Mi       RWO                           22h
   ```

## 集群测试

此时，Milvus 服务已成功部署到 Kubernetes 上。但是，Kubernetes 的默认服务为ClusterIP，集群内的其它应用可以访问该服务，而集群外部无法进行访问。所以，如果我们想在 Internet 或者生产环境中使用集群，我们需要更换 Service 以暴露应用。Kubernetes的两种可以暴露服务的 Service 类型为：NodePort 和 LoadBalancer。下面我们将介绍如何使用 NodePort 服务在外部访问集群。                         

1. 修改服务方式

   ```bash
   $ vim values.yaml
   ```
   
   修改 `service.type` 参数为NodePort
   
2. 更新 Milvus release

   ```bash
   $ helm upgrade --set cluster.enabled=true --set persistence.enabled=true --set mysql.enabled=true my-release --set web.enabled=true  .
   ```

3. 查看此时端口状态

   ```bash
   $ kubectl get service
   # You are expected to see the following output.
   NAME                         TYPE        CLUSTER-IP     EXTERNAL-IP   PORT(S)               AGE
   kubernetes                   ClusterIP   10.96.0.1      <none>        443/TCP               24h
   my-release-milvus            NodePort    10.99.64.80    <none>        19530:32227/TCP       30m
   my-release-milvus-readonly   ClusterIP   10.99.29.32    <none>        19530/TCP,19121/TCP   30m
   my-release-milvus-writable   ClusterIP   10.98.84.247   <none>        19530/TCP,19121/TCP   30m
   my-release-mysql             ClusterIP   10.97.182.37   <none>        3306/TCP              30m
   ```

   > 此时，在集群外部便可以通过访问 master 节点或 node 节点的32227端口来运行 Milvus 服务。
   >
   > 关于更多暴露应用的方法，请参考 [Expose Your App Publicly](https://kubernetes.io/docs/tutorials/kubernetes-basics/expose/)

