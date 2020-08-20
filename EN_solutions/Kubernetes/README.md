# Deploy Milvus distributed clusters based on Kubernetes

This example shows how to install shared storage and how to build a Milvus cluster using both Helm and Kubectl.

This example does not include how to build a Kubernetes cluster or how to install Helm.

## Preparation

- Kubernetes 1.10+
- Helm >= 2.12.0
- Docker >= 19.03

## Build shared storage

Shared storage is needed when we want a storage volume in a Kubernetes cluster to be simultaneously mounted by multiple pods and multiple pods modify the same data at the same time. Common protocols for sharing resources include NFS and CIFS. In the following, we will demonstrate how to build NFS storage resources and deploy NFS Server in Kubernetes.

* [**Build NFS Storage Resources with docker**](https://github.com/ehough/docker-nfs-server)

  **server**

  1. Create and run containers

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

     > `/data/nfs` :the directory of the shared folder on the server side.
     
  2. Install NFS model

     ```bash
      $ sudo apt install nfs-kernel-server
     ```

  3. View container status

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

  **client**

  Let's configure and start the client, and check if NFS has been built successfully.

  1. mount

     ```bash
     $ mount -t nfs -o rw,nfsvers=3 192.168.1.31:/nfs /data/nfs
     ```

     > 192.168.1.31 : IP of server
     >
     > `/data/nfs` :the path to the client end mount.

  2. View Mount Information

     ```bash
     $ df -h
     ```

* [**Set StorageClass with Helm**](https://github.com/helm/charts/tree/master/stable/nfs-client-provisioner)

  1. Pull source code

     ```bash
     $ git clone https://github.com/helm/charts.git
     $ cd charts/stable/nfs-client-provisioner
     ```

  2. Install NFS chart

     > [chart](https://github.com/helm/charts) is a pre-configured installer resource, similar to Ubuntu's APT and CentOS's YUM. a release is created when chart is installed into Kubernetes.
     >
     > The NFS Client Provisioner is an automation plug-in for automatic creation of Kubernetes PV. It automatically creates Kubernetes PV based on a configured NFS Server.

     ```bash
     $ vim values.yaml
     # nfs:
     # server: 192.168.1.31
     # path: /nfs
     # mountOptions:
     #   - rw
     #   - nfsvers=3
     $ helm install nfs-client .
     ```
     
  3. Check if the nfs-client release was installed successfully

     ```bash
     $ helm list
     NAME      	NAMESPACE	REVISION	UPDATED                                	STATUS  	CHART                       	APP VERSION
     nfs-client	default  	1       	2020-07-16 16:33:11.528645222 +0800 CST	deployed	nfs-client-provisioner-1.2.8	3.1.0             
     ```


## Deploy Milvus with Helm

1. Pull source code

   ```bash
   $ git clone -b 0.10.0 https://github.com/milvus-io/milvus-helm.git
   $ cd milvus-helm
   ```
   
2. Deploy Milvus

   ```bash
   $ git clone https://github.com/milvus-io/milvus-helm.git
   $ cd milvus-helm
   $ helm install --set cluster.enabled=true --set persistence.enabled=true --set mysql.enabled=true my-release  .
   ```

   > For detailed parameters of the Milvus server, please refer to [Milvus Configuration](https://github.com/milvus-io/milvus-helm/tree/0.10.0#configuration)

3. Check if Milvus release was installed successfully

   ```bash
   $ helm list
   NAME      	NAMESPACE	REVISION	UPDATED                                	STATUS  	CHART                       	APP VERSION
   my-release	default  	1       	2020-07-16 15:04:09.735039543 +0800 CST	deployed	milvus-0.10.0               	0.10.0     
   nfs-client	default  	1       	2020-07-16 14:20:16.652557193 +0800 CST	deployed	nfs-client-provisioner-1.2.8	3.1.0    
   ```

4. Check if pods started successfully

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

   > If any of the pods failed to start, use `kubectl logs <NAME>` or `kubectl describe pod <NAME>` for error checking!
   >
   > For more information on the use of Helm, please refer to [Helm](https://helm.sh/docs/).

## Deploy Milvus with kubectl

The essence of deploying an application using kubectl is to deploy the content defined in the YAML file. Therefore, we need to install the schelm plugin using the go language. The schelm plug-in retrieves the manifest files, which are resource descriptions in YAML format that Kubernetes can recognize.

1. Pull source code

   ```bash
   $ git clone -b 0.10.0 https://github.com/milvus-io/milvus-helm.git
   $ cd milvus-helm
   ```

2. Download Go

   ```bash
   $ wget https://dl.google.com/go/go1.14.6.linux-amd64.tar.gz
   $ sudo tar -C /usr/local -xzf go1.14.6.linux-amd64.tar.gz
   ```
   
3. Add environment variable to `/etc/profile` or `$HOME/.profile`.

   ```bash
   export PATH=$PATH:/usr/local/go/bin
   ```

   > For installation procedures for other systems, please refer to [Install the Go tools](https://golang.org/doc/install).

4. Install schelm

   ```bash
   $ go get -u github.com/databus23/schelm
   $ sh
   sh               sha224sum        sha384sum        shadowconfig     sh.distrib       shopt            showconsolefont  showrgb          shuf             
   sha1sum          sha256sum        sha512sum        shasum           shift            shotwell         showkey          shred            shutdown         
   ```

5. Get manifest files for Milvus

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
   
6. Apply configuration file to pods

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

   > If a format conversion error occurs, please modify the corresponding .yaml file.

7. Check if pods started successfully

   ```bash
   $ kubectl get pods
   ```

8. Check pvc status

   ```bash
   $ kubectl get pvc -A
   NAMESPACE   NAME                                    STATUS   VOLUME                                     CAPACITY   ACCESS MODES   STORAGECLASS   AGE
   default     my-release-milvus                       Bound    pvc-8a5c6706-ccc3-44d0-a13b-e50632aafb01   50Gi       RWX            nfs-client     54s
   default     my-release-mysql                        Bound    pvc-a5599f51-06b9-4743-aacd-1d00f9fd9fe0   4Gi        RWO            nfs-client     29s
   default     pvc-nfs-client-nfs-client-provisioner   Bound    pv-nfs-client-nfs-client-provisioner       10Mi       RWO                           22h
   ```

## Test Cluster

At this point, the Milvus service has been successfully deployed on Kubernetes. However, the default service for Kubernetes is ClusterIP, which can be accessed by other applications within the cluster, but not outside the cluster. So, if we want to use the cluster on the Internet or in a production environment, we need to change the service to expose the application.The two types of Kubernetes services that can expose the service are NodePort and LoadBalancer. In the following, we will explain how to access the cluster externally using the NodePort service.

1. Modify service

   ```bash
   $ vim values.yaml
   # service.type: NodePort
   ```
   
2. Update Milvus release

   ```bash
   $ helm upgrade --set cluster.enabled=true --set persistence.enabled=true --set mysql.enabled=true my-release --set web.enabled=true  .
   ```

3. Check the status of ports

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

   > At this point, Milvus services can be run outside the cluster by accessing port 32227 of Master node or Worker node.
   >
   > For more ways to expose your application, please refer to [Expose Your App Publicly](https://kubernetes.io/docs/tutorials/kubernetes-basics/expose/).

