# Deploy Milvus on KubeSphere

Milvus is an open-source vector database for unstructured data.

This tutorial demonstrates how to deploy Milvus on KubeSphere.

## Prerequisites

- To install KubeSphere 3.2.1 on Kubernetes, your Kubernetes version must be v1.19.x, v1.20.x, v1.21.x or v1.22.x (experimental)
- Make sure your machine meets the minimal hardware requirement: CPU >= 16 Core, Memory >= 64 GB.
- A default Storage Class in your Kubernetes cluster needs to be configured before the installation.

## Step 1: Install KubeSphere

After you make sure your machine meets the conditions, perform the following steps to install KubeSphere.

1. Run the following commands to start installation:

   ```shell
   kubectl apply -f https://github.com/kubesphere/ks-installer/releases/download/v3.2.1/kubesphere-installer.yaml
      
   kubectl apply -f https://github.com/kubesphere/ks-installer/releases/download/v3.2.1/cluster-configuration.yaml
   ```

2. After KubeSphere is successfully installed, you can run the following command to view the installation logs:

   ```shell
   kubectl logs -n kubesphere-system $(kubectl get pod -n kubesphere-system -l app=ks-install -o jsonpath='{.items[0].metadata.name}') -f
   ```

3. Use `kubectl get pod --all-namespaces` to see whether all Pods are running normally in relevant namespaces of KubeSphere. If they are, check the port (`30880` by default) of the console by running the following command:

   ```shell
   kubectl get svc/ks-console -n kubesphere-system
   ```

4. Make sure port `30880` is opened in your security group and access the web console through the NodePort (`IP:30880`) with the default account and password (`admin/P@88w0rd`).

5. After logging in to the console, you can check the status of different components in **System Components**. You may need to wait for some components to be up and running if you want to use related services.

## Step 2: Enable the App Store

1. Log in to the console as `admin`. Click **Platform** in the top-left corner and select **Cluster Management**.
2. Click **CRDs** and enter `clusterconfiguration` in the search bar. Click the result to view its detail page.
3. In **Resource List**, click ![three-dots.png](https://kubesphere.io/images/docs/enable-pluggable-components/kubesphere-app-store/three-dots.png) on the right of `ks-installer` and select **Edit YAML**.
4. In this YAML file, navigate to `openpitrix` and change `false` to `true` for `enabled`. After you finish, click **Update** in the bottom-right corner to save the configuration.

```yaml
openpitrix:
  store:
    enabled: true # Change "false" to "true".
```

## Step3: Create User, workspace and project

The multi-tenant system of KubeSphere features three levels of hierarchical structure which are cluster, workspace, and project. A project in KubeSphere is a Kubernetes namespace.

1. **Create a User**： Click **Platform** in the upper-left corner, and then select **Access Control**. In the left nevigation pane, select **Users**, and click **Create**. In the displayed dialog box, provide all the necessary information (marked with *) and select `platform-regular` for **Role**. Refer to the following image as an example.

![image-20220318172633333](/Users/bennu/Library/Application Support/typora-user-images/image-20220318172633333.png)

2. **Create a workspace**: Select **workspace** in the navigation bar at Access Control, then click **create**, and set the user `milvus-manager` as the workspace Administrator.

![image-20220318172745769](/Users/bennu/Library/Application Support/typora-user-images/image-20220318172745769.png)

3. **Create a project**: Log in to KubeSphere as `milvus-manager` which has the permission to manage the workspace `milvus-ws`. In **Projects**, click **Create**. (A project in KubeSphere is a Kubernetes namespace)

![image-20220318172913153](/Users/bennu/Library/Application Support/typora-user-images/image-20220318172913153.png)

## Step5: Add Milvus to KubeSphere app repository

1. In your workspace, go to **App Repositories** under **App Management**, and then click **Add**.
2. In the dialog that appears, set a name for the repository (for example, `milvus`) and enter the URL `https://milvus-io.github.io/milvus-helm/`. Click **Validate** to verify the URL.  If the URL is available, click **OK** to continue.
3. The app repository displays in the list after it is successfully imported.

## Step6: Deploy Milvus

1. Chose the project **milvus-helm**. Go to **Apps** under **Application Workloads**, and then click **Create**.
2. In the dialog that appears, choose **From App Template**. 
   - **From App Store**: Select apps from the official APP Store of Kubephere.
   - **From App Template**: Select apps from workspace app templates and the third-party Helm app templates of App Repository.
3. In the drop-down list, choose `milvus`, and then click **milvus**.

4. Click **install**, a n d under **Basic Information**, set a name for the app. Check the app version and the deployment location, and then click **Next**.

![image-20220318175622525](/Users/bennu/Library/Application Support/typora-user-images/image-20220318175622525.png)

6. Under **App Settings**, you can edit the configuration file or directly click **Install**.

