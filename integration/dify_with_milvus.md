# Deploying Dify with Milvus
[Dify](https://dify.ai/) is an open-source platform designed to simplify building AI applications by combining Backend-as-a-Service with LLMOps. It supports mainstream LLMs, offers an intuitive prompt orchestration interface, high-quality RAG engines, and a flexible AI agent framework. With low-code workflows, easy-to-use interfaces, and APIs, Dify enables both developers and non-technical users to focus on creating innovative, real-world AI solutions without dealing with complexity.

In this tutorial, we will show you how to deploy Dify with Milvus, to enable efficient retrieval and RAG engine.

> This documentation is primarily based on the official [Dify documentation](https://docs.dify.ai/). If you find any outdated or inconsistent content, please prioritize the official documentation and feel free to raise an issue for us.

## Prerequisites

### Clone the Repository
Clone the Dify source code to your local machine:

```shell
git clone https://github.com/langgenius/dify.git
```

### Prepare Environment Configuration
Navigate to the Docker directory in the Dify source code

```shell
cd dify/docker
```
Copy the environment configuration file

```shell
cp .env.example .env
```

## Deployment Options

You can deploy Dify with Milvus using two different approaches. Choose the one that best fits your needs:

## Option 1: Using Milvus with Docker

This option runs Milvus containers alongside Dify on your local machine using Docker Compose.

### Configure Environment Variables

Edit the `.env` file with the following Milvus configuration:

```
VECTOR_STORE=milvus
MILVUS_URI=http://host.docker.internal:19530
MILVUS_TOKEN=
```

> **Note**: 
> - The `MILVUS_URI` uses `host.docker.internal:19530` which allows Docker containers to access Milvus running on the host machine through Docker's internal network.
> - `MILVUS_TOKEN` can be left empty for local Milvus deployments.

### Start the Docker Containers

Start the containers with the `milvus` profile to include Milvus services:

```shell
docker compose --profile milvus up -d
```

This command will start the Dify service along with the `milvus-standalone`, `etcd`, and `minio` containers.

## Option 2: Using Zilliz Cloud

This option connects Dify to a managed Milvus service on Zilliz Cloud.

### Configure Environment Variables

Edit the `.env` file with your Zilliz Cloud connection details:

```
VECTOR_STORE=milvus
MILVUS_URI=YOUR_ZILLIZ_CLOUD_ENDPOINT
MILVUS_TOKEN=YOUR_ZILLIZ_CLOUD_API_KEY
```

> **Note**: 
> - Replace `YOUR_ZILLIZ_CLOUD_ENDPOINT` with your [Public Endpoint](https://docs.zilliz.com/docs/on-zilliz-cloud-console#free-cluster-details) from Zilliz Cloud.
> - Replace `YOUR_ZILLIZ_CLOUD_API_KEY` with your [API key](https://docs.zilliz.com/docs/on-zilliz-cloud-console#free-cluster-details) from Zilliz Cloud.

### Start the Docker Containers

Start only the Dify containers without the Milvus profile:

```shell
docker compose up -d
```

## Accessing Dify

### Log in to Dify
Open your browser and go to the Dify installation page, and you can set your admin account here:
`http://localhost/install` , 
And then log in the main Dify page for further usage.

For further usage and guidance, please refer to the [Dify documentation](https://docs.dify.ai/).