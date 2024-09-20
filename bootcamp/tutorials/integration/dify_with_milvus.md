# Deploying Dify with Milvus
[Dify](https://dify.ai/) is an open-source platform designed to simplify building AI applications by combining Backend-as-a-Service with LLMOps. It supports mainstream LLMs, offers an intuitive prompt orchestration interface, high-quality RAG engines, and a flexible AI agent framework. With low-code workflows, easy-to-use interfaces, and APIs, Dify enables both developers and non-technical users to focus on creating innovative, real-world AI solutions without dealing with complexity.

In this tutorial, we will show you how to deploy Dify with Milvus, to enable efficient retrieval and RAG engine.


### Clone the Repository
Clone the Dify source code to your local machine:


```shell
git clone https://github.com/langgenius/dify.git
```


### Set the Environment Variables
Navigate to the Docker directory in the Dify source code


```shell
cd dify/docker
```
Copy the environment configuration file


```shell
cp .env.example .env
```

Change the value `VECTOR_STORE` in the `.env` file 
```
VECTOR_STORE=milvus
```
Change the Milvus configuration in the `.env` file
```
MILVUS_URI=xxx
MILVUS_TOKEN=xxx
```

In this setup, please use the external URI of the server, e.g.`http://172.16.16.16:19530`, as your `MILVUS_URI`.

For the `MILVUS_TOKEN`, if you have not set a token for your Milvus server, you can set it to an empty string like `MILVUS_TOKEN=`, otherwise, you need to set it to your Milvus token. For more information about how to set token in Milvus, you can refer the [authenticate page](https://milvus.io/docs/authenticate.md?tab=docker#Update-user-password).

### Start the Docker Containers

Choose the appropriate command to start the containers based on the Docker Compose version on your system. You can use the `$ docker compose version` command to check the version, and refer to the Docker documentation for more information:

If you have Docker Compose V2, use the following command:


```shell
docker compose up -d
```
If you have Docker Compose V1, use the following command:


```shell
docker-compose up -d
```

### Log in to Dify
Open your browser and go to the Dify installation page, and you can set your admin account here:
`http://localhost/install` , 
And then log in the main Dify page for further usage.

For further usage and guidance, please refer to the [Dify documentation](https://docs.dify.ai/).