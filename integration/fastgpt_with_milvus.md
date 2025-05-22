# Deploying FastGPT with Milvus
[FastGPT](https://fastgpt.in/) is a knowledge-based question and answer system built on the LLM large language model, offering ready-to-use capabilities for data processing and model invocation. Furthermore, it enables workflow orchestration through Flow visualization, thus facilitating complex question and answer scenarios. This tutorial will guide you on how to swiftly deploy your own exclusive FastGPT application using [Milvus](https://milvus.io/).

## Download docker-compose.yml
Ensure that you have already installed [Docker Compose](https://docs.docker.com/compose/).  
Execute the command below to download the docker-compose.yml file.
```shell
mkdir fastgpt
cd fastgpt
curl -O https://raw.githubusercontent.com/labring/FastGPT/main/projects/app/data/config.json

# milvus version
curl -o docker-compose.yml https://raw.githubusercontent.com/labring/FastGPT/main/files/docker/docker-compose-milvus.yml
# zilliz version
# curl -o docker-compose.yml https://raw.githubusercontent.com/labring/FastGPT/main/files/docker/docker-compose-zilliz.yml
```  
> If you're using the Zilliz version, adjust the `MILVUS_ADDRESS` and `MILVUS_TOKEN` link parameters in the docker-compose.yml file, which corresponds to the [Public Endpoint and Api key](https://docs.zilliz.com/docs/on-zilliz-cloud-console#free-cluster-details) in [Zilliz Cloud](https://zilliz.com/cloud).

## Launch the Container
Execute in the same directory as docker-compose.yml. Ensure that the docker-compose version is ideally above 2.17, as some automation commands may not function otherwise.
```shell
# Launch the container
docker-compose up -d
# Wait for 10s, OneAPI typically needs to restart a few times to initially connect to Mysql
sleep 10
# Restart oneapi (Due to certain issues with the default Key of OneAPI, it will display 'channel not found' if not restarted, this can be temporarily resolved by manually restarting once, while waiting for the author's fix)
docker restart oneapi
```

## Access OneAPI to Add Models
OneAPI can be accessed at `ip:3001`. The default username is root, and the password is 123456. You can alter the password after logging in.  
Using OpenAI's model as an example, click on the "Channel" tab, and select your chat model and embedding model under "Models".  
Input your [OpenAI API Key](https://platform.openai.com/docs/quickstart) in the "Secrets" section.  
For the use of models beyond OpenAI, and further information, please consult [One API](https://doc.fastgpt.in/docs/development/one-api/).

## Setting Tokens
Click on the "Tokens" tab. By default, there is a token `Initial Root Token`. You can also create a new token and set a quota on your own.  
Click "Copy" on your token, ensuring that the value of this token matches the `CHAT_API_KEY` value set in the docker-compose.yml file.

## Accessing FastGPT
At present, FastGPT can be directly accessed at `ip:3000` (please mind the firewall). The login username is root, with the password set to `DEFAULT_ROOT_PSW` within the docker-compose.yml environment variable. Should you require domain name access, you would need to install and configure [Nginx](https://nginx.org/en/) on your own.

## Stop the Container
Run the following command to stop the container.
```shell
docker-compose down
```