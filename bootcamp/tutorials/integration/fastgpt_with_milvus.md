# 使用Milvus部署FastGPT

[FastGPT](https://fastgpt.in/) 是一个基于 LLM 大语言模型的知识库问答系统，提供开箱即用的数据处理、模型调用等能力。同时可以通过 Flow 可视化进行工作流编排，从而实现复杂的问答场景。本教程将介绍如何使用 [Milvus](https://milvus.io/) 快速部署一个你自己的专属 FastGPT 应用。


## 下载 docker-compose.yml

请保证你的已经安装[Docker Compose](https://docs.docker.com/compose/).

运行下面命令下载 docker-compose.yml 文件。
```shell
mkdir fastgpt
cd fastgpt
curl -O https://raw.githubusercontent.com/labring/FastGPT/main/projects/app/data/config.json

# milvus 版本
curl -o docker-compose.yml https://raw.githubusercontent.com/labring/FastGPT/main/files/docker/docker-compose-milvus.yml
# zilliz 版本
# curl -o docker-compose.yml https://raw.githubusercontent.com/labring/FastGPT/main/files/docker/docker-compose-zilliz.yml
```

> 如果是Zilliz版本，修改docker-compose.yml文件中的`MILVUS_ADDRESS`和`MILVUS_TOKEN`链接参数，分别对应 [Zilliz Cloud](https://zilliz.com/cloud) 中的 [Public Endpoint 和 Api key](https://cdn.jsdelivr.net/gh/yangchuansheng/fastgpt-imgs@main/imgs/zilliz_key.png).

## 启动容器 
在 docker-compose.yml 同级目录下执行。请确保docker-compose版本最好在2.17以上，否则可能无法执行自动化命令。
```shell
# 启动容器
docker-compose up -d
# 等待10s，OneAPI第一次总是要重启几次才能连上Mysql
sleep 10
# 重启一次oneapi(由于OneAPI的默认Key有点问题，不重启的话会提示找不到渠道，临时手动重启一次解决，等待作者修复)
docker restart oneapi
```

## 打开 OneAPI 添加模型

可以通过`ip:3001`访问OneAPI，默认账号为root密码为123456。登陆后可修改密码。

以OpenAI的模型为例，点击"渠道"选项卡，在"模型"中选择你的chat模型和embedding模型。

在"密钥"中填入你的[OpenAI API Key](https://platform.openai.com/docs/quickstart).

如果你想用除OpenAI之外的其它模型，以及更多信息，请参考[One API](https://doc.fastgpt.in/docs/development/one-api/).

## 设置令牌

点击"令牌"选项卡，默认存在一个令牌`Initial Root Token`，你也可以自行添加一个新的令牌并设置额度。

在你的令牌点击"复制"，确保这个令牌的值就是在docker-compose.yml文件里设置的`CHAT_API_KEY`的值。

## 访问 FastGPT

目前可以通过 `ip:3000` 直接访问(注意防火墙)。登录用户名为 root，密码为docker-compose.yml环境变量里设置的 `DEFAULT_ROOT_PSW`。如果需要域名访问，请自行安装并配置 [Nginx](https://nginx.org/en/).

## 停止容器
执行以下命令停止容器。
```shell
docker-compose down
```


