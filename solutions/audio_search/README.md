# 基于 Milvus 的音频检索系统

本项目后续将不再维护更新，最新内容将更新在https://github.com/zilliz-bootcamp/audio_search。

该项目使用 [PANNs](https://github.com/qiuqiangkong/audioset_tagging_cnn) 对音频做 Embeddings 提取，然后利用 Milvus 检索出相似的音频数据。 

## 系统部署

### 环境要求

- [Milvus 0.11.0](https://milvus.io/docs/v0.11.0/milvus_docker-cpu.md) (请注意 Milvus 的版本)
- [MySQL](https://hub.docker.com/r/mysql/mysql-server)
- [Python3](https://www.python.org/downloads/)

### 启动系统服务

1. **安装 python依赖**

   ```bash
   $ cd bootcamp/solutions/audio_search/webserver/
   $ pip install -r audio_requirements.txt
   ```

2. **修改配置参数**

   在运行代码前，请修改该文件中的相关配置：**webserver/audio/common/config.py**:

   | Parameter    | Description               | Default setting |
   | ------------ | ------------------------- | --------------- |
   | MILVUS_HOST  | milvus service ip address | 127.0.0.1       |
   | MILVUS_PORT  | milvus service port       | 19530           |
   | MYSQL_HOST   | postgresql service ip     | 127.0.0.1       |
   | MYSQL_PORT   | postgresql service port   | 3306            |
   | MYSQL_USER   | postgresql user name      | root            |
   | MYSQL_PWD    | postgresql password       | 123456          |
   | MYSQL_DB     | postgresql datebase name  | mysql           |
   | MILVUS_TABLE | default table name        | milvus_audio    |

3. **启动服务**

   ```bash
   $ cd webserver
   $ python main.py
   ```



## 系统使用

在浏览器输入 `127.0.0.1:8002/docs` 就可以看到系统中所有的 API 。

![](./pic/all_API.png)

- 插入数据

  你可以下载示例数据 [game_sound.zip](https://github.com/shiyu22/bootcamp/blob/0.11.0/solutions/audio_search/data/game_sound.zip?raw=true) 并导入系统。

  > 注意：要求压缩包文件中的声音都是 wav 格式。

  ![](./pic/insert.png)

- 音频检索

  导入 [test.wav](https://github.com/shiyu22/bootcamp/blob/0.11.0/solutions/audio_search/data/test.wav) 在系统中检索与该音频最相似的结果。
  
  ![](./pic/search.png)

> 如果你需要前端界面，请参考https://zilliz.com/demos/。