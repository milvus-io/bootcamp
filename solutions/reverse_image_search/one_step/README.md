# Reverse Image Search with One Step


## How to build docker images

```bash
# build base
docker build -t milvusbootcamp/one-step-img-search:milvus-2.1.0 . -f docker/Dockerfile.milvus
# build server
docker build -t milvusbootcamp/one-step-img-search:server-2.1.0 . -f docker/Dockerfile.server
```