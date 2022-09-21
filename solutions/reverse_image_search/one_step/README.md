# Reverse Image Search with One Step


## Play with Docker

```bash
docker run --rm -it -v `pwd`/images:/images -p 8000:80 milvusbootcamp/one-step-img-search:2.1.0
```

## How to build docker images

```bash
# step1: build base
docker build -t milvusbootcamp/one-step-img-search:milvus-2.1.0 . -f docker/Dockerfile.milvus

# step2: build server and client
docker build -t milvusbootcamp/one-step-img-search:server-2.1.0 . -f docker/Dockerfile.server

# step2: build server and client
cd client
docker build -t milvusbootcamp/one-step-img-search:client-2.1.0 . -f docker/Dockerfile
cd ..

# step3: build finial image
docker build -t milvusbootcamp/one-step-img-search:2.1.0 . -f docker/Dockerfile
```