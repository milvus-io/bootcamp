# Reverse Image Search with One Step


## Play with Docker

```bash
docker run --rm -it -v `pwd`/images:/images -p 8080:8080 -p 5000:5000 milvusbootcamp/one-step-img-search:2.1.0
```

## How to build docker images

```bash
# build base
docker build -t milvusbootcamp/one-step-img-search:milvus-2.1.0 . -f docker/Dockerfile.milvus
# build server
docker build -t milvusbootcamp/one-step-img-search:server-2.1.0 . -f docker/Dockerfile.server

cd client
# build front-end assets
docker build -t milvusbootcamp/one-step-img-search:assets-1.1 . -f docker/Dockerfile.base
docker build -t milvusbootcamp/one-step-img-search:client-2.1.0 . -f docker/Dockerfile.debian
cd ..

# build finial image
docker build -t milvusbootcamp/one-step-img-search:2.1.0 . -f docker/Dockerfile
```