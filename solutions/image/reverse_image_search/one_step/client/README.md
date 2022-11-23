# Client For Reverse Image Search Quick Deploy

## How to Build Docker Image

1. Build the base image.

```bash
docker build -t milvusbootcamp/img-search-client:assets-1.1 . -f docker/Dockerfile.base
```

2. Build the app image with nginx or standalone dockerfile.

```bash
docker build -t milvusbootcamp/img-search-client:1.1 . -f docker/Dockerfile.nginx
# or
docker build -t milvusbootcamp/img-search-client:1.1-standalone . -f docker/Dockerfile.standalone
```