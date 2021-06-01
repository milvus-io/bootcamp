all:fronted api
	echo "build finish"
fronted:
	cd milvus-demo-img2video && docker build -t milvus.io/search-video-demo:v1 .
api:
	docker build -t milvus.io/search-video-api:v1 .
