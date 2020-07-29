#!/bin/bash

echo "Updating Milvus HNSW index..."

CORD19_HNSW_INDEX_NAME=cord19-hnsw-index-milvus
CORD19_HNSW_INDEX_METADATA_URL=https://ai2-semanticscholar-cord-19.s3-us-west-2.amazonaws.com/latest/metadata.csv
CORD19_HNSW_INDEX_SPECTER_URL=https://ai2-semanticscholar-cord-19.s3-us-west-2.amazonaws.com/latest/cord_19_embeddings.tar.gz
CORD19_HNSW_INDEX_FOLDER=api/index/${CORD19_HNSW_INDEX_NAME}


echo "Updating CORD-19 Milvus HNSW index..."
rm -rf ${CORD19_HNSW_INDEX_FOLDER}
mkdir ${CORD19_HNSW_INDEX_FOLDER}
wget ${CORD19_HNSW_INDEX_METADATA_URL} -O ${CORD19_HNSW_INDEX_FOLDER}/metadata.csv
wget ${CORD19_HNSW_INDEX_SPECTER_URL} -O ${CORD19_HNSW_INDEX_FOLDER}/${CORD19_HNSW_INDEX_NAME}.tar.gz
tar xvzf ${CORD19_HNSW_INDEX_FOLDER}/${CORD19_HNSW_INDEX_NAME}.tar.gz
mv cord_19_embeddings*.csv ${CORD19_HNSW_INDEX_FOLDER}/specter.csv
rm ${CORD19_HNSW_INDEX_FOLDER}/${CORD19_HNSW_INDEX_NAME}.tar.gz
python milvus/index_hnsw.py


echo "Successfully updated Milvus HNSW index at api/index/"
