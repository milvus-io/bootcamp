#!/bin/bash

echo "Updating Anserini index..."

CORD19_INDEX_NAME=lucene-index-cord19-paragraph
CORD19_INDEX_DATE=2020-05-26
CORD19_INDEX_URL=https://www.dropbox.com/s/ng4hwlr9414o4ju/lucene-index-cord19-paragraph-2020-05-26.tar.gz

TRIALSTREAMER_INDEX_NAME=lucene-index-trialstreamer
TRIALSTREAMER_INDEX_DATE=2020-04-15
TRIALSTREAMER_INDEX_URL=https://www.dropbox.com/s/d2s92i6y927s1c7/lucene-index-trialstreamer-2020-04-15.tar.gz

CORD19_HNSW_INDEX_NAME=cord19-hnsw-index-milvus
CORD19_HNSW_INDEX_METADATA_URL=https://ai2-semanticscholar-cord-19.s3-us-west-2.amazonaws.com/latest/metadata.csv
CORD19_HNSW_INDEX_SPECTER_URL=https://ai2-semanticscholar-cord-19.s3-us-west-2.amazonaws.com/latest/cord_19_embeddings.tar.gz
CORD19_HNSW_INDEX_FOLDER=api/index/${CORD19_HNSW_INDEX_NAME}


echo "Updating CORD-19 index..."
wget ${CORD19_INDEX_URL}
rm -rf api/index/${CORD19_INDEX_NAME}
mkdir api/index/${CORD19_INDEX_NAME}
tar xvfz ${CORD19_INDEX_NAME}-${CORD19_INDEX_DATE}.tar.gz \
    -C api/index/${CORD19_INDEX_NAME} --strip-components 1
rm ${CORD19_INDEX_NAME}-${CORD19_INDEX_DATE}.tar.gz


echo "Updating Trialstreamer index..."
wget ${TRIALSTREAMER_INDEX_URL}
rm -rf api/index/${TRIALSTREAMER_INDEX_NAME}
mkdir api/index/${TRIALSTREAMER_INDEX_NAME}
tar xvfz ${TRIALSTREAMER_INDEX_NAME}-${TRIALSTREAMER_INDEX_DATE}.tar.gz \
    -C api/index/${TRIALSTREAMER_INDEX_NAME} --strip-components 1
rm ${TRIALSTREAMER_INDEX_NAME}-${TRIALSTREAMER_INDEX_DATE}.tar.gz


echo "Updating CORD-19 index..."
rm -rf ${CORD19_HNSW_INDEX_FOLDER}
mkdir ${CORD19_HNSW_INDEX_FOLDER}
wget ${CORD19_HNSW_INDEX_METADATA_URL} -O ${CORD19_HNSW_INDEX_FOLDER}/metadata.csv
wget ${CORD19_HNSW_INDEX_SPECTER_URL} -O ${CORD19_HNSW_INDEX_FOLDER}/${CORD19_HNSW_INDEX_NAME}.tar.gz
tar xvzf ${CORD19_HNSW_INDEX_FOLDER}/${CORD19_HNSW_INDEX_NAME}.tar.gz
mv cord_19_embeddings*.csv ${CORD19_HNSW_INDEX_FOLDER}/specter.csv
rm ${CORD19_HNSW_INDEX_FOLDER}/${CORD19_HNSW_INDEX_NAME}.tar.gz


echo "Successfully updated all indices at api/index/"