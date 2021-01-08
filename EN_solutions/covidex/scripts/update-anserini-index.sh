#!/bin/bash


CORD19_INDEX_NAME=lucene-index-cord19-paragraph
CORD19_DATE=$1
COVIDEX_CORD19_INDEX_PATH=api/index/${CORD19_INDEX_NAME}

TRIALSTREAMER_INDEX_NAME=lucene-index-trialstreamer
TRIALSTREAMER_INDEX_DATE=2020-04-15
TRIALSTREAMER_INDEX_URL=https://www.dropbox.com/s/d2s92i6y927s1c7/lucene-index-trialstreamer-2020-04-15.tar.gz


# Check if date provided
if [ $# -eq 0 ]; then
    echo "USAGE: sh scripts/update-anserini.sh [DATE]"
    exit 1
fi


echo "Updating CORD-19 index for $1..."

python3 -m pip install tqdm # required for indexing script
[ -d anserini ] && echo "Found Anserini folder..." || git clone https://github.com/castorini/anserini.git

cd anserini
git pull && git submodule update --init --recursive
mvn clean package appassembler:assemble -Dmaven.javadoc.skip=true && \
    python3 src/main/python/trec-covid/index_cord19.py --date $CORD19_DATE --all
cd ..

rm -rf $COVIDEX_CORD19_INDEX_PATH
mkdir $COVIDEX_CORD19_INDEX_PATH
mv anserini/indexes/$CORD19_INDEX_NAME-$CORD19_DATE/* $COVIDEX_CORD19_INDEX_PATH/


echo "Updating Trialstreamer index..."
wget ${TRIALSTREAMER_INDEX_URL}
rm -rf api/index/${TRIALSTREAMER_INDEX_NAME}
mkdir api/index/${TRIALSTREAMER_INDEX_NAME}
tar xvfz ${TRIALSTREAMER_INDEX_NAME}-${TRIALSTREAMER_INDEX_DATE}.tar.gz \
    -C api/index/${TRIALSTREAMER_INDEX_NAME} --strip-components 1
rm ${TRIALSTREAMER_INDEX_NAME}-${TRIALSTREAMER_INDEX_DATE}.tar.gz


echo "Successfully updated search indices at api/index/"
