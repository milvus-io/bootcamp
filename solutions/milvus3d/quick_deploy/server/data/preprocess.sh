#!/bin/bash

cd ../data

IF_BATCH=$1
FILE_NAME=$2
FILE_PATH=$3

if ${IF_BATCH}
then
python3 compress.py --batch 'T'
docker run -it --rm -v `pwd`:/data pymesh/pymesh /bin/bash -c "cd /data && python preprocess_npy.py --batch 'T'"

else
python3 compress.py --batch "F" --filename ${FILE_NAME} --path ${FILE_PATH}
docker run -it --rm -v `pwd`:/data pymesh/pymesh /bin/bash -c "cd /data && python preprocess_npy.py --batch 'F' --filename ${FILE_NAME}"

fi