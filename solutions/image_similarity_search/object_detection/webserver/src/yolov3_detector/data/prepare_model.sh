#!/bin/bash -ex
## model url from mask_rcnn
url='https://github.com/ABNER-1/omnisearch-operators/releases/download/v1.0/yolov3_darknet.tar.gz'
file='yolov3_darknet.tar.gz'
dir='yolov3_darknet'

if [[ ! -d "${dir}" ]]; then
    if [[ ! -f "${file}" ]]; then
        echo "[INFO] Model tar package does not exist, begin to download..."
        wget ${url}
        echo "[INFO] Model tar package download successfully!"
    fi

    echo "[INFO] Model directory does not exist, begin to untar..."
    tar -zxvf ${file}
    rm ${file}
    echo "[INFO] Model directory untar successfully!"
fi

if [[ -d "${dir}" ]];then
    echo "[INFO] Model has been prepared successfully!"
    exit 0
fi

echo "[ERROR] Failed to prepare model due to unexpected reason!"
exit 1
