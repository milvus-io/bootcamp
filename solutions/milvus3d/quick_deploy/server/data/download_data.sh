#!/bin/bash

pip install gdown


mkdir models
cd models
gdown "https://drive.google.com/uc?id=1t5jyJ4Ktmlck6GYhNTPVTFZuRP7wPUYq"

cd ..
gdown "https://drive.google.com/uc?id=1iJNcFliFL7zEmroBHR0iH0a40lVQ8pDR"

tar -xvf ModelNet40.tar.gz
rm ModelNet40.tar.gz
