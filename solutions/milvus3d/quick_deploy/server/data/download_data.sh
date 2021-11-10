#!/bin/bash

pip install gdown


cd ..
gdown "https://drive.google.com/uc?id=1iJNcFliFL7zEmroBHR0iH0a40lVQ8pDR"

tar -xvf ModelNet40.tar.gz
rm ModelNet40.tar.gz
