#!/bin/bash

pip install gdown

cd ..

cd ../server/data


gdown "https://drive.google.com/uc?id=1nWD8lwlgpA-qOEadkzkxjB1iRCl9nA9u"

gdown "https://drive.google.com/uc?id=14fGV3GYcsJR_78XHrxISoceB5bmffkHL"

tar -xvf test_search_data.tar.gz
tar -xvf test_load_feature.tar.gz
rm test_search_data.tar.gz
rm test_load_feature.tar.gz