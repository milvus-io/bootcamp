BASE_ROOT=/home/labyrinth7x/Codes/PersonSearch/Deep-Cross-Modal-Projection-Learning-for-Image-Text-Matching

IMAGE_ROOT=$BASE_ROOT/data/CUHK-PEDES/imgs
JSON_ROOT=$BASE_ROOT/data/reid_raw.json
OUT_ROOT=$BASE_ROOT/data/processed_data


echo "Process CUHK-PEDES dataset and save it as pickle form"

python ${BASE_ROOT}/datasets/preprocess.py \
        --img_root=${IMAGE_ROOT} \
        --json_root=${JSON_ROOT} \
        --out_root=${OUT_ROOT} \
        --min_word_count 3 
