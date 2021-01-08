GPUS=1
export CUDA_VISIBLE_DEVICES=$GPUS

BASE_ROOT=/data/mia/DCPL/webserver/src
IMAGE_DIR=/data/mia/DCPL/webserver/src/data
ANNO_DIR=$BASE_ROOT/data/processed_data
CKPT_DIR=$BASE_ROOT/data/model_data
LOG_DIR=$BASE_ROOT/data/logs
IMAGE_MODEL=mobilenet_v1
lr=0.0002
batch_size=16
lr_decay_ratio=0.9
epoches_decay=80_150_200


python ${BASE_ROOT}/app.py \
    --bidirectional \
    --model_path $CKPT_DIR/lr-$lr-decay-$lr_decay_ratio-batch-$batch_size/99.pth.tar \
    --image_model $IMAGE_MODEL \
    --log_dir $LOG_DIR/lr-$lr-decay-$lr_decay_ratio-batch-$batch_size \
    --image_dir $IMAGE_DIR \
    --anno_dir $ANNO_DIR \
    --gpus $GPUS \
    --epoch_ema 0
