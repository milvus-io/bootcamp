import argparse
from config import log_config
import logging


def parse_args():
    parser = argparse.ArgumentParser(description='command for evaluate on CUHK-PEDES')
    # Directory
    parser.add_argument('--image_dir', type=str, help='directory to store dataset',
                        default='/data/mia/DCPL/webserver/src/data')
    parser.add_argument('--anno_dir', type=str, help='directory to store anno',
                        default='/data/mia/DCPL/webserver/src/data/processed_data')
    parser.add_argument('--model_path', type=str, help='directory to load checkpoint',
                        default='/data/mia/DCPL/webserver/src/data/lr-0.0002-decay-0.9-batch-16/99.pth.tar')
    parser.add_argument('--log_dir', type=str, help='directory to store log',
                        default="/data/mia/DCPL/webserver/src/data/logs/lr-0.0002-decay-0.9-batch-16/")

    # LSTM setting
    parser.add_argument('--embedding_size', type=int, default=512)
    parser.add_argument('--num_lstm_units', type=int, default=512)
    parser.add_argument('--vocab_size', type=int, default=12000)
    parser.add_argument('--lstm_dropout_ratio', type=float, default=0.7)
    parser.add_argument('--bidirectional', action='store_true')
    parser.add_argument('--max_length', type=int, default=100)
    parser.add_argument('--feature_size', type=int, default=512)

    parser.add_argument('--image_model', type=str, default='mobilenet_v1')
    parser.add_argument('--cnn_dropout_keep', type=float, default=0.999)

    parser.add_argument('--epoch_ema', type=int, default=0)

    # Default setting
    parser.add_argument('--gpus', type=str, default='1')
    args = parser.parse_args()
    return args


def config():
    args = parse_args()
    log_config(args, 'test')
    return args
