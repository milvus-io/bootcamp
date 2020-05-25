#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
import os

import click

from deep_speaker.audio import Audio
from deep_speaker.batcher import KerasFormatConverter
from deep_speaker.constants import SAMPLE_RATE, NUM_FRAMES
from deep_speaker.test import test
from deep_speaker.train import start_training
from deep_speaker.utils import ClickType as Ct, ensures_dir
from deep_speaker.utils import init_pandas

logger = logging.getLogger(__name__)

VERSION = '3.0a'


@click.group()
def cli():
    logging.basicConfig(format='%(asctime)12s - %(levelname)s - %(message)s', level=logging.INFO)
    init_pandas()


@cli.command('version', short_help='Prints the version.')
def version():
    print(f'Version is {VERSION}.')


@cli.command('build-mfcc-cache', short_help='Build audio cache.')
@click.option('--working_dir', required=True, type=Ct.output_dir())
@click.option('--audio_dir', default=None)
@click.option('--sample_rate', default=SAMPLE_RATE, show_default=True, type=int)
def build_audio_cache(working_dir, audio_dir, sample_rate):
    ensures_dir(working_dir)
    if audio_dir is None:
        audio_dir = os.path.join(working_dir, 'LibriSpeech')
    Audio(cache_dir=working_dir, audio_dir=audio_dir, sample_rate=sample_rate)


@cli.command('build-keras-inputs', short_help='Build inputs to Keras.')
@click.option('--working_dir', required=True, type=Ct.input_dir())
@click.option('--counts_per_speaker', default='600,100', show_default=True, type=str)  # train,test
def build_keras_inputs(working_dir, counts_per_speaker):
    counts_per_speaker = [int(b) for b in counts_per_speaker.split(',')]
    kc = KerasFormatConverter(working_dir)
    kc.generate(max_length=NUM_FRAMES, counts_per_speaker=counts_per_speaker)
    kc.persist_to_disk()


@cli.command('test-model', short_help='Test a Keras model.')
@click.option('--working_dir', required=True, type=Ct.input_dir())
@click.option('--checkpoint_file', required=True, type=Ct.input_file())
def test_model(working_dir, checkpoint_file=None):
    # export CUDA_VISIBLE_DEVICES=0; python cli.py test-model
    # --working_dir /home/philippe/ds-test/triplet-training/
    # --checkpoint_file ../ds-test/checkpoints-softmax/ResCNN_checkpoint_102.h5
    # f-measure = 0.789, true positive rate = 0.733, accuracy = 0.996, equal error rate = 0.043

    # export CUDA_VISIBLE_DEVICES=0; python cli.py test-model
    # --working_dir /home/philippe/ds-test/triplet-training/
    # --checkpoint_file ../ds-test/checkpoints-triplets/ResCNN_checkpoint_175.h5
    # f-measure = 0.849, true positive rate = 0.798, accuracy = 0.997, equal error rate = 0.025
    test(working_dir, checkpoint_file)


@cli.command('train-model', short_help='Train a Keras model.')
@click.option('--working_dir', required=True, type=Ct.input_dir())
@click.option('--pre_training_phase/--no_pre_training_phase', default=False, show_default=True)
def train_model(working_dir, pre_training_phase):
    # PRE TRAINING

    # commit a5030dd7a1b53cd11d5ab7832fa2d43f2093a464
    # Merge: a11d13e b30e64e
    # Author: Philippe Remy <premy.enseirb@gmail.com>
    # Date:   Fri Apr 10 10:37:59 2020 +0900
    # LibriSpeech train-clean-data360 (600, 100). 0.985 on test set (enough for pre-training).

    # TRIPLET TRAINING
    # [...]
    # Epoch 175/1000
    # 2000/2000 [==============================] - 919s 459ms/step - loss: 0.0077 - val_loss: 0.0058
    # Epoch 176/1000
    # 2000/2000 [==============================] - 917s 458ms/step - loss: 0.0075 - val_loss: 0.0059
    # Epoch 177/1000
    # 2000/2000 [==============================] - 927s 464ms/step - loss: 0.0075 - val_loss: 0.0059
    # Epoch 178/1000
    # 2000/2000 [==============================] - 948s 474ms/step - loss: 0.0073 - val_loss: 0.0058
    start_training(working_dir, pre_training_phase)


if __name__ == '__main__':
    cli()
