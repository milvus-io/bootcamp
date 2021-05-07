import sys

import numpy as np

import triplet_loss
from batcher import KerasFormatConverter, TripletBatcherSelectHardNegatives, TripletBatcher
from constants import NUM_FBANKS, NUM_FRAMES, CHECKPOINTS_TRIPLET_DIR, CHECKPOINTS_SOFTMAX_DIR, BATCH_SIZE
from conv_models import DeepSpeakerModel
from triplet_loss import deep_speaker_loss
from utils import load_best_checkpoint


def predict(x):
    y = np.tile(np.expand_dims(np.mean(x, axis=(1, 2, 3)), axis=1), (1, 512))
    return y
    # norm_y = np.linalg.norm(y, axis=1, ord=2, keepdims=True)
    # return y / (norm_y + 1e-12)


def main2():
    num_utterances_per_speaker = 50
    num_speakers = 100
    num_samples = num_speakers * num_utterances_per_speaker
    kx_train = np.zeros(shape=(num_samples, 32, 64, 1))
    ky_train = np.zeros(shape=(num_samples, num_speakers))
    for i in range(num_samples):
        speaker_id = i % num_speakers
        ky_train[i, speaker_id] = 1
        kx_train[i] = speaker_id
    kx_test = np.array(kx_train)
    ky_test = np.array(ky_train)

    tpshn = TripletBatcherSelectHardNegatives(kx_train, ky_train, kx_test, ky_test, None)
    tp = TripletBatcher(kx_train, ky_train, kx_test, ky_test)
    avg = []
    avg2 = []
    while True:
        bx, by = tp.get_batch(BATCH_SIZE, is_test=False)
        avg.append(float(triplet_loss.deep_speaker_loss(predict(bx), predict(bx))))

        bx, by = tpshn.get_batch(BATCH_SIZE, is_test=False, predict=predict)
        avg2.append(float(triplet_loss.deep_speaker_loss(predict(bx), predict(bx))))

        print(np.mean(avg), np.mean(avg2))


def main():
    select = True
    try:
        sys.argv[1]
    except:
        select = False
    print('select', select)

    working_dir = '/media/philippe/8TB/deep-speaker'
    # by construction this  losses should be much higher than the normal losses.
    # we select batches this way.
    batch_input_shape = [None, NUM_FRAMES, NUM_FBANKS, 1]
    print('Testing with the triplet losses.')
    dsm = DeepSpeakerModel(batch_input_shape, include_softmax=False)
    triplet_checkpoint = load_best_checkpoint(CHECKPOINTS_TRIPLET_DIR)
    pre_training_checkpoint = load_best_checkpoint(CHECKPOINTS_SOFTMAX_DIR)
    if triplet_checkpoint is not None:
        print(f'Loading triplet checkpoint: {triplet_checkpoint}.')
        dsm.m.load_weights(triplet_checkpoint)
    elif pre_training_checkpoint is not None:
        print(f'Loading pre-training checkpoint: {pre_training_checkpoint}.')
        # If `by_name` is True, weights are loaded into layers only if they share the
        # same name. This is useful for fine-tuning or transfer-learning models where
        # some of the layers have changed.
        dsm.m.load_weights(pre_training_checkpoint, by_name=True)
    dsm.m.compile(optimizer='adam', loss=deep_speaker_loss)
    kc = KerasFormatConverter(working_dir)
    if select:
        print('TripletBatcherSelectHardNegatives()')
        batcher = TripletBatcherSelectHardNegatives(kc.kx_train, kc.ky_train, kc.kx_test, kc.ky_test, dsm)
    else:
        print('TripletBatcher()')
        batcher = TripletBatcher(kc.kx_train, kc.ky_train, kc.kx_test, kc.ky_test)
    batch_size = BATCH_SIZE
    losses = []
    while True:
        _bx, _by = batcher.get_batch(batch_size, is_test=False)
        losses.append(dsm.m.evaluate(_bx, _by, verbose=0, batch_size=BATCH_SIZE))
        print(np.mean(losses))


if __name__ == '__main__':
    main2()
