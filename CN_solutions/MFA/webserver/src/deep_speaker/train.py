import logging
import os

from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import SGD
from tqdm import tqdm

from deep_speaker.batcher import KerasFormatConverter, LazyTripletBatcher
from deep_speaker.constants import BATCH_SIZE, CHECKPOINTS_SOFTMAX_DIR, CHECKPOINTS_TRIPLET_DIR, NUM_FRAMES, NUM_FBANKS
from deep_speaker.conv_models import DeepSpeakerModel
from deep_speaker.triplet_loss import deep_speaker_loss
from deep_speaker.utils import load_best_checkpoint, ensures_dir

logger = logging.getLogger(__name__)

# Otherwise it's just too much logging from Tensorflow...
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def fit_model(dsm: DeepSpeakerModel, working_dir: str, max_length: int = NUM_FRAMES, batch_size=BATCH_SIZE):
    batcher = LazyTripletBatcher(working_dir, max_length, dsm)

    # build small test set.
    test_batches = []
    for _ in tqdm(range(200), desc='Build test set'):
        test_batches.append(batcher.get_batch_test(batch_size))

    def test_generator():
        while True:
            for bb in test_batches:
                yield bb

    def train_generator():
        while True:
            yield batcher.get_random_batch(batch_size, is_test=False)

    checkpoint_name = dsm.m.name + '_checkpoint'
    checkpoint_filename = os.path.join(CHECKPOINTS_TRIPLET_DIR, checkpoint_name + '_{epoch}.h5')
    checkpoint = ModelCheckpoint(monitor='val_loss', filepath=checkpoint_filename, save_best_only=True)
    dsm.m.fit(x=train_generator(), y=None, steps_per_epoch=2000, shuffle=False,
              epochs=1000, validation_data=test_generator(), validation_steps=len(test_batches),
              callbacks=[checkpoint])


def fit_model_softmax(dsm: DeepSpeakerModel, kx_train, ky_train, kx_test, ky_test,
                      batch_size=BATCH_SIZE, max_epochs=1000, initial_epoch=0):
    checkpoint_name = dsm.m.name + '_checkpoint'
    checkpoint_filename = os.path.join(CHECKPOINTS_SOFTMAX_DIR, checkpoint_name + '_{epoch}.h5')
    checkpoint = ModelCheckpoint(monitor='val_accuracy', filepath=checkpoint_filename, save_best_only=True)

    # if the accuracy does not increase by 0.1% over 20 epochs, we stop the training.
    early_stopping = EarlyStopping(monitor='val_accuracy', min_delta=0.001, patience=20, verbose=1, mode='max')

    # if the accuracy does not increase over 10 epochs, we reduce the learning rate by half.
    reduce_lr = ReduceLROnPlateau(monitor='val_accuracy', factor=0.5, patience=10, min_lr=0.0001, verbose=1)

    max_len_train = len(kx_train) - len(kx_train) % batch_size
    kx_train = kx_train[0:max_len_train]
    ky_train = ky_train[0:max_len_train]
    max_len_test = len(kx_test) - len(kx_test) % batch_size
    kx_test = kx_test[0:max_len_test]
    ky_test = ky_test[0:max_len_test]

    dsm.m.fit(x=kx_train,
              y=ky_train,
              batch_size=batch_size,
              epochs=initial_epoch + max_epochs,
              initial_epoch=initial_epoch,
              verbose=1,
              shuffle=True,
              validation_data=(kx_test, ky_test),
              callbacks=[early_stopping, reduce_lr, checkpoint])


def start_training(working_dir, pre_training_phase=True):
    ensures_dir(CHECKPOINTS_SOFTMAX_DIR)
    ensures_dir(CHECKPOINTS_TRIPLET_DIR)
    batch_input_shape = [None, NUM_FRAMES, NUM_FBANKS, 1]
    if pre_training_phase:
        logger.info('Softmax pre-training.')
        kc = KerasFormatConverter(working_dir)
        num_speakers_softmax = len(kc.categorical_speakers.speaker_ids)
        dsm = DeepSpeakerModel(batch_input_shape, include_softmax=True, num_speakers_softmax=num_speakers_softmax)
        dsm.m.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        pre_training_checkpoint = load_best_checkpoint(CHECKPOINTS_SOFTMAX_DIR)
        if pre_training_checkpoint is not None:
            initial_epoch = int(pre_training_checkpoint.split('/')[-1].split('.')[0].split('_')[-1])
            logger.info(f'Initial epoch is {initial_epoch}.')
            logger.info(f'Loading softmax checkpoint: {pre_training_checkpoint}.')
            dsm.m.load_weights(pre_training_checkpoint)  # latest one.
        else:
            initial_epoch = 0
        fit_model_softmax(dsm, kc.kx_train, kc.ky_train, kc.kx_test, kc.ky_test, initial_epoch=initial_epoch)
    else:
        logger.info('Training with the triplet loss.')
        dsm = DeepSpeakerModel(batch_input_shape, include_softmax=False)
        triplet_checkpoint = load_best_checkpoint(CHECKPOINTS_TRIPLET_DIR)
        pre_training_checkpoint = load_best_checkpoint(CHECKPOINTS_SOFTMAX_DIR)
        if triplet_checkpoint is not None:
            logger.info(f'Loading triplet checkpoint: {triplet_checkpoint}.')
            dsm.m.load_weights(triplet_checkpoint)
        elif pre_training_checkpoint is not None:
            logger.info(f'Loading pre-training checkpoint: {pre_training_checkpoint}.')
            # If `by_name` is True, weights are loaded into layers only if they share the
            # same name. This is useful for fine-tuning or transfer-learning models where
            # some of the layers have changed.
            dsm.m.load_weights(pre_training_checkpoint, by_name=True)
        dsm.m.compile(optimizer=SGD(), loss=deep_speaker_loss)
        fit_model(dsm, working_dir, NUM_FRAMES)
