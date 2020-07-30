import logging
import os

import numpy as np
import tensorflow.keras.backend as K
from tensorflow.keras import layers
from tensorflow.keras import regularizers
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Lambda, Dense
from tensorflow.keras.layers import Reshape
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

from deep_speaker.constants import NUM_FBANKS, NUM_FRAMES
from deep_speaker.triplet_loss import deep_speaker_loss

logger = logging.getLogger(__name__)


class DeepSpeakerModel:

    # I thought it was 3 but maybe energy is added at a 4th dimension.
    # would be better to have 4 dimensions:
    # MFCC, DIFF(MFCC), DIFF(DIFF(MFCC)), ENERGIES (probably tiled across the frequency domain).
    # this seems to help match the parameter counts.
    def __init__(self, batch_input_shape=(None, NUM_FRAMES, NUM_FBANKS, 1), include_softmax=False,
                 num_speakers_softmax=None):
        self.include_softmax = include_softmax
        if self.include_softmax:
            assert num_speakers_softmax > 0
        self.clipped_relu_count = 0

        # http://cs231n.github.io/convolutional-networks/
        # conv weights
        # #params = ks * ks * nb_filters * num_channels_input

        # Conv128-s
        # 5*5*128*128/2+128
        # ks*ks*nb_filters*channels/strides+bias(=nb_filters)

        # take 100 ms -> 4 frames.
        # if signal is 3 seconds, then take 100ms per 100ms and average out this network.
        # 8*8 = 64 features.

        # used to share all the layers across the inputs

        # num_frames = K.shape() - do it dynamically after.
        inputs = Input(batch_shape=batch_input_shape, name='input')
        x = self.cnn_component(inputs)

        x = Reshape((-1, 2048))(x)
        # Temporal average layer. axis=1 is time.
        x = Lambda(lambda y: K.mean(y, axis=1), name='average')(x)
        if include_softmax:
            logger.info('Including a Dropout layer to reduce overfitting.')
            # used for softmax because the dataset we pre-train on might be too small. easy to overfit.
            x = Dropout(0.5)(x)
        x = Dense(512, name='affine')(x)
        if include_softmax:
            # Those weights are just when we train on softmax.
            x = Dense(num_speakers_softmax, activation='softmax')(x)
        else:
            # Does not contain any weights.
            x = Lambda(lambda y: K.l2_normalize(y, axis=1), name='ln')(x)
        self.m = Model(inputs, x, name='ResCNN')

    def keras_model(self):
        return self.m

    def get_weights(self):
        w = self.m.get_weights()
        if self.include_softmax:
            w.pop()  # last 2 are the W_softmax and b_softmax.
            w.pop()
        return w

    def clipped_relu(self, inputs):
        relu = Lambda(lambda y: K.minimum(K.maximum(y, 0), 20), name=f'clipped_relu_{self.clipped_relu_count}')(inputs)
        self.clipped_relu_count += 1
        return relu

    def identity_block(self, input_tensor, kernel_size, filters, stage, block):
        conv_name_base = f'res{stage}_{block}_branch'

        x = Conv2D(filters,
                   kernel_size=kernel_size,
                   strides=1,
                   activation=None,
                   padding='same',
                   kernel_initializer='glorot_uniform',
                   kernel_regularizer=regularizers.l2(l=0.0001),
                   name=conv_name_base + '_2a')(input_tensor)
        x = BatchNormalization(name=conv_name_base + '_2a_bn')(x)
        x = self.clipped_relu(x)

        x = Conv2D(filters,
                   kernel_size=kernel_size,
                   strides=1,
                   activation=None,
                   padding='same',
                   kernel_initializer='glorot_uniform',
                   kernel_regularizer=regularizers.l2(l=0.0001),
                   name=conv_name_base + '_2b')(x)
        x = BatchNormalization(name=conv_name_base + '_2b_bn')(x)

        x = self.clipped_relu(x)

        x = layers.add([x, input_tensor])
        x = self.clipped_relu(x)
        return x

    def conv_and_res_block(self, inp, filters, stage):
        conv_name = 'conv{}-s'.format(filters)
        # TODO: why kernel_regularizer?
        o = Conv2D(filters,
                   kernel_size=5,
                   strides=2,
                   activation=None,
                   padding='same',
                   kernel_initializer='glorot_uniform',
                   kernel_regularizer=regularizers.l2(l=0.0001), name=conv_name)(inp)
        o = BatchNormalization(name=conv_name + '_bn')(o)
        o = self.clipped_relu(o)
        for i in range(3):
            o = self.identity_block(o, kernel_size=3, filters=filters, stage=stage, block=i)
        return o

    def cnn_component(self, inp):
        x = self.conv_and_res_block(inp, 64, stage=1)
        x = self.conv_and_res_block(x, 128, stage=2)
        x = self.conv_and_res_block(x, 256, stage=3)
        x = self.conv_and_res_block(x, 512, stage=4)
        return x

    def set_weights(self, w):
        for layer, layer_w in zip(self.m.layers, w):
            layer.set_weights(layer_w)
            logger.info(f'Setting weights for [{layer.name}]...')


def main():
    # Looks correct to me.
    # I have 37K but paper reports 41K. which is not too far.
    dsm = DeepSpeakerModel()
    dsm.m.summary()

    # I suspect num frames to be 32.
    # Then fbank=64, then total would be 32*64 = 2048.
    # plot_model(dsm.m, to_file='model.png', dpi=300, show_shapes=True, expand_nested=True)


def _train():
    # x = np.random.uniform(size=(6, 32, 64, 4))  # 6 is multiple of 3.
    # y_softmax = np.random.uniform(size=(6, 100))
    # dsm = DeepSpeakerModel(batch_input_shape=(None, 32, 64, 4), include_softmax=True, num_speakers_softmax=100)
    # dsm.m.compile(optimizer=Adam(lr=0.01), loss='categorical_crossentropy')
    # print(dsm.m.predict(x).shape)
    # print(dsm.m.evaluate(x, y_softmax))
    # w = dsm.get_weights()
    dsm = DeepSpeakerModel(batch_input_shape=(None, 32, 64, 4), include_softmax=False)
    # dsm.m.set_weights(w)
    dsm.m.compile(optimizer=Adam(lr=0.01), loss=deep_speaker_loss)

    # it works!!!!!!!!!!!!!!!!!!!!
    # unit_batch_size = 20
    # anchor = np.ones(shape=(unit_batch_size, 32, 64, 4))
    # positive = np.array(anchor)
    # negative = np.ones(shape=(unit_batch_size, 32, 64, 4)) * (-1)
    # batch = np.vstack((anchor, positive, negative))
    # x = batch
    # y = np.zeros(shape=(len(batch), 512))  # not important.
    # print('Starting to fit...')
    # while True:
    #     print(dsm.m.train_on_batch(x, y))

    # should not work... and it does not work!
    unit_batch_size = 20
    negative = np.ones(shape=(unit_batch_size, 32, 64, 4)) * (-1)
    batch = np.vstack((negative, negative, negative))
    x = batch
    y = np.zeros(shape=(len(batch), 512))  # not important.
    print('Starting to fit...')
    while True:
        print(dsm.m.train_on_batch(x, y))


def _test_checkpoint_compatibility():
    dsm = DeepSpeakerModel(batch_input_shape=(None, 32, 64, 4), include_softmax=True, num_speakers_softmax=10)
    dsm.m.save_weights('test.h5')
    dsm = DeepSpeakerModel(batch_input_shape=(None, 32, 64, 4), include_softmax=False)
    dsm.m.load_weights('test.h5', by_name=True)
    os.remove('test.h5')


if __name__ == '__main__':
    _test_checkpoint_compatibility()
