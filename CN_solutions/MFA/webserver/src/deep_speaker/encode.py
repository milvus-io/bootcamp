import numpy as np
import random
from deep_speaker.audio import read_mfcc
from deep_speaker.batcher import sample_from_mfcc
from deep_speaker.constants import SAMPLE_RATE, NUM_FRAMES
from deep_speaker.conv_models import DeepSpeakerModel
from deep_speaker.test import batch_cosine_similarity


def voc_to_vec(voc):
    np.random.seed(123)
    random.seed(123)

    model = DeepSpeakerModel()
    model.m.load_weights('/app/src/deep_speaker/checkpoints/ResCNN_triplet_training_checkpoint_265.h5', by_name=True)

    mfcc = sample_from_mfcc(read_mfcc(voc, SAMPLE_RATE), NUM_FRAMES)
    predict = model.m.predict(np.expand_dims(mfcc, axis=0))
    vec = list(map(float,predict.tolist()[0]))

    return vec