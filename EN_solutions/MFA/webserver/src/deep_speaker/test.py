import logging

import numpy as np
from tqdm import tqdm

from deep_speaker.audio import Audio
from deep_speaker.batcher import LazyTripletBatcher
from deep_speaker.constants import NUM_FBANKS, NUM_FRAMES, CHECKPOINTS_TRIPLET_DIR, BATCH_SIZE
from deep_speaker.conv_models import DeepSpeakerModel
from deep_speaker.eval_metrics import evaluate
from deep_speaker.utils import load_best_checkpoint, enable_deterministic

logger = logging.getLogger(__name__)


def batch_cosine_similarity(x1, x2):
    # https://en.wikipedia.org/wiki/Cosine_similarity
    # 1 = equal direction ; -1 = opposite direction
    mul = np.multiply(x1, x2)
    s = np.sum(mul, axis=1)

    # l1 = np.sum(np.multiply(x1, x1),axis=1)
    # l2 = np.sum(np.multiply(x2, x2), axis=1)
    # as values have have length 1, we don't need to divide by norm (as it is 1)
    return s


def eval_model(working_dir: str, model: DeepSpeakerModel):
    enable_deterministic()
    audio = Audio(working_dir)
    batcher = LazyTripletBatcher(working_dir, NUM_FRAMES, model)
    speakers_list = list(audio.speakers_to_utterances.keys())
    num_negative_speakers = 99
    num_speakers = len(speakers_list)
    y_pred = np.zeros(shape=(num_speakers, num_negative_speakers + 1))  # negatives + positive
    for i, positive_speaker in tqdm(enumerate(speakers_list), desc='test', total=num_speakers):
        # convention id[0] is anchor speaker, id[1] is positive, id[2:] are negative.
        input_data = batcher.get_speaker_verification_data(positive_speaker, num_negative_speakers)
        # batch size is not relevant. just making sure we don't push too much on the GPU.
        predictions = model.m.predict(input_data, batch_size=BATCH_SIZE)
        anchor_embedding = predictions[0]
        for j, other_than_anchor_embedding in enumerate(predictions[1:]):  # positive + negatives
            y_pred[i][j] = batch_cosine_similarity([anchor_embedding], [other_than_anchor_embedding])[0]
        # y_pred[i] = softmax(y_pred[i])
    # could apply softmax here.
    y_true = np.zeros_like(y_pred)  # positive is at index 0.
    y_true[:, 0] = 1.0
    print(np.matrix(y_true))
    print(np.matrix(y_pred))
    print(np.min(y_pred), np.max(y_pred))
    fm, tpr, acc, eer = evaluate(y_pred, y_true)
    return fm, tpr, acc, eer


def test(working_dir, checkpoint_file=None):
    batch_input_shape = [None, NUM_FRAMES, NUM_FBANKS, 1]
    dsm = DeepSpeakerModel(batch_input_shape)
    if checkpoint_file is None:
        checkpoint_file = load_best_checkpoint(CHECKPOINTS_TRIPLET_DIR)
    if checkpoint_file is not None:
        logger.info(f'Found checkpoint [{checkpoint_file}]. Loading weights...')
        dsm.m.load_weights(checkpoint_file, by_name=True)
    else:
        logger.info(f'Could not find any checkpoint in {checkpoint_file}.')
        exit(1)

    fm, tpr, acc, eer = eval_model(working_dir, model=dsm)
    logger.info(f'f-measure = {fm:.3f}, true positive rate = {tpr:.3f}, '
                f'accuracy = {acc:.3f}, equal error rate = {eer:.3f}')
