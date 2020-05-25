import numpy as np

from batcher import LazyTripletBatcher
from constants import NUM_FBANKS, NUM_FRAMES
from conv_models import DeepSpeakerModel
from triplet_loss import deep_speaker_loss


def main2():
    batch_input_shape = [None, NUM_FRAMES, NUM_FBANKS, 1]
    dsm = DeepSpeakerModel(batch_input_shape, include_softmax=False)
    dsm.m.compile(optimizer='adam', loss=deep_speaker_loss)
    dsm.m.load_weights('/Users/premy/deep-speaker/ResCNN_checkpoint_102.h5', by_name=True)
    dsm.m.summary()
    batcher = LazyTripletBatcher(working_dir='/Users/premy/deep-speaker', max_length=NUM_FRAMES, model=dsm)
    bs = 18

    print(np.mean(
        [dsm.m.evaluate(*batcher.get_batch_train(batch_size=bs), batch_size=bs, verbose=0) for _ in range(100)]))
    print(
        np.mean([dsm.m.evaluate(*batcher.get_batch_test(batch_size=bs), batch_size=bs, verbose=0) for _ in range(100)]))
    print(np.mean(
        [dsm.m.evaluate(*batcher.get_random_batch(batch_size=bs, is_test=False), batch_size=bs, verbose=0) for _ in
         range(100)]))
    print(np.mean(
        [dsm.m.evaluate(*batcher.get_random_batch(batch_size=bs, is_test=True), batch_size=bs, verbose=0) for _ in
         range(100)]))


if __name__ == '__main__':
    main2()
