# Audio Fingerprint

Audio fingerprinting is the process of extracting features to represent audio in digital numbers. Normally the process cuts the input audio into shorter clips with a fixed length. Then it converts each clip to a single fingerprint piece in a fixed size. With all small pieces together ordered by timestamps, a complete fingerprint is generated for the input audio. With audio fingerprints as identities, a system can recognize music with various transformations.

This audio fingerprint bootcamp mainly includes notebooks for music detection using neural network for audio fingerprinting. You can learn about music detection solutions and basic concepts of [Towhee](https://towhee.io/) and [Milvus](https://milvus.io/) from these notebook tutorials.

## Learn from Notebook

- [Audio Fingerprint I: Build a Demo with Towhee & Milvus](https://github.com/towhee-io/examples/tree/main/audio/audio_fingerprint/audio_fingerprint_beginner.ipynb)

In this notebook, you will build a basic music detection system using a pretrained deep learning model with sample data (100 candidate music), and measure the system performance with example queries (100 audio with noise). At the end, you are able to build up an online music detection system with simple user interface.

- [Audio Fingerprint II: Music Detection with Temporal Localization](https://github.com/towhee-io/examples/tree/main/audio/audio_fingerprint/audio_fingerprint_advanced.ipynb)

In this notebook, you will learn about temporal localization with Towhee. With temporal localization, you are able to identify overlapping ranges between two audios if there exists any similar parts. This method can also be used as an additional postprocessing step of querying in music detection system. This notebook applies the temporal localization operator by Towhee to the music detection system, and evaluates the system performance.
