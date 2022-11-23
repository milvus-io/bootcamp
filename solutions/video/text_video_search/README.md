# Text-Video Retrieval

The objective of video retrieval is as follows: given a text query and a pool of candidate videos, select the video which corresponds to the text query. Typically, the videos are returned as a ranked list of candidates and scored via document retrieval metrics.

This text-video retrieval example mainly consists of two notebooks, and I think everyone can learn the basic operations of Towhee and Milvus through the [**getting started notebook**](https://github.com/towhee-io/examples/blob/main/video/text_video_retrieval/1_text_video_retrieval_engine.ipynb). And the [**deep dive notebook**](https://github.com/towhee-io/examples/blob/main/video/text_video_retrieval/2_deep_dive_text_video_retrieval.ipynb) will make the engine more feasible in production.

## Learn from Notebook

- [Getting started](https://github.com/towhee-io/examples/blob/main/video/text_video_retrieval/1_text_video_retrieval_engine.ipynb)

In this notebook you will get prerequisites, build and use a basic text-video retrieval system, visualize sample results, make simple optimizations, and measure the system with performance metrics.

- [Deep Dive](https://github.com/towhee-io/examples/blob/main/video/text_video_retrieval/2_deep_dive_text_video_retrieval.ipynb)

In this notebook, you will learn how to reduce resource usage, speed up system,  and ensure stability.
