# Face Recognition Bootcamp
Image search is one of the core ideas behind many different technologies, from autonomous cars recognizing objects to iPhones recognizing the correct user. Here you will find how one can easily empower their vision programs using Milvus. Milvus is a world-class distributed vector database. Here, I have added a face recognition demo to demonstrate the ability of Milvus and to teach users step-by-step how to use Milvus in real-life AI applications.

## Try notebook
In this [notebook](face_recognition_bootcamp.ipynb) we will be going over the code required to perform face recognition. This example uses MTCNN to detect faces & FaceNet model for extracting embeddings that are then used with Milvus to build a system that can perform the searches. 

If you want to learn how to use various models and evaluate recall metrics, also providing object detection method. You can refer [this tutorial](https://github.com/towhee-io/examples/blob/main/image/reverse_image_search/2_deep_dive_image_search.ipynb).




## How to Deploy
Here is the [quick start](./quick_deploy) for a deployable version of a face recognition. And you can also run with Docker.

In addition, there is [quick start](mtcnn_face_detection) about image similarity search with face detection.


