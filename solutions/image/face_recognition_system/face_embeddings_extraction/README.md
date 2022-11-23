### Face Embeddings Extraction
This is an implementation Facial Embeddings Extraction Facenet & OpenCV.


#### Setup:

1. Create `dataset` folder
2. Create sub-folders with names of the faces you want to recognize
3. In each sub-folder add the image of the person. There is no need to crop images.
4. Download the openface nn4.small2.v1.t7 model from the open face portal link (https://storage.cmusatyalab.org/openface-models/nn4.small2.v1.t7)
You may also see the accuracy reports in the following link - http://cmusatyalab.github.io/openface/models-and-accuracies/
8. Install the relevant libraries of `opencv-python`, `imutils`, `pickle`.



#### Run the code:

- Extract the 128-D embeddings of the images in the dataset. This uses the Openface implementation of Facenet model. The embeddings are saved in the "embeddings/embeddings.pickle" file

```
python extract_embeddings.py --dataset dataset --embeddings embeddings/embeddings.pickle --detector face_detection_model --embedding-model nn4.small2.v1.t7
```
