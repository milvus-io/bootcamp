import numpy as np
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input as preprocess_input_resnet50
from tensorflow.keras.preprocessing import image
from numpy import linalg as LA


class Resnet50:
    def __init__(self):
        self.input_shape = (224, 224, 3)
        self.weight = 'imagenet'
        self.pooling = 'max'
        self.model_resnet50 = ResNet50(weights='imagenet',
                                       input_shape=(self.input_shape[0], self.input_shape[1], self.input_shape[2]),
                                       pooling=self.pooling, include_top=False)
        self.model_resnet50.predict(np.zeros((1, 224, 224, 3)))

    def resnet50_extract_feat(self, img_path):
        img = image.load_img(img_path, target_size=(self.input_shape[0], self.input_shape[1]))
        img = image.img_to_array(img)
        img = np.expand_dims(img, axis=0)
        img = preprocess_input_resnet50(img)
        feat = self.model_resnet50.predict(img)
        norm_feat = feat[0] / LA.norm(feat[0])
        return norm_feat.tolist()
