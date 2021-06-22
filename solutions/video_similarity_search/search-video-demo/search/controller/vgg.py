import os
import numpy as np
from numpy import linalg as LA
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input as preprocess_input_vgg
from keras.preprocessing import image

from common.config import STAGE_PREDICT
from common.common import save_status


class VGGNet:
    def __init__(self):
        self.input_shape = (224, 224, 3)
        self.weight = 'imagenet'
        self.pooling = 'max'
        self.model_vgg = VGG16(weights=self.weight,
                               input_shape=(self.input_shape[0], self.input_shape[1], self.input_shape[2]),
                               pooling=self.pooling,
                               include_top=False)
        self.model_vgg.predict(np.zeros((1, 224, 224, 3)))

    def vgg_extract_feat(self, img_path):
        img = image.load_img(img_path, target_size=(self.input_shape[0], self.input_shape[1]))
        img = image.img_to_array(img)
        img = np.expand_dims(img, axis=0)
        img = preprocess_input_vgg(img)
        feat = self.model_vgg.predict(img)
        norm_feat = feat[0] / LA.norm(feat[0])
        norm_feat = [i.item() for i in norm_feat]
        return norm_feat


def single_feature_extract(img_path):
    model = VGGNet()
    norm_feat = model.vgg_extract_feat(img_path)
    img_name = os.path.split(img_path)[1]
    return norm_feat, img_name


def extract_with_session(img_path, model, sess, graph):
     with sess.as_default():
        with graph.as_default():
            img = image.load_img(img_path, target_size=(224, 224))
            img = image.img_to_array(img)
            img = np.expand_dims(img, axis=0)
            img = preprocess_input_vgg(img)
            feat = model.predict(img)
            norm_feat = feat[0] / LA.norm(feat[0])
            norm_feat = [i.item() for i in norm_feat]
            img_name = os.path.split(img_path)[1]
            return norm_feat, img_name


def predict(images, id, model=None, session=None, graph=None):
    res = []
    count, all_images = 0, len(images)
    for i in images:
        #        feat, name = extract_with_session(i, model, session, graph)
        feat, name = single_feature_extract(i)
        data = {"videoId": id, "name": name, "feat": feat}
        count += 1
        save_status(id, STAGE_PREDICT, count/all_images)
        res.append(data)
    return res
