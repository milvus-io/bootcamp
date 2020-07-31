import argparse
import cv2
import sys
import os
import numpy as np
sys.path.append(os.path.join(os.path.dirname(__file__)))
import face_model

parser = argparse.ArgumentParser(description='face model test')
# general
model_dir = os.path.join(os.path.dirname(__file__), 'models/model-r100-ii/model')


parser.add_argument('--image-size', default='112,112', help='')
parser.add_argument('--model', default= model_dir + ',0', help='path to load model.')
parser.add_argument('--ga-model', default='', help='path to load model.')
parser.add_argument('--gpu', default=-1, type=int, help='gpu id')
parser.add_argument('--det', default=0, type=int, help='mtcnn option, 1 means using R+O, 0 means detect from begining')
parser.add_argument('--flip', default=0, type=int, help='whether do lr flip aug')
parser.add_argument('--threshold', default=1.24, type=float, help='ver dist threshold')
args = parser.parse_args()



def img_to_vec(img):
    model = face_model.FaceModel(args)
    img = cv2.imread(img)
    img = model.get_input(img)
    if img is None:
        print('------------------No face detected or Multiple faces detected')
        return None
    else:
        print('----------------feature')
        f1 = model.get_feature(img)
        return f1.tolist()



