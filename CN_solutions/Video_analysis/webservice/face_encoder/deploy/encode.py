import argparse
import cv2
import sys
import numpy as np
from numpy import linalg as LA
from face_encoder.deploy import face_model_muti


class Encode:
	def __init__(self):
		parser = argparse.ArgumentParser(description='face model test')
		parser.add_argument('--image-size', default='112,112', help='')
		parser.add_argument('--model', default='/data1/mia/insight_model/model-r100-ii/model,0', help='path to load model.')
		parser.add_argument('--ga-model', default='', help='path to load model.')
		parser.add_argument('--gpu', default=0, type=int, help='gpu id')
		parser.add_argument('--det', default=0, type=int, help='mtcnn option, 1 means using R+O, 0 means detect from begining')
		parser.add_argument('--flip', default=0, type=int, help='whether do lr flip aug')
		parser.add_argument('--threshold', default=1.24, type=float, help='ver dist threshold')
		args = parser.parse_args()
		self.model = face_model_muti.FaceModel(args)

	def execute(self, img_path):
		input = cv2.imread(img_path)
		imgs = self.model.get_input(input)
		fets = []
		for img in imgs:
			f1 = self.model.get_feature(img)
			f1 = f1 / LA.norm(f1)
			fets.append(f1.tolist())
		return fets