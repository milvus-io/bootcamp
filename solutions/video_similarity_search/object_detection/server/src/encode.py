from PIL import Image
from numpy import linalg as LA
from towhee import pipeline

class CustomOperator:
    """
    Say something about the ExampleCalass...

    Args:
        args_0 (`type`):
        ...
    """
    def __init__(self):
        self.resnet_embedding=pipeline('image-embedding')
        self.yolo_embedding = pipeline('shiyu/img_object_embedding_pytorch_yolov5_resnet50')

    def execute(self, img_path):
        # Get an image embedding with resnet50 pipeline
        img = Image.open(img_path)
        feat = self.resnet_embedding(img)
        norm_feat = feat[0] / LA.norm(feat[0])
        return norm_feat.tolist()[0]

    def yolo(self, img_path):
        # Get objects' embeddings of an image
        objs = self.yolo_embedding(img_path)
        norm_objs = []
        for feat in objs:
            norm_feat = feat[0] / LA.norm(feat[0])
            norm_objs.append(norm_feat.tolist())
        return norm_objs
