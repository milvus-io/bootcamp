from towhee import pipeline
from numpy import linalg as LA

# os.environ['KERAS_HOME'] = os.path.abspath(os.path.join('.', 'data'))


class CustomOperator:
    """
    Say something about the ExampleCalass...

    Args:
        args_0 (`type`):
        ...
    """
    def __init__(self):
        #self.img_embedding = pipeline('image-embedding')
        self.yolo_embedding = pipeline('shiyu/img_object_embedding_pytorch_yolov5_resnet50')

    def execute(self, img_path):
        objs = self.yolo_embedding(img_path)
        norm_objs = []
        for feat in objs:
            norm_feat = feat[0] / LA.norm(feat[0])
            norm_objs.append(norm_feat.tolist())
        return norm_objs
