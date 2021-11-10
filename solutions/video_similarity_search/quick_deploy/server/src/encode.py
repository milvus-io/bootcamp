from towhee import pipeline
from PIL import Image
from numpy import linalg as LA


class Resnet50:
    """
    Say something about the ExampleCalass...

    Args:
        args_0 (`type`):
        ...
    """
    def __init__(self):
        self.img_embedding = pipeline('image-embedding')

    def resnet50_extract_feat(self, img_path):
        # Return the normalized embedding of the images
        img = Image.open(img_path)
        feat = self.img_embedding(img)
        norm_feat = feat / LA.norm(feat)
        return norm_feat.tolist()[0][0]
