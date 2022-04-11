import towhee
from PIL import Image
from numpy import linalg as LA


class Resnet50:
    """
    Say something about the ExampleCalass...

    Args:
        args_0 (`type`):
        ...
    """

    def resnet50_extract_feat(self, img_path):
        feat = towhee.glob(img_path) \
                .image_decode() \
                .image_embedding.timm(model_name='resnet50') \
                .to_list()
        # Return the normalized embedding([[vec]]) of image
        norm_feat = feat / LA.norm(feat)
        return norm_feat.tolist()[0]
