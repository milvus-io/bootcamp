import towhee
from PIL import Image


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
                     .tensor_normalize() \
                     .to_list()
        return feat[0]
