import towhee
from towhee.functional.option import _Reason

class ResNet50:
    def __init__(self):
        self.pipe = (towhee.dummy_input()
                    .image_decode()
                    .image_embedding.timm(model_name='resnet50')
                    .tensor_normalize()
                    .as_function()
        )
      
    def resnet50_extract_feat(self, img_path):
        feat = self.pipe(img_path)
        if isinstance(feat, _Reason):
            raise feat.exception
        return feat


if __name__ == "__main__":
    ResNet50().resnet50_extract_feat('https://github.com/towhee-io/towhee/raw/main/towhee_logo.png')
