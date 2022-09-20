import towhee


class ResNet50:
    def __init__(self):
        self.pipe = (towhee.dummy_input()
                    .image_decode()
                    .image_embedding.timm(model_name='resnet50')
                    .tensor_normalize()
                    .as_function()
        )
      
    def resnet50_extract_feat(self, img_path):
        return self.pipe(img_path)
