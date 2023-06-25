from towhee import pipe, ops


class Resnet50:
    def __init__(self):
        self.image_embedding_pipe = (
            pipe.input('path')
                .map('path', 'img', ops.image_decode.cv2_rgb())
                .map('img', 'embedding', ops.image_embedding.timm(model_name='resnet50'))
                .map('embedding', 'embedding', ops.towhee.np_normalize())
                .output('embedding')
            )

    def resnet50_extract_feat(self, img_path):
        feat = self.image_embedding_pipe(img_path)
        return feat.get()[0]
