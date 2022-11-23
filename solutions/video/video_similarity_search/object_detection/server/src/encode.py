from PIL import Image
from numpy import linalg as LA
import towhee
import cv2

class CustomOperator:
    """
    Say something about the ExampleCalass...

    Args:
        args_0 (`type`):
        ...
    """
    def execute(self, img_path):
        # Get an image embedding with resnet50 pipeline
        feat = towhee.glob(img_path) \
                .image_decode() \
                .image_embedding.timm(model_name='resnet50') \
                .to_list()
        # Return the normalized embedding([[vec]]) of image
        norm_feat = feat / LA.norm(feat)
        return norm_feat.tolist()[0]

    def yolo(self, img_path):
        # Get objects' embeddings of an image
        boxes, _, _ = towhee.glob(img_path) \
	        .image_decode() \
	        .object_detection.yolov5() \
	        .to_list()[0]
         
        imgs = self.get_imgs_list(img_path, boxes)
        embeddings = towhee.dc(imgs) \
                .image_embedding.timm(model_name='resnet50') \
                .to_list()

        norm_objs = []
        for feat in embeddings:
            norm_feat = feat / LA.norm(feat)
            norm_objs.append(norm_feat.tolist())
        return norm_objs


    @staticmethod
    def get_imgs_list(img_path, boxes):
        img_list = []
        img = cv2.imread(img_path)
        
        for box in boxes:
            tmp_obj = img[int(box[1]):int(box[3]), int(box[0]):int(box[2])]
            img_list.append(towhee._types.Image(tmp_obj, 'BGR'))
        return img_list
