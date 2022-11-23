import towhee
import cv2


class CustomOperator:
    def execute(self, img_path):
        boxes, _, _ = towhee.glob(img_path) \
	                        .image_decode() \
	                        .object_detection.yolov5() \
	                        .to_list()[0]

        imgs = self.get_imgs_list(img_path, boxes)
        norm_embeddings = towhee.dc(imgs) \
                                .image_embedding.timm(model_name='resnet50') \
                                .tensor_normalize() \
                                .to_list()

        return norm_embeddings


    @staticmethod
    def get_imgs_list(img_path, boxes):
        img_list = []
        img = cv2.imread(img_path)

        for box in boxes:
            tmp_obj = img[int(box[1]):int(box[3]), int(box[0]):int(box[2])]
            img_list.append(towhee._types.Image(tmp_obj, 'BGR'))
        return img_list
