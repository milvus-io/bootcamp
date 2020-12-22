import os
import uuid
import base64
import logging
import urllib.request
import time
import numpy as np
import yaml
import cv2
import paddle.fluid as fluid
from yolov3_detector.yolo_infer import offset_to_lengths
from yolov3_detector.yolo_infer import coco17_category_info, bbox2out
from yolov3_detector.yolo_infer import Preprocess
from common.config import DATA_PATH, COCO_MODEL_PATH, YOLO_CONFIG_PATH


# def temp_directory():
#     return os.path.abspath(os.path.join('.', 'data'))


# COCO_MODEL_PATH = os.path.join(temp_directory(), "yolov3_darknet")
# YOLO_CONFIG_PATH = os.path.join(COCO_MODEL_PATH, "yolo.yml")


class BoundingBox:
    def __init__(self, x1, y1, x2, y2, score, label=None):
        self.x1 = x1
        self.x2 = x2
        self.y1 = y1
        self.y2 = y2
        self.score = score
        self.label = label


def cv2base64(image, fps, path):
    try:
        tmp_file_name = os.path.join(path, "object/%d-%s.jpg" % (fps, uuid.uuid1()))
        cv2.imwrite(tmp_file_name, image)
        with open(tmp_file_name, "rb") as f:
            base64_data = base64.b64encode(f.read())
            base64_data = base64_data.decode("utf-8")
        return base64_data
    except Exception as e:
        err_msg = "Convert cv2 object to base64 failed: "
        logging.error(err_msg, e, exc_info=True)
        raise e


class YOLO_v3:
    def __init__(self):
        self.model_init = False
        self.fps = 0
        self.user_config = self.get_operator_config()
        self.model_path = COCO_MODEL_PATH
        self.config_path = YOLO_CONFIG_PATH
        with open(self.config_path) as f:
            self.conf = yaml.safe_load(f)

        self.infer_prog, self.feed_var_names, self.fetch_targets = fluid.io.load_inference_model(
            dirname=self.model_path,
            executor=self.executor,
            model_filename='__model__',
            params_filename='__params__')
        self.clsid2catid, self.catid2name = coco17_category_info(False)
        self.execute(np.zeros((300, 300, 3), dtype='float32'), DATA_PATH)

    def get_operator_config(self):
        try:
            config = {}
            self.device_str = os.environ.get("device_id", "/cpu:0")
            if "gpu" not in self.device_str.lower():
                self.place = fluid.CPUPlace()
            else:
                gpu_device_id = int(self.device_str.split(':')[-1])
                gpu_mem_limit = float(os.environ.get("gpu_mem_limit", 0.3))
                os.environ['FLAGS_fraction_of_gpu_memory_to_use'] = str(
                    gpu_mem_limit)
                config["gpu_memory_limit"] = gpu_mem_limit
                self.place = fluid.CUDAPlace(gpu_device_id)
            self.executor = fluid.Executor(self.place)
            return config
        except Exception as e:
            logging.error("unexpected error happen during read config",
                          exc_info=True)
            raise e

    def get_bboxes(self, bbox_results, threshold=0.5):
        bboxes = [[]]
        for item in bbox_results:
            box, score, cls = item["bbox"], item["score"], item["category_id"]
            idx = item["image_id"]
            if score > threshold:
                assert idx == 0, "get_bboxes function now must input image = 1"
                bboxes[idx].append(BoundingBox(x1=box[0], y1=box[1],
                                               x2=box[0] + box[2],
                                               y2=box[1] + box[3],
                                               score=score,
                                               label=self.catid2name[int(cls)]))
        return bboxes

    @staticmethod
    def get_obj_image(self, images, bboxes, path):
        obj_images = []
        for i, frame_bboxes in enumerate(bboxes):
            frame_object = []
            for j, bbox in enumerate(frame_bboxes):
                tmp_obj = images[i][int(bbox.y1):int(
                    bbox.y2), int(bbox.x1):int(bbox.x2)]
                frame_object.append(cv2base64(tmp_obj, self.fps, path))
            
            self.fps += 1
            obj_images.append(frame_object)
        return obj_images

    def execute(self, image, path):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        img_data = Preprocess(image,
                              self.conf['arch'],
                              self.conf['Preprocess'])
        data_dict = {k: v for k, v in zip(self.feed_var_names, img_data)}
        outs = self.executor.run(self.infer_prog,
                                 feed=data_dict,
                                 fetch_list=self.fetch_targets,
                                 return_numpy=False)
        out = outs[-1]
        lod = out.lod()
        lengths = offset_to_lengths(lod)
        np_data = np.array(out)

        res = {'bbox': (np_data, lengths), 'im_id': np.array([[0]])}
        bbox_results = bbox2out([res], self.clsid2catid, False)
        bboxes = self.get_bboxes(bbox_results, 0.5)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        objs = self.get_obj_image(self, [image], bboxes, path)
        return objs[0]


def run(detector, path):
    result_images = []
    images = os.listdir(path)
    images.sort()
    start = time.time()
    if not os.path.exists(path + '/object'):
        os.mkdir(path + '/object')

    try:
        for image_path in images:
            if not image_path.endswith(".jpg"):
                continue
            # print(path + '/' + image_path)
            image = cv2.imread(path + '/' + image_path)
            result_images.append(detector.execute(image, path))
    except Exception as e:
        logging.error("something error: %s", str(e), exc_info=True)
    end = time.time()
    logging.info('%s cost: {:.3f}s, get %d results'.format(end - start),
                 "yolov3 detector", len(result_images))
    return result_images


def main():
    detector = YOLO_v3()
    datas = DATA_PATH + '/' + 'test-f1577db8-0dea-11eb-9433-ac1f6ba128da'
    result_images = run(detector, datas)


if __name__ == '__main__':
    main()