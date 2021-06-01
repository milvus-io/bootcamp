import os
import cv2
from common.common import save_status
from common.config import STAGE_EXTRACT


def extract_frame(file_path, fps, prefix, id):
    count, frame_count = 0, 0
    cap = cv2.VideoCapture(file_path)
    framerate = cap.get(cv2.CAP_PROP_FPS)
    allframes = int(cv2.VideoCapture.get(cap, int(cv2.CAP_PROP_FRAME_COUNT)))
    success, image = cap.read()
    os.mkdir("/tmp/%s" % id)
    while success:
        if count % (int(framerate)/fps) == 0:
            cv2.imwrite("/tmp/%s/%s%d.jpg" % (id, prefix, frame_count), image)
            frame_count += 1
        success, image = cap.read()
        count += 1
        save_status(id, STAGE_EXTRACT, count/allframes)
    cap.release()
