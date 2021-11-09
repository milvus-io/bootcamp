import cv2
import uuid
import os
from config import UPLOAD_PATH


class FrameExtract:
    """
    Say something about the ExampleCalass...

    Args:
        args_0 (`type`):
        ...
    """
    def __init__(self, fps=1):
        self.fps = fps

    def extract_frame(self, file_path):
        prefix = file_path.split("/")[-1].split(".")[0] + "-" + str(uuid.uuid1())
        count, frame_count = 0, 0
        cap = cv2.VideoCapture(file_path)
        framerate = cap.get(cv2.CAP_PROP_FPS)
        # allframes = int(cv2.VideoCapture.get(cap, int(cv2.CAP_PROP_FRAME_COUNT)))
        success, image = cap.read()
        if not os.path.exists(UPLOAD_PATH + '/' + prefix):
            os.mkdir(UPLOAD_PATH + '/' + prefix)
        images = []
        while success:
            if count % (int(framerate)/self.fps) == 0:
                frames = str(frame_count+1).zfill(6)
                file_name = f'{UPLOAD_PATH}/{prefix}/{frames}.jpg'
                cv2.imwrite(file_name, image)
                frame_count += 1
                images.append(file_name)
            success, image = cap.read()
            count += 1
        cap.release()
        object_path = f"{UPLOAD_PATH}/{prefix}/"
        return object_path, images
