import cv2
import uuid
import os
from common.config import DATA_PATH

def extract_frame(file_path, fps, prefix):
    count, frame_count = 0, 0
    cap = cv2.VideoCapture(file_path)
    framerate = cap.get(cv2.CAP_PROP_FPS)
    allframes = int(cv2.VideoCapture.get(cap, int(cv2.CAP_PROP_FRAME_COUNT)))
    success, image = cap.read()
    # print(DATA_PATH + '/' + prefix)
    if not os.path.exists(DATA_PATH + '/' + prefix):
        os.mkdir(DATA_PATH + '/' + prefix)
    images = []
    while success:
        if count % (int(framerate)/fps) == 0:
            file_name = "%s/%s/%d.jpg" % (DATA_PATH, prefix, frame_count+1)
            cv2.imwrite(file_name, image)
            frame_count += 1
            images.append(file_name)
        success, image = cap.read()
        count += 1
    cap.release()
    return images


def main():
    avi = "test.avi"
    prefix = avi.split(".")[0] + "-" + str(uuid.uuid1())
    images = extract_frame(avi, 1, prefix)
    print("images:", images)


if __name__ == '__main__':
    main()