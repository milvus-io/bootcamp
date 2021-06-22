import sys
import os
from diskcache import Cache
from logs import LOGGER

sys.path.append("..")
from config import DEFAULT_TABLE


# Get the path to the image
def get_video(path):
    video = []
    for f in os.listdir(path):
        if ((f.endswith(extension) for extension in
             ['.gif']) and not f.startswith('.DS_Store')):
            video.append(os.path.join(path, f))
    return video


# Get the vector of images
def extract_features(video_dir, model, frame):
    try:
        cache = Cache('./tmp')
        feats = []
        names = []
        video_list = get_video(video_dir)
        total = len(video_list)
        cache['total'] = total
        for i, video_path in enumerate(video_list):
            imgs = frame.extract_frame(video_path)
            for img_path in imgs:
                norm_feat = model.resnet50_extract_feat(img_path)
                feats.append(norm_feat)
                names.append(video_path.encode())
            cache['current'] = i + 1
            print("%d video in total, extracting feature from video No. %d , and the video has %d frames." % (i + 1, total, len(imgs)))
        return feats, names
    except Exception as e:
        LOGGER.error(" Error with extracting feature from image {}".format(e))
        sys.exit(1)

# Combine the id of the vector and the name of the image into a list
def format_data(ids, names):
    data = []
    for i in range(len(ids)):
        value = (str(ids[i]), names[i])
        data.append(value)
    return data

def do_load(table_name, image_dir, model, frame, milvus_client, mysql_cli):
    if not table_name:
        table_name = DEFAULT_TABLE

    vectors, names = extract_features(image_dir, model, frame)
    ids = milvus_client.insert(table_name, vectors)
    mysql_cli.create_mysql_table(table_name)
    mysql_cli.load_data_to_mysql(table_name, format_data(ids, names))
    return len(ids)