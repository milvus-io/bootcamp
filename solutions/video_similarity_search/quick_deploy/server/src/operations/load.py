import sys
import os
from diskcache import Cache
from logs import LOGGER

sys.path.append("..")
from config import DEFAULT_TABLE


def get_video(path):
    # Get the path to the image
    video = []
    for f in os.listdir(path):
        if ((f.endswith(extension) for extension in
             ['.gif']) and not f.startswith('.DS_Store')):
            video.append(os.path.join(path, f))
    return video

def extract_features(video_dir, model, frame):
    # Get the vector of images
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
            print(f"{total} video in total, extracting feature from video No. {i + 1} , and the video has {len(imgs)} frames.")
        return feats, names
    except Exception as e:
        LOGGER.error(f"Error with extracting feature from image {e}")
        sys.exit(1)

def format_data(ids, names):
    # Combine the id of the vector and the name of the image into a list
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
