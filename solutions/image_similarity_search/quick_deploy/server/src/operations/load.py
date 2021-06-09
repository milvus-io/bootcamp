import sys
import os

sys.path.append("..")
from config import DEFAULT_TABLE


def get_imgs(path):
    pics = []
    for f in os.listdir(path):
        if ((f.endswith(extension) for extension in
             ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG']) and not f.startswith('.DS_Store')):
            pics.append(os.path.join(path, f))
    return pics


def extract_features(img_dir, model):
    feats = []
    names = []
    img_list = get_imgs(img_dir)
    total = len(img_list)
    for i, img_path in enumerate(img_list):
        norm_feat = model.resnet50_extract_feat(img_path)
        feats.append(norm_feat)
        names.append(img_path.encode())
        current = i + 1
        print("extracting feature from image No. %d , %d images in total" % (current, total))
    return feats, names


def format_data(ids, names):
    data = []
    for i in range(len(ids)):
        value = (str(ids[i]), names[i])
        data.append(value)
    return data


def do_load(table_name, image_dir, model, mil_client, mysql_cli):
    if not table_name:
        table_name = DEFAULT_TABLE
    vectors, names = extract_features(image_dir, model)
    ids = mil_client.insert(table_name, vectors)
    mysql_cli.create_mysql_table(table_name)
    mysql_cli.load_data_to_mysql(table_name, format_data(ids, names))
    return len(ids)
