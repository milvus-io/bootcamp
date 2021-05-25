import os
from encoder.utils import get_imlist
from diskcache import Cache
from common.config import DEFAULT_CACHE_DIR


def feature_extract(database_path, model):
    cache = Cache(DEFAULT_CACHE_DIR)
    feats = []
    names = []
    img_list = get_imlist(database_path)
    model = model
    for i, img_path in enumerate(img_list):
        norm_feat = model.resnet50_extract_feat(img_path)
        img_name = os.path.split(img_path)[1]
        feats.append(norm_feat)
        names.append(img_name.encode())
        current = i + 1
        total = len(img_list)
        cache['current'] = current
        cache['total'] = total
        print("extracting feature from image No. %d , %d images in total" % (current, total))
    return feats, names
