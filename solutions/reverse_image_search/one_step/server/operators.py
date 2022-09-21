import sys
from glob import glob
from diskcache import Cache
from config import DEFAULT_TABLE
from logs import LOGGER


def do_upload(table_name, img_path, model, milvus_client):
    try:
        if not table_name:
            table_name = DEFAULT_TABLE
        milvus_client.create_collection(table_name)
        feat = model.resnet50_extract_feat(img_path)
        ids = milvus_client.insert(table_name, [img_path], [feat])
        return ids
    except Exception as e:
        LOGGER.error(f"Error with upload : {e}")
        sys.exit(1)


def extract_features(img_dir, model):
    img_list = []
    for path in ['/*.png', '/*.jpg', '/*.jpeg', '/*.PNG', '/*.JPG', '/*.JPEG']:
        img_list.extend(glob(img_dir+path))
    try:
        if len(img_list) == 0:
            raise FileNotFoundError(f"There is no image file in {img_dir} and endswith ['/*.png', '/*.jpg', '/*.jpeg', '/*.PNG', '/*.JPG', '/*.JPEG']")
        cache = Cache('./tmp')
        feats = []
        names = []
        total = len(img_list)
        cache['total'] = total
        for i, img_path in enumerate(img_list):
            try:
                norm_feat = model.resnet50_extract_feat(img_path)
                feats.append(norm_feat)
                names.append(img_path)
                cache['current'] = i + 1
                print(f"Extracting feature from image No. {i + 1} , {total} images in total")
            except Exception as e:
                LOGGER.error(f"Error with extracting feature from image:{img_path}, error: {e}")
                continue
        return feats, names
    except Exception as e:
        LOGGER.error(f"Error with extracting feature from image {e}")
        sys.exit(1)


def do_load(table_name, image_dir, model, milvus_client):
    if not table_name:
        table_name = DEFAULT_TABLE
    milvus_client.create_collection(table_name)
    vectors, paths = extract_features(image_dir, model)
    ids = milvus_client.insert(table_name, paths, vectors)
    return len(ids)


def do_search(table_name, img_path, top_k, model, milvus_client):
    try:
        if not table_name:
            table_name = DEFAULT_TABLE
        feat = model.resnet50_extract_feat(img_path)
        vectors = milvus_client.search_vectors(table_name, [feat], top_k)
        paths = [str(x.id) for x in vectors[0]]
        distances = [x.distance for x in vectors[0]]
        return paths, distances
    except Exception as e:
        LOGGER.error(f"Error with search : {e}")
        sys.exit(1)


def do_count(table_name, milvus_cli):
    if not table_name:
        table_name = DEFAULT_TABLE
    try:
        if not milvus_cli.has_collection(table_name):
            return None
        num = milvus_cli.count(table_name)
        return num
    except Exception as e:
        LOGGER.error(f"Error with count table {e}")
        sys.exit(1)


def do_drop(table_name, milvus_cli):
    if not table_name:
        table_name = DEFAULT_TABLE
    try:
        if not milvus_cli.has_collection(table_name):
            return f"Milvus doesn't have a collection named {table_name}"
        status = milvus_cli.delete_collection(table_name)
        return status
    except Exception as e:
        LOGGER.error(f"Error with drop table: {e}")
        sys.exit(1)
