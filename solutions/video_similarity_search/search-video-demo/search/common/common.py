from diskcache import Cache
from common.config import ALLOWED_EXTENSIONS
from common.config import LOCAL_CACHE_PATH


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def save_status(key, stage, percent):
    cache = Cache(LOCAL_CACHE_PATH)
    with cache as ref:
        ref.set(key, [stage, percent], expire=360000)


def read_status(key):
    cache = Cache(LOCAL_CACHE_PATH)
    with cache as ref:
        return ref.get(key)
