import redis
import json
from itertools import zip_longest
from common.config import REDIS_ADDR, REDIS_PORT, REDIS_DB


def batcher(iterable, n):
    args = [iter(iterable)] * n
    return zip_longest(*args)


def insert2redis(ids, dataList):
    # dataList [{"videoId": id, "feat": feat, "name": name}]
    r = redis.Redis(host=REDIS_ADDR, port=REDIS_PORT, db=REDIS_DB)
    if len(ids) != len(dataList):
        # TODO return error
        return ""
    for k, v in enumerate(ids):
        r_key = v
        r_value = dataList[k]
        r.set(r_key, json.dumps(r_value))


def clean_with_video_id(id):
    r = redis.StrictRedis(host=REDIS_ADDR, port=REDIS_PORT, db=REDIS_DB)
    for keybatch in batcher(r.scan_iter('*'), 500):
        for i in keybatch:
            if i:
                if json.loads(r.get(i)).get("videoId") == id:
                    r.delete(i)

def total_images():
    r = redis.StrictRedis(host=REDIS_ADDR, port=REDIS_PORT, db=REDIS_DB, decode_responses=True)
    total = r.get("IMG_TOTAL")
    if not total:
        r.set("IMG_TOTAL", 0)
        return 0
    return int(total)

def total_images_add():
    r = redis.StrictRedis(host=REDIS_ADDR, port=REDIS_PORT, db=REDIS_DB, decode_responses=True)
    total = r.get("IMG_TOTAL")
    if not total:
        r.set("IMG_TOTAL", 1)
        return 1
    r.set("IMG_TOTAL", int(total)+1)
    return int(total)+1

def total_images_reduce():
    r = redis.StrictRedis(host=REDIS_ADDR, port=REDIS_PORT, db=REDIS_DB, decode_responses=True)
    total = r.get("IMG_TOTAL")
    if not total:
        r.set("IMG_TOTAL", 0)
        return 0
    r.set("IMG_TOTAL", int(total)-1)
    return int(total)-1

def search(ids):
    res = []
    r = redis.Redis(host=REDIS_ADDR, port=REDIS_PORT, db=REDIS_DB)
    for i in ids:
        v = r.get(i.id)
        if v and json.loads(v).get('videoId') not in res:
            res.append([json.loads(v).get('videoId'), i.distance])
    return res
