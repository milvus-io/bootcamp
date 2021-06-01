import os
import uuid
from datetime import timedelta
from operator import itemgetter
from multiprocessing import Process
from typing import List
from minio import Minio
from minio.error import ResponseError
from common.config import MINIO_ADDR, MINIO_ACCESS_KEY, MINIO_SECRET_KEY, MINIO_1ST_BUCKET, MINIO_BUCKET_NUM
from common.config import UPLOAD_FOLDER, ALL_STAGE
from controller.decimation_frame import extract_frame
from controller.vgg import predict
from controller.indexer import MilvusOperator
from common.config import MILVUS_ADDR, MILVUS_PORT
from controller.database import insert2redis, clean_with_video_id
from controller.database import search as redis_search
from common.common import read_status
from common.config import SEARCH_MAGIC_NUM, SEARCH_COUNT_NUM
from controller.database import total_images, total_images_add, total_images_reduce


class Video:
    def __init__(self, MINIO_ADDR, MINIO_ACCESS_KEY, MINIO_SECRET_KEY, MINIO_1ST_BUCKET):
        self.minio_client = Minio(
            MINIO_ADDR,
            access_key=MINIO_ACCESS_KEY,
            secret_key=MINIO_SECRET_KEY,
            secure=False
        )
        self.minio_1st_bucket = MINIO_1ST_BUCKET
        self.max_num = MINIO_BUCKET_NUM
        self.check_before_use()

    def check_before_use(self):
        for i in range(self.max_num):
            current = str(hex(self.max_num-i))[2:]
            if not self.minio_client.bucket_exists("{}-{}".format(self.minio_1st_bucket, current)):
                try:
                    self.minio_client.make_bucket("{}-{}".format(self.minio_1st_bucket, current))
                except ResponseError as err:
                    return err
        if not self.minio_client.bucket_exists(self.minio_1st_bucket):
            try:
                self.minio_client.make_bucket(self.minio_1st_bucket)
            except ResponseError as err:
                return err

    def choose_bucket(self, id):
        for i in range(self.max_num):
            current = str(hex(self.max_num-i))[2:]
            if id.startswith(current):
                return "{}-{}".format(self.minio_1st_bucket, current)
        return self.minio_1st_bucket


    def get_object_data(self, object_name, bucket):
        try:
            data = self.minio_client.presigned_get_object(bucket, object_name)
            return data
        except Exception as err:
            return err

    def all_videos_with_time(self, reverse=False, count=5, page=0):
        res, c = [], -1
        buckets = self.minio_client.list_buckets()
        for bucket in buckets:
            objects = self.minio_client.list_objects_v2(bucket.name)
            for obj in objects:
                c += 1
                if c < count*(page+1):
                    res.append({
                        "name": obj.object_name,
                        "bucket": bucket.name
                    })
                    if len(res) == count:
                        return self.object_datas(res)
        return self.object_datas(res)

    def object_datas(self, objects):
        for x in objects:
            x['data'] = self.get_object_data(x['name'], x['bucket'])
        return objects

    def upload_video(self, ufile, id, bucket=MINIO_1ST_BUCKET):
        with open(ufile, 'rb') as file_data:
            file_stat = os.stat(ufile)
            self.minio_client.put_object(bucket, id, file_data, file_stat.st_size)
            total_images_add()

    def delete_video(self, name):
        bucket = self.choose_bucket(name)
        res = []
        objects = self.minio_client.list_objects_v2(bucket, prefix=name)
        for i in objects:
            self.minio_client.remove_object(bucket, i.object_name)
            total_images_reduce()
            res.append(i.object_name)
        return res

    def videos_by_prefix(self, video_meta=[], bucket=MINIO_1ST_BUCKET):
        res = []
        all_videos = []
        for i in video_meta:
            bucket = self.choose_bucket(i[0])
            objects = self.minio_client.list_objects_v2(bucket, prefix=i[0])
            distance = i[1]
            for obj in objects:
                if obj.object_name not in all_videos:
                    res.append({
                        "name": obj.object_name,
                        "last_modified": obj.last_modified.timestamp(),
                        "data": self.get_object_data(obj.object_name, bucket),
                        "bucket": bucket,
                        "distance": distance
                    })
                    all_videos.append(obj.object_name)
        return res

def list_video(count=5, page=0, reverse=False):
    v = Video(MINIO_ADDR, MINIO_ACCESS_KEY, MINIO_SECRET_KEY, MINIO_1ST_BUCKET)
    try:
        v.check_before_use()
        all_objects = v.all_videos_with_time(reverse=reverse, count=count, page=page)
        return {
            "Total": total_images(),
            "Data": all_objects
        }
    except ResponseError as err:
        return err
    return {}


def pagination(l: List, count: int, page: int) -> dict:
    if len(l)/count == 0:
        data = l
    if page*count < len(l) and len(l) < (page+1) * count:
        data = l[page*count:]
    data = l[page*count:(page+1)*count]
    res = {
        "Total": len(l),
        "Data": data
    }
    return res


def upload_video(filename, model, sess, graph):
    id = uuid.uuid4().hex

    def runner():
        frame_datas = []
        v = Video(MINIO_ADDR, MINIO_ACCESS_KEY, MINIO_SECRET_KEY, MINIO_1ST_BUCKET)
        bucket = v.choose_bucket(id)
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        file_type = filename.split(".")[-1]
        extract_frame(file_path, 1, "prefix", id)
        images = [os.path.join("/tmp", id, x) for x in os.listdir("/tmp/%s"%id)]
        res = predict(images, id, model, sess, graph)
        v.upload_video(file_path, id + "." + file_type, bucket)
        m = MilvusOperator(MILVUS_ADDR, MILVUS_PORT)
        ids = m.insert_feats([x['feat'] for x in res])
        for x in res:
            x["bucket"] = bucket
            frame_datas.append(x)
        insert2redis(ids, frame_datas)
    p = Process(target=runner, args=())
    p.start()
    return {"id": id}


def process_status(id):
    res = read_status(id)
    if res:
        return {
            "state": res[0],
            "percent": res[1]
        }
    # TODO error
    return {}


def delete_video(name):
    v = Video(MINIO_ADDR, MINIO_ACCESS_KEY, MINIO_SECRET_KEY, MINIO_1ST_BUCKET)
    res = v.delete_video(name)
    for i in res:
        id = i.split(".")[0]
        clean_with_video_id(id)
    return {"videos": res}


def search_video(filename, num, model, session, graph):
    r = []
    data = []
    max_num = 2048
    m = MilvusOperator(MILVUS_ADDR, MILVUS_PORT)
    v = Video(MINIO_ADDR, MINIO_ACCESS_KEY, MINIO_SECRET_KEY, MINIO_1ST_BUCKET)
    id = uuid.uuid4().hex
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    res = predict([file_path], "search_" + id, model, session, graph)
    if res:
        def search_more(base, target):
            count = 0
            while count < SEARCH_COUNT_NUM:
                count += 1
                current_search_num = num*SEARCH_MAGIC_NUM*(count+1)
                if current_search_num > 2048:
                    current_search_num = 2048
                results = m.search([res[0]["feat"]], current_search_num)
                current = redis_search(results[0])
                if len(current) >= num:
                    return current[:num-1]
            return current
        results = m.search([res[0]["feat"]], num*SEARCH_MAGIC_NUM)
        current = redis_search(results[0])
        if len(current) < num:
            videos = search_more(num*SEARCH_MAGIC_NUM, num-len(current))
        else:
            videos = current[:num-1]
    videos_metas = v.videos_by_prefix(videos)
    data = videos_metas
    # for video in videos:
    #     video_meta = v.videos_by_prefix(video[0])
    #     if video_meta:
    #         video_meta = video_meta[0]
    #         if video_meta.get('name', "") not in r:
    #             r.append(video_meta.get('name', ""))
    #             video_meta['distance'] = video[1]
    #             data.append(video_meta)
    return {
        "Data": data,
        "Total": len(data)
    }
