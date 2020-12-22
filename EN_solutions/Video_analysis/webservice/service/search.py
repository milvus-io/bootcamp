import logging as log
from common.config import LOGO_TABLE, FACE_TABLE
from indexer.index import milvus_client, search_vectors, get_vector_by_ids
from indexer.tools import connect_mysql, search_by_milvus_id
from frame_extract import extract_frame
import uuid
import os
from common.config import DATA_PATH
from yolov3_detector.paddle_yolo import run, YOLO_v3 as Detector
import numpy as np


def get_object_vector(image_encoder, path):
    images = os.listdir(path)
    images.sort()
    vectors = []
    for image in images:
        vector = image_encoder.execute(path + '/' + image)
        vectors.append(vector)
    return vectors, images


def get_face_vector(face_encoder, path):
    images = os.listdir(path)
    images.sort()
    vectors = []
    for image in images:
        vector = face_encoder.execute(path + '/' + image)
        vectors.append(vector)
    return vectors, images


def get_face_info(conn, cursor, table_name, results):
    info = []
    for entity in results:
        if entity[0].distance < 1.15:
            re = search_by_milvus_id(conn, cursor, table_name, entity[0].id)
            info.append(re)
    return info


def get_object_info(conn, cursor, table_name, results, obj_images):
    info = []
    times = []
    i = 0
    for entities in results:
        if entities[0].distance < 0.65:
            re = search_by_milvus_id(conn, cursor, table_name, entities[0].id)
            info.append(re)
            times.append(obj_images[i])
        i += 1
    return info, times


def do_search_logo(image_encoder, index_client, conn, cursor, table_name, filename):
    detector = Detector()
    if not table_name:
        table_name = LOGO_TABLE

    prefix = filename.split("/")[2].split(".")[0] + "-" + uuid.uuid4().hex
    images = extract_frame(filename, 1, prefix)
    run(detector, DATA_PATH + '/' + prefix)

    vectors, obj_images = get_object_vector(image_encoder, DATA_PATH + '/' + prefix + '/object')
    results = search_vectors(index_client, table_name, vectors, "L2")

    info, times = get_object_info(conn, cursor, table_name, results, obj_images)
    return info, times


def do_only_her(face_encoder, index_client, conn, cursor, table_name, filename):
    if not table_name:
        table_name = FACE_TABLE

    prefix = filename.split("/")[2].split(".")[0] + "-" + uuid.uuid4().hex
    images = extract_frame(filename, 1, prefix)
    vectors, face_images = get_face_vector(face_encoder, DATA_PATH + '/' + prefix)
    global_info = []
    for faces in vectors:
        results = search_vectors(index_client, table_name, faces, "L2")
        info = get_face_info(conn, cursor, table_name, results)
        global_info.append(info)
    return global_info


def do_search_face(face_encoder, index_client, conn, cursor, table_name, filename):
    if not table_name:
        table_name = FACE_TABLE
    faces = face_encoder.execute(filename)
    results = search_vectors(index_client, table_name, faces, "L2")
    info = get_face_info(conn, cursor, table_name, results)
    return info
