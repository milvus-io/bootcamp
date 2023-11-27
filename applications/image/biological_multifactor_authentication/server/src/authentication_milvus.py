from pymilvus import (
    connections,
    utility,
    FieldSchema,
    CollectionSchema,
    DataType,
    Collection,
)
import insightface
import cv2
import os
# from speechbrain.pretrained import EncoderClassifier
from voice_embedding import encode_voices
from moviepy.editor import *

def encode_faces(videoinpath):
    capture= cv2.VideoCapture(videoinpath)
    model = insightface.app.FaceAnalysis()
    model.prepare(ctx_id=0, det_thresh=0.45)
    flag=False
    i=0
    if capture.isOpened():
        while True:
            ret,img_src=capture.read()
            if not ret:break
            if i%20:
                i = i + 1
                continue
            res = model.get(img_src)
            if(len(res)==1):
                return res[0].embedding
            i = i + 1
        return ["视频中没有人脸或者有多个人脸"]
    else:
        return ["视频打开失败"]
    
def mp4_to_mp3(path):
    ffmpeg_tools.ffmpeg_extract_audio(path, 'audio_.wav')
    # video = VideoFileClip(path)
    # audio = video.audio
    # audio.write_audiofile('audio_.wav')

# flag, face_embedding = encode_faces()
# print(face_embedding.shape)

# def encode_voices():
#     classifier = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb")
#     signal =classifier.load_audio(path = './static/output.wav')
#     embeddings = classifier.encode_batch(signal)
#     return embeddings
    
# voice_embedding = encode_voices()
# print(voice_embedding.shape)
    

collection_name_face = 'face_authentication'
collection_name_voice = 'voice_authentication'

collection_face = None
collection_voice = None

MILVUS_HOST = os.getenv('MILVUS_HOST',default='localhost')
MILVUS_PORT = os.getenv('MILVUS_PORT',default=19530)

connections.connect("default", host=MILVUS_HOST, port=MILVUS_PORT)

# Delete the collection
def delete_collection():
    utility.drop_collection(collection_name_face)
    utility.drop_collection(collection_name_voice)

# Creates a milvus collection
def create_collection():
    
    global collection_face
    global collection_voice

    print("Creating the face collection...")
    if not utility.has_collection(collection_name_face):
        fields = [
        FieldSchema(name='name', dtype=DataType.VARCHAR, descrition='name',is_primary=True, auto_id=False, max_length=100),
        FieldSchema(name='embedding', dtype=DataType.FLOAT_VECTOR, descrition='embedding vectors', dim=512)
        ]
        schema = CollectionSchema(fields=fields, description='face recognition system')
        collection_face = Collection(name=collection_name_face, schema=schema)
        print("Face collection created.")
        
        # Indexing the collection
        print("Indexing the face collection...")
        # create IVF_FLAT index for collection.
        index_params = {
            'metric_type':'L2',
            'index_type':"IVF_FLAT",
            'params':{"nlist":4096}
        }
        collection_face.create_index(field_name="embedding", index_params=index_params)
        print("Face collection indexed.")
    else:
        print("Face collection present already.")
        collection_face = Collection(collection_name_face)


    print("Creating the voice collection...")
    if not utility.has_collection(collection_name_voice):
        fields = [
        FieldSchema(name='name', dtype=DataType.VARCHAR, descrition='name',is_primary=True, auto_id=False, max_length=100),
        FieldSchema(name='embedding', dtype=DataType.FLOAT_VECTOR, descrition='embedding vectors', dim=192)
        ]
        schema = CollectionSchema(fields=fields, description='voice recognition system')
        collection_voice = Collection(name=collection_name_voice, schema=schema)
        print("Voice collection created.")
        
        # Indexing the collection
        print("Indexing the voice collection...")
        # create IVF_FLAT index for collection.
        index_params = {
            'metric_type':'L2',
            'index_type':"IVF_FLAT",
            'params':{"nlist":4096}
        }
        collection_voice.create_index(field_name="embedding", index_params=index_params)
        print("Voice collection indexed.")
    else:
        print("Voice collection present already.")
        collection_voice = Collection(collection_name_voice)

def insert_embedding(name):

    mp4_to_mp3('./media_.mp4')

    print(type(name))
    print(name)
    
    global collection_face
    global collection_voice
    
    entities = [0,0]
    
    face_embedding = encode_faces('./media_.mp4')
    face_embedding = face_embedding.reshape(1,-1)
    entities[0] = [name]
    entities[1] = face_embedding
    print(collection_face.insert(entities))

    voice_embedding = encode_voices('./audio_.wav')
    voice_embedding = voice_embedding.reshape(1,-1)
    print(voice_embedding.shape)
    entities[0] = [name]
    entities[1] = voice_embedding
    print(collection_voice.insert(entities))

def search_collection():

    mp4_to_mp3('./media_.mp4')
    
    face_embedding = encode_faces('./media_.mp4')
    face_embedding = face_embedding.reshape(1,-1)
    voice_embedding = encode_voices('./audio_.wav')
    voice_embedding = voice_embedding.reshape(1,-1)
    
    global collection_face
    global collection_voice

    print("Start loading")
    collection_face.load()

    print("Searching for image... ")
    search_params = {
        "metric_type": "L2",
        "params": {"nprobe": 2056},
    }
    results1 = collection_face.search(face_embedding, "embedding", search_params, limit=2)
    if(len(results1[0])==0):
        return False , " "

    print("Start loading")
    collection_voice.load()

    print("Searching for image... ")
    search_params = {
        "metric_type": "L2",
        "params": {"nprobe": 2056},
    }
    results2 = collection_voice.search(voice_embedding, "embedding", search_params, limit=2)
    if(len(results2[0])==0):
        return False , " "
    
    print(results1)
    print(results2)
    if(results1[0][0].id == results2[0][0].id and results1[0][0].distance<=400 and results2[0][0].distance<=1500000):
        print(results1[0][0])
        print(results2[0][0])
        return True, results1[0][0].id
    else:
        return False, " "

    
# delete_collection()
# create_collection()
# search_collection()
# insert_embedding("tour")