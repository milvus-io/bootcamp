from html import entities
import torch
import os
import pickle
import prepare_data
import argparse
import numpy as np
import pandas as pd
import matplotlib.image as mpimg
from pymilvus import (
    connections,
    utility,
    FieldSchema,
    CollectionSchema,
    DataType,
    Collection,
)
# from milvus import Milvus, IndexType, MetricType, Status
from PIL import Image, ImageDraw
from facenet_pytorch import MTCNN, InceptionResnetV1
from torchvision import datasets
from torch.utils.data import DataLoader
from  matplotlib import pyplot as plt

_HOST = 'localhost'

_PORT = '19530' 

collection_name = 'celebrity_faces_'

_DIM = 512  

_INDEX_FILE_SIZE = 32  

id_to_identity = None

collection = None

# Connecting to Server
# milvus = Milvus(_HOST, _PORT)
connections.connect("default", host=_HOST, port=_PORT)

workers = 0 if os.name == 'nt' else 4
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Running on device: {}'.format(device))

mtcnn = MTCNN(
        image_size=160, margin=0, min_face_size=20,
        thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True, keep_all=True,
        device=device
    )

resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

# Turns all the celeb image data into embeddings.
def preprocess_images():

    def collate_fn(x):
        return x[0]

    dataset = datasets.ImageFolder('./celeb_reorganized')
    dataset.idx_to_class = {i:c for c, i in dataset.class_to_idx.items()}
    loader = DataLoader(dataset, collate_fn=collate_fn, num_workers=workers)


    encoded = []
    identity = []
    count = len(loader)

    for x, y in loader:
        try:
            x_aligned, prob = mtcnn(x, return_prob=True)
        except:
            print(x)
            plt.imshow(x)
            plt.show()
        if x_aligned is not None:
            x_aligned = x_aligned.to(device)
            embeddings = resnet(x_aligned).detach().cpu()
            embeddings = embeddings.numpy()
            encoded.append(embeddings)
            for x in range(embeddings.shape[0]):
                identity.append(dataset.idx_to_class[y])
            if count %1000 == 0:
                print(count, x_aligned.shape, dataset.idx_to_class[y])
            count -= 1
           
    encoded = np.concatenate(encoded, 0)
    encoded = np.squeeze(encoded)
    print(encoded.shape)
    identity = np.array(identity)
    np.save("identity_save.npy", identity)
    np.save("encoded_save.npy", encoded)
    encoded = np.load("encoded_save.npy")
    identity = np.load("identity_save.npy")
    print(encoded.shape, identity.shape)



# Creates a milvus collection
def create_collection():
    global id_to_identity
    global collection
    print("Creating the collection...")
    if not utility.has_collection(collection_name):
        fields = [
        FieldSchema(name='id', dtype=DataType.INT64, descrition='ids', is_primary=True, auto_id=False),
        FieldSchema(name='embedding', dtype=DataType.FLOAT_VECTOR, descrition='embedding vectors', dim=_DIM)
        ]
        schema = CollectionSchema(fields=fields, description='face recognition system')
        collection = Collection(name=collection_name, schema=schema)
        print("Collection created.")
        
        # Indexing the collection
        print("Indexing the Collection...")
        # create IVF_FLAT index for collection.
        index_params = {
            'metric_type':'L2',
            'index_type':"IVF_FLAT",
            'params':{"nlist":4096}
        }
        collection.create_index(field_name="embedding", index_params=index_params)
        print("Collection indexed.")
        return 1  
    else:
        print("Collection present already.")
        collection = Collection(collection_name)
        # first_load()
        # utility.drop_collection(collection_name)
        # print("Collection Dropped")
        try:
            with open ('id_to_class', 'rb') as fp:
                id_to_identity = pickle.load(fp)
            return 0
        except:
            return 1

def split(a, n):
    k, m = divmod(len(a), n)
    return (a[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(n))

# Imports all the celeb embeddings into the created collection
def first_load():
    global id_to_identity
    global collection
    print("Loading in encoded vectors...")
    encoded = np.load("encoded_save.npy")
    identity = np.load("identity_save.npy")

    encoded = np.array_split(encoded, 4, axis=0)
    identity = identity.astype(int)

    identity = np.array_split(identity, 4, axis=0)

    id_to_identity = []

    entities = [0,0]
    embeddings = []
    indexing = []
    counter = 1
    for encode in encoded:
        for embed in encode:
            embeddings.append(embed)
            indexing.append(counter)
            counter += 1
            # print(counter)

    indexing = list(split(indexing, 5))
    embeddings = list(split(embeddings, 5))

    for i in range(5):
        entities[0] = indexing[i]
        entities[1] = embeddings[i]
        print("Initiating Data Insertion {}".format(i))
        print(collection.insert(entities))
        print("Data Inserted {}".format(i))

    for x in range(len(encoded)):
        for z in range(len(indexing)):
            id_to_identity.append((indexing[z], identity[x][z]))
    print("Id to identity Done")
    collection.load()
    print("Vectors loaded")

    with open('id_to_class', 'wb') as fp:
        pickle.dump(id_to_identity, fp)
    print("Vectors loaded in.")

# Gets embeddings for all the faces in the image. 
def get_image_vectors(file_loc):
    img = Image.open(file_loc)
    bbx, prob = mtcnn.detect(img)
    embeddings = None
    if (bbx is not None):
        face_cropped = mtcnn.extract(img,bbx,None).to(device)
        embeddings = resnet(face_cropped).detach().cpu()
        embeddings = embeddings.numpy()
        draw = ImageDraw.Draw(img)
        for i, box in enumerate(bbx):
            draw.rectangle(box.tolist(), outline=(255,0,0))
            draw.text((box.tolist()[0] + 2,box.tolist()[1]), "Face-" + str(i), fill=(255,0,0))

    return embeddings, img


# Search for the nearest neighbor of the given image. 
def search_image(file_loc):
    global collection
    query_vectors, insert_image = get_image_vectors(file_loc)

    print("Searching for image... ")
    search_params = {
        "params": {"nprobe": 2056},
    }
    results = collection.search(query_vectors, "embedding", search_params, limit=3)
    # if(len(results[0]) > 1 or len(results[1]) > 1):
    #     print("Similar images found....!!!")
    # print(results)

    if results:
        temp = []
        plt.imshow(insert_image)
        for x in range(len(results)):
            for i, v in id_to_identity:
                if results[x][0].id == i:
                    temp.append(v)
        # print(temp)
        for i, x in enumerate(temp):
            fig = plt.figure()
            fig.suptitle('Face-' + str(i) + ", Celeb Folder: " + str(x))
            currentFolder = './celeb_reorganized/' + str(x)
            total = min(len(os.listdir(currentFolder)), 6)

            for i, file in enumerate(os.listdir(currentFolder)[0:total], 1):
                fullpath = currentFolder+ "/" + file
                img = mpimg.imread(fullpath)
                plt.subplot(2, 3, i)
                plt.imshow(img)
        plt.show(block = False)
        if(len(temp))!=0:
            print("Wohoo, Similar Images found!🥳️")
        print(temp)

# Delete the collection
def delete_collection():
    utility.drop_collection(collection_name)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Find out which celebrities.')
    
    parser.add_argument('filename')

    args = parser.parse_args()

    test_file = args.filename

    # delete_collection()
    if not os.path.isdir("./celeb_reorganized"):
        print("Unzipping Data...")
        prepare_data.unzip()
        print("Reorganizing Data...")
        prepare_data.reorganize()
    if not (os.path.isfile("./encoded_save.npy") and os.path.isfile("./identity_save.npy")):
        print("Processing Images...")
        delete_collection()
        preprocess_images()
    if not (os.path.isfile("./id_to_class")):
        delete_collection()
    if create_collection():
        first_load()
    search_image(test_file)
    plt.show()