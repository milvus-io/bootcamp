from datasets import load_dataset
from pymilvus import MilvusClient
import numpy as np
import os
from PIL import Image
import json
from magiclens.magiclens import MagicLensEmbeddding
from cfg import Config

from retrieve import Retriever

encoder = Retriever()


def insert_data():
    config = Config()
    image_folder = f"{config.download_path}" + "/{}"
    client = MilvusClient(uri=config.milvus_uri)
    client.create_collection(
        collection_name=config.collection_name,
        overwrite=True,
        auto_id=True,
        dimension=768,
        enable_dynamic_field=True,
    )
    count = 0
    with open("categories.txt") as fw:
        lines = fw.readlines()
        for line in lines:
            l = line.strip()
            meta_dataset = load_dataset(
                "McAuley-Lab/Amazon-Reviews-2023", f"raw_meta_{l}", split="full"
            )
            for i in range(config.imgs_per_category):
                if len(meta_dataset[i]["images"]["large"]) > 0:
                    print(count)
                    count = count + 1
                    img_name = meta_dataset[i]["images"]["large"][0]
                    name = os.path.basename(img_name)
                    if os.path.exists(image_folder.format(name)) is True:
                        feat = encoder.encode_query(image_folder.format(name), "")
                        spec = json.dumps(meta_dataset[i])
                        res = client.insert(
                            collection_name=config.collection_name,
                            data={
                                "vector": np.array(feat.flatten()),
                                "spec": spec,
                                "name": f"{l}_{i}",
                            },
                        )


insert_data()
