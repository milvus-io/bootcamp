import uform
from uform import get_model, Modality
import requests
from io import BytesIO
from PIL import Image
import numpy as np

import pymilvus, time
from pymilvus import (
    MilvusClient, utility, connections,
    FieldSchema, CollectionSchema, DataType, IndexType,
    Collection, AnnSearchRequest, RRFRanker, WeightedRanker
)
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Use the light-weight portable ONNX model.
# Available combinations: cpu & fp32, gpu & fp32, gpu & fp16.
# See Unum's Hugging Face space for more details: 
# https://huggingface.co/unum-cloud

# Define a class to compute embeddings.
class ComputeEmbeddings:
    def __init__(self, modelname):
        # Load the pre-trained model.
        self.model_name = modelname
        self.modalities = [Modality.TEXT_ENCODER, Modality.IMAGE_ENCODER]

        # Get the preprocessing function for the model.
        self.processors, self.models = get_model(self.model_name, modalities=self.modalities)

        # Get the text and image encoders.
        self.model_image = self.models[Modality.IMAGE_ENCODER]
        self.model_text = self.models[Modality.TEXT_ENCODER]
        self.processor_image = self.processors[Modality.IMAGE_ENCODER]
        self.processor_text = self.processors[Modality.TEXT_ENCODER]

    def __call__(self, batch_images=[], batch_texts=[]):

        img_converted_values = []
        text_converted_values = []
        
        # Encode a batch of images.
        if len(batch_images) > 0:

            # Process the images into embeddings.
            image_data = self.processor_image(batch_images)
            image_embeddings = self.model_image.encode(image_data, return_features=False)
            image_embeddings = np.array(image_embeddings / np.linalg.norm(image_embeddings))

            # Milvus requires list of `np.ndarray` arrays of `np.float32` numbers.
            img_converted_values = list(map(np.float32, image_embeddings))
            assert isinstance(img_converted_values, list)
            assert isinstance(img_converted_values[0], np.ndarray)
            assert isinstance(img_converted_values[0][0], np.float32)
            
        # Encode a batch of texts.
        if len(batch_texts) > 0:

            # Process the texts into embeddings.
            try:
                text_data = self.processor_text(batch_texts)
            except:
                text_data = None

            if text_data is not None:    
                text_embeddings = self.model_text.encode(text_data, return_features=False)
                text_embeddings = np.array(text_embeddings / np.linalg.norm(text_embeddings))

                # Milvus requires list of `np.ndarray` arrays of `np.float32` numbers.
                text_converted_values = list(map(np.float32, text_embeddings))
                assert isinstance(text_converted_values, list)
                assert isinstance(text_converted_values[0], np.ndarray)
                assert isinstance(text_converted_values[0][0], np.float32)
        
        return img_converted_values, text_converted_values
    
# Matplotlib function to display images.
def display_images(results, num_rows, num_cols, 
                   fig_size, text_only, image_only, 
                   query_text):
    
    already_displayed_images = set()
    plt.figure(figsize=fig_size)

    for i, result in enumerate(results):
        # Skip already displayed images.
        if result.entity.image_filepath in already_displayed_images:
            continue

        # Otherwise print matching images.
        with Image.open(f"./images/{result.entity.image_filepath}.jpg") as img:
            plt.subplot(num_rows, num_cols, i+1)
            plt.imshow(img)
            plt.title(f"COSINE distance: {round(result.distance,4)}")
            plt.axis('off')

        # Add the image to the set of displayed images.
        already_displayed_images.add(result.entity.image_filepath)

    # Display a super title across images.
    if text_only:
        plt.suptitle(f"Query: {query_text}")
    elif image_only:
        plt.suptitle(f"Query: using image on the left")
    else:
        plt.suptitle(f"Query: {query_text} AND image on the left")
    plt.show();


# Define a search function.
def multi_modal_search(query_text, query_image,
                       embedding_model, col,
                       output_fields,
                       text_only=False, 
                       image_only=False,
                       top_k=2):

    # Embed the question using the same encoder.
    query_img_embeddings, query_text_embeddings = \
        embedding_model(
            batch_images=[query_image],
            batch_texts=[query_text])

    # Prepare the search requests for both vector fields
    image_search_params = {"metric_type": "COSINE"}
    image_req = AnnSearchRequest(
                    query_img_embeddings,
                    "image_vector", image_search_params, limit=top_k)

    text_search_params = {"metric_type": "COSINE"}
    text_req = AnnSearchRequest(
                    query_text_embeddings,
                    "text_vector", text_search_params, limit=top_k)
    
    # Run semantic vector search using Milvus.
    start_time = time.time()

    # User gave an image query only.
    if image_only:
        results = col.hybrid_search(
                    reqs=[image_req, text_req], 
                    rerank=WeightedRanker(1.0, 0.0),
                    limit=top_k, 
                    output_fields=output_fields
                    )
    
    # User gave a text query only.
    elif text_only:
        results = col.hybrid_search(
                    reqs=[image_req, text_req], 
                    rerank=WeightedRanker(0.0, 1.0),
                    limit=top_k, 
                    output_fields=output_fields
                    )
        
    # Use the both the text and image part of query.
    else:
        results = col.hybrid_search(
                    reqs=[image_req, text_req], 
                    rerank=RRFRanker(),
                    limit=top_k, 
                    output_fields=output_fields)

    elapsed_time = time.time() - start_time
    # print(f"Milvus Client search time for {len(dict_list)} vectors: {elapsed_time} seconds")
    print(f"Milvus search time: {elapsed_time} seconds")

    # Currently Milvus only support 1 query in the same hybrid search request, so
    # we inspect res[0] directly. In future release Milvus will accept batch
    # hybrid search queries in the same call.
    results = results[0]

    # Display 2x2 grid of images.
    num_rows = int(round(top_k/2,0))
    if top_k == 2:
        display_images(results, 1, 2, (10,5), 
                       text_only, image_only, query_text)
    else:
        display_images(results, num_rows, 2, (10,10), 
                       text_only, image_only, query_text)

    return results