from __future__ import print_function
import requests
from ultralytics import YOLO
import sys
import io
import json
import shutil
import sys
import datetime
import subprocess
import sys
import os
import math
import base64
from time import gmtime, strftime
import random, string   
import time
import psutil
import base64
import uuid
import socket
from pymilvus import connections
from pymilvus import utility
from pymilvus import FieldSchema, CollectionSchema, DataType, Collection
import glob
import torch
from torchvision import transforms
from PIL import Image
import timm
from sklearn.preprocessing import normalize
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from pymilvus import MilvusClient
import os
from pymilvus import model
from pymilvus.model.dense import SentenceTransformerEmbeddingFunction


# For our text built from the current weather at the camera location, we will need something that works well for
# embedding short sentences of Text data.   This Milvus included model does the job well.

model = SentenceTransformerEmbeddingFunction('all-MiniLM-L6-v2',device='cpu' )

# -----------------------------------------------------------------------------
# If we choose we can delete all of our previous runs that contain our YOLO results.
# This will generate a lot of images if we don't delete it, but I like keeping unstructured data forever.
#shutil.rmtree('runs/detect')

# NYC URL for Street Cams List
url = os.environ["NYURL"]

# Milvus Constants
COLLECTION_NAME = 'nycstreetcameras'  # Collection name

# When you are running your Milvus application, you can run it in Zilliz Cloud, Docker, K8 or with Milvus Lite in a 
# local database file.   In my example I am connecting to a Docker deployment on another machine.
MILVUS_URL = "http://192.168.1.153:19530" 

# -----------------------------------------------------------------------------
# We will use a standard feature extractor to vectorize our image.  This utilizes Pytorch and the ResNet-34 (Residual neural network) to prepare
# the vector embedding for our camera image.

# See this great article for more details:   https://medium.com/vector-database/how-to-get-the-right-vector-embeddings-83295ced7f35 

class FeatureExtractor:
    def __init__(self, modelname):
        # Load the pre-trained model
        self.model = timm.create_model(
            modelname, pretrained=True, num_classes=0, global_pool="avg"
        )
        self.model.eval()

        # Get the input size required by the model
        self.input_size = self.model.default_cfg["input_size"]

        config = resolve_data_config({}, model=modelname)
        # Get the preprocessing function provided by TIMM for the model
        self.preprocess = create_transform(**config)

    def __call__(self, imagepath):
        # Preprocess the input image
        input_image = Image.open(imagepath).convert("RGB")  # Convert to RGB if needed
        input_image = self.preprocess(input_image)

        # Convert the image to a PyTorch tensor and add a batch dimension
        input_tensor = input_image.unsqueeze(0)

        # Perform inference
        with torch.no_grad():
            output = self.model(input_tensor)

        # Extract the feature vector
        feature_vector = output.squeeze().numpy()

        return normalize(feature_vector.reshape(1, -1), norm="l2").flatten()

# Prepare the extractor with model:  resnet34.  
extractor = FeatureExtractor("resnet34")

#----- format weather details string
def formatweather(latitude, longitude, weatherfields):
    weathertext = "The current weather observation for {0} [{1}] in {2} @ {3},{4} for {5} is {6} with a temperature of {7}F, a dew point of {8} and relative humidity of {9}% with a wind speed of {10} in a wind direction of {11} with a visibility of {12} at an elevation of {13} and an altimeter reading of {14} for the {15} area."

    try:
        weatherdetails = weathertext.format( weatherfields.setdefault("weathername", "NA"), 
                                             weatherfields.setdefault("weatherid", "NA"), 
                                             weatherfields.setdefault("state", "NY"), 
                                             latitude, longitude, 
                                             weatherfields.setdefault("observationdate","NA"),
                                             weatherfields.setdefault("weather","NA"),
                                             weatherfields.setdefault("temperature","0"),
                                             weatherfields.setdefault("dewpoint","NA"), 
                                             weatherfields.setdefault("relativehumidity","NA"),
                                             weatherfields.setdefault("windspeed","NA"),
                                             weatherfields.setdefault("winddirection","NA"),
                                             weatherfields.setdefault("visibility","NA"),
                                             weatherfields.setdefault("elevation","NA"), 
                                             weatherfields.setdefault("altimeter","NA"), 
                                             weatherfields.setdefault("areadescription'","NA"))
    
    except Exception:
        print("Error in weather formatting")
    
    return weatherdetails
    
# Since we have the latitude and longitude of the street camera, it would be nice to enrich our metadata with the local weather conditions at
# that location.   So with weatherparse, we call our URL, convert the JSON string to a Python structure and parse all the fields we want.

#------ weather lookup
def weatherparse(url):

    weatherfields = {}
    
    if ( url is None ):
        return weatherfields

    try:
        weatherjson = requests.get(url).content
        weatherjsonobject = json.loads(weatherjson)
        weatherfields['creationdate'] = str(weatherjsonobject['creationDate'])
    
        locationfields = weatherjsonobject['location']
    
        weatherfields['areadescription'] = str(locationfields['areaDescription'])
        weatherfields['elevation'] = str(locationfields['elevation'])
        weatherfields['county'] = str(locationfields['county'])
        weatherfields['metar'] = str(locationfields['metar'])
    
        currentobservation = weatherjsonobject['currentobservation']
    
        weatherfields['weatherid']= str(currentobservation['id'])
        weatherfields['weathername'] = str(currentobservation['name'])
        weatherfields['observationdate'] = str(currentobservation['Date'])
        weatherfields['temperature'] = str(currentobservation['Temp'])
        weatherfields['dewpoint'] = str(currentobservation['Dewp'])
        try:
            weatherfields['relativehumidity'] = str(currentobservation['Relh'])
        except Exception:
            print("Error relh missing")
    
        weatherfields['windspeed'] = str(currentobservation['Winds'])
        weatherfields['winddirection'] = str(currentobservation['Windd'])
        weatherfields['gust'] = str(currentobservation['Gust'])
        weatherfields['weather'] = str(currentobservation['Weather'])
        weatherfields['visibility'] = str(currentobservation['Visibility'])
        weatherfields['altimeter'] = str(currentobservation['Altimeter'])
        weatherfields['slp'] = str(currentobservation['SLP'])
        weatherfields['timezone'] = str(currentobservation['timezone'])
        weatherfields['state'] = str(currentobservation['state'])
        weatherfields['windchill'] = str(currentobservation['WindChill'])
    except Exception as ex:
        print("Error building the weather", ex)
    
    return weatherfields
    

# -----------------------------------------------------------------------------
# ultralytics Yolo v8 Model
# this is a pretrained model that will do classifications very well for us, since
# the items we wish to find are cars.

yolomodel = YOLO('yolov8n.pt')  # pretrained YOLOv8n model

# -----------------------------------------------------------------------------
# Connect to Milvus

# Local Docker Server
milvus_client = MilvusClient( uri=MILVUS_URL )

# -----------------------------------------------------------------------------
# Create collection which includes the id, filepath of the image, and image embedding

# The most interesting part of this application is our rich and large data model.   
# We have chosen to make all the weather fields as scalars in our schema.  We could have
# put them in as JSON, but since we know all the fields and want it fixed we choose to explicitly define
# every field and type.   There are also a few metadata fields from our initial REST call to get the camera,
# these include the critical  (latitude,longitude), Road Way Name and Direction of Travel.

fields = [
    FieldSchema(name='id', dtype=DataType.INT64, is_primary=True, auto_id=True),
    FieldSchema(name='latitude', dtype=DataType.VARCHAR, max_length=200),
    FieldSchema(name='longitude', dtype=DataType.VARCHAR, max_length=200),
    FieldSchema(name='name', dtype=DataType.VARCHAR, max_length=200),
    FieldSchema(name='roadwayname', dtype=DataType.VARCHAR, max_length=200),
    FieldSchema(name='directionoftravel', dtype=DataType.VARCHAR, max_length=200),
    FieldSchema(name='videourl', dtype=DataType.VARCHAR, max_length=200),
    FieldSchema(name='url', dtype=DataType.VARCHAR, max_length=200),
    FieldSchema(name='filepath', dtype=DataType.VARCHAR, max_length=200),
    FieldSchema(name='creationdate', dtype=DataType.VARCHAR, max_length=200),
    FieldSchema(name='areadescription', dtype=DataType.VARCHAR, max_length=200),
    FieldSchema(name='elevation', dtype=DataType.VARCHAR, max_length=200), 
    FieldSchema(name='county', dtype=DataType.VARCHAR, max_length=200), 
    FieldSchema(name='metar', dtype=DataType.VARCHAR, max_length=200), 
    FieldSchema(name='weatherid', dtype=DataType.VARCHAR, max_length=200),
    FieldSchema(name='weathername', dtype=DataType.VARCHAR, max_length=200), 
    FieldSchema(name='observationdate', dtype=DataType.VARCHAR, max_length=200), 
    FieldSchema(name='temperature', dtype=DataType.FLOAT), 
    FieldSchema(name='dewpoint', dtype=DataType.VARCHAR, max_length=200), 
    FieldSchema(name='relativehumidity', dtype=DataType.VARCHAR, max_length=200), 
    FieldSchema(name='windspeed', dtype=DataType.VARCHAR, max_length=200), 
    FieldSchema(name='winddirection', dtype=DataType.VARCHAR, max_length=200),
    FieldSchema(name='gust', dtype=DataType.VARCHAR, max_length=200), 
    FieldSchema(name='weather', dtype=DataType.VARCHAR, max_length=200), 
    FieldSchema(name='visibility', dtype=DataType.VARCHAR, max_length=200), 
    FieldSchema(name='altimeter', dtype=DataType.VARCHAR, max_length=200),
    FieldSchema(name='slp', dtype=DataType.VARCHAR, max_length=200), 
    FieldSchema(name='timezone', dtype=DataType.VARCHAR, max_length=200), 
    FieldSchema(name='state', dtype=DataType.VARCHAR, max_length=200), 
    FieldSchema(name='windchill', dtype=DataType.VARCHAR, max_length=200),
    FieldSchema(name='weatherdetails', dtype=DataType.VARCHAR, max_length=8000),    
    FieldSchema(name='image_vector', dtype=DataType.FLOAT_VECTOR, dim=512),
    FieldSchema(name='weather_text_vector', dtype=DataType.FLOAT_VECTOR, dim=384)
]

# As you can see we have two vector fields, one for image and one for weather text.

schema = CollectionSchema(fields=fields)

# If we haven't done so yet, create out collection and add the schema and indexes.
if milvus_client.has_collection(collection_name=COLLECTION_NAME):
    print("Collection Exists.")
else:
    milvus_client.create_collection(COLLECTION_NAME, schema=schema, metric_type="COSINE", auto_id=True)

    index_params = milvus_client.prepare_index_params()
    index_params.add_index(field_name = "image_vector", metric_type="COSINE")
    
    index_params.add_index(
        field_name="id",
        index_type="STL_SORT"
    )
    
    index_params.add_index(
        field_name="weather_text_vector",
        index_type="IVF_FLAT",
        metric_type="L2",
        params={"nlist": 100}
    )
    
    milvus_client.create_index(COLLECTION_NAME, index_params)
    res = milvus_client.get_load_state(
        collection_name = COLLECTION_NAME
    )
    print(res)

# -----------------------------------------------------------------------------
# Access NYC 511 to Get List of Cameras
response = requests.get(url).content

# -----------------------------------------------------------------------------
# json format for NYC result
json_object = json.loads(response)

# -----------------------------------------------------------------------------
# Intialize our local variables
latitude = ""
longitude = ""
strid = ""
strname = ""
directionoftravel = ""
url = ""
videourl = ""
roadwayname = ""

# -----------------------------------------------------------------------------
# Iterate json urls

# We will iterate through all the web camera records where the camera is not diabled or blocked
for jsonitems in json_object:
    if (  not jsonitems['Disabled'] and not jsonitems['Blocked'] ):

        # we set out latitude and longitude to make our call to get the weather forecast
        latitude = str(jsonitems.setdefault("Latitude", "0"))
        longitude = str(jsonitems.setdefault("Longitude", "0"))
        weatherurl = "https://forecast.weather.gov/MapClick.php?lat={0}&lon={1}&unit=0&lg=english&FcstType=json".format(str(latitude), str(longitude))
        weatherfields = weatherparse(weatherurl)
        strid = jsonitems.setdefault("ID", "NA")
        detectiondetails = ""
        strname = jsonitems.setdefault("Name", "NA")
        directionoftravel = jsonitems.setdefault("DirectionOfTravel","NA")
        roadwayname = jsonitems.setdefault("RoadwayName", "NA")
        url = jsonitems.setdefault("Url", " ")
        videourl = jsonitems.setdefault("VideoUrl", " ")
        
        uuid2 = "{0}_{1}".format(strftime("%Y%m%d%H%M%S",gmtime()),uuid.uuid4())
        url = str(url) + "#" + str(uuid2)
        img = requests.get(url)
        strfilename = str(uuid2) + ".png"
        weatherdetails = formatweather(latitude, longitude, weatherfields)
        
        # print(weatherdetails)
        filepath = "camimages/" + strfilename
        if img.status_code == 200:
            with open(filepath, 'wb') as f:
                f.write(img.content)

        results = None
        try:
            results = yolomodel.predict(filepath, stream=False, save=True, imgsz=640, conf=0.5, verbose=False)
        except Exception as e:
            print("An error in yolo predict " + filepath + ":", e)

        if ( results is None):
            continue
            
# -----------------------------------------------------------------------------
# Iterate results
        for result in results:
            outputimage = result.path
            savedir = result.save_dir
            speed = result.speed
            names = result.names
            boxes = result.boxes  # Boxes object for bounding box outputs
            masks = result.masks  # Masks object for segmentation masks outputs
            keypoints = result.keypoints  # Keypoints object for pose outputs
            probs = result.probs  # Probs object for classification outputs
            obb = result.obb  # Oriented boxes object for OBB outputs
            resultfilename = "camimages/yolo{0}.png".format(uuid.uuid4())
            result.save(filename=resultfilename)  # save to disk
            detectiondetails = str(result.verbose())
            strText = ":tada:" + str(strname) + ":" + str(roadwayname) + ":" + str(detectiondetails)

# -----------------------------------------------------------------------------
# Milvus insert
            try:
                imageembedding = extractor(resultfilename)
                weatherembedding = model(weatherdetails)
                if ( videourl is None):
                    videourl = "NA"
                temperature = weatherfields.setdefault("temperature","0.00")
                if ( temperature is None):
                    temperature = 0.00
                if ( roadwayname is None):
                    roadwayname = "NA"
                areadescription = weatherfields.setdefault("areadescription","NA")
                if ( areadescription is None):
                    areadescription = detectiondetails
                else:
                    areadescription = str(areadescription) + " " + str(detectiondetails)
                    
                milvus_client.insert( COLLECTION_NAME, {"image_vector": imageembedding, "weather_text_vector": weatherembedding, 
                                                        "filepath": resultfilename, "url": url,  "latitude": str(latitude), "longitude": str(longitude), 
                                                        "name": strname, "roadwayname": str(roadwayname), "directionoftravel": directionoftravel, 
                                                        "videourl": videourl, "creationdate": weatherfields.setdefault("creationdate","NA"), 
                                                        "areadescription": areadescription,
                "elevation": weatherfields.setdefault("elevation","NA") , 
                "county": weatherfields.setdefault("county","NA"),
                "metar": weatherfields.setdefault("metar","NA"), 
                "weatherid": weatherfields.setdefault("weatherid","NA"), 
                "weathername": weatherfields.setdefault("weathername","NA"),
                "observationdate": weatherfields.setdefault("observationdate","NA"), 
                "temperature": float(temperature),
                "dewpoint": weatherfields.setdefault("dewpoint","NA"), 
                "relativehumidity": weatherfields.setdefault("relativehumidity","NA"),
                "windspeed": weatherfields.setdefault("windspeed","NA"), 
                "winddirection": weatherfields.setdefault("winddirection","NA"),
                "gust": weatherfields.setdefault("gust","NA"),
                "weather": weatherfields.setdefault("weather","NA"), 
                "visibility": weatherfields.setdefault("visibility","NA"), 
                "altimeter": weatherfields.setdefault("altimeter","NA"), 
                "slp": weatherfields.setdefault("slp","NA"), 
                "timezone": weatherfields.setdefault("timezone","NA"), 
                "state": weatherfields.setdefault("state","NA"),
                "windchill": weatherfields.setdefault("windchill","NA"),
                "weatherdetails": str(weatherdetails) })
            except Exception as e:
                print("An error in insert " + areadescription + " :", e)

# -----------------------------------------------------------------------------
