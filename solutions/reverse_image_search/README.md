# Reverse Image Search Based on Milvus and Resnet

In this example we will be going over the code required to perform reverse image search. This example uses a ResNet model to extract image features that are then used with Milvus to build a system that can perform the searches. 

**Here is the [quick start](QUICK_START.md) for a deployable version of a reverse image search.**

## Data

This example uses the PASCAL VOC image set, which contains 17125 images with 20 categories: human; animals (birds, cats, cows, dogs, horses, sheep); transportation (planes, bikes,boats, buses, cars, motorcycles, trains); household (bottles, chairs, tables, pot plants, sofas, TVs)

Dataset size: ~ 2 GB.

Download location: http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar

Directory Structure:  
The file loader used in this requires that the folders containing the images are subfolders. Once the example files are downloaded, place the 

```bash
__data_directory  
    |__sub_folder_1  
    |   |__image1.jpg  
    |   |__image2.jpg  
    |__sub_folder_2  
        |__imageX.jpg  
```

> Note: You can also use other images for testing. This example only requires that the images are PIL compatible.

## Requirements

| Python Packages   | Docker Servers    |
|-                  | -                 |
| PyMilvus          | Milvus-1.0.0      |
| Redis             | Redis             |
| PyTorch           |
| TorchVision       |
| MatPlotLib        |
| PIL               |

## Up and Running


### 1. Start Milvus Server

```bash
$  docker run -d --name milvus_cpu_1.0.0 --network my-net --ip 10.0.0.2 \
-p 19530:19530 \
-p 19121:19121 \
-v /home/$USER/milvus/db:/var/lib/milvus/db \
-v /home/$USER/milvus/conf:/var/lib/milvus/conf \
-v /home/$USER/milvus/logs:/var/lib/milvus/logs \
-v /home/$USER/milvus/wal:/var/lib/milvus/wal \
milvusdb/milvus:1.0.0-cpu-d030521-1ea92e
```

This demo uses Milvus 1.0. Refer to the [Install Milvus](https://milvus.io/docs/v1.0.0/milvus_docker-cpu.md) for how to install Milvus docker. 

### 2. Start Redis Server

```bash
$ docker run --name some-redis -d redis
```

We are using Redis as a metadata storage service. Code can easily be modified to use python dictionary, but that usually does not work in any use case outside of quick examples.

## Code Overview
### Connecting to Servers
We first start off by connecting to the servers. In this case the docker containers are running on localhost and the ports are the default ports. 

```python
milv = milvus.Milvus(host = '127.0.0.1', port = 19530)
red = redis.Redis(host = '127.0.0.1', port=6379, db=0)
```

### Building Collection and Setting Index

The next step involves creating a collection. A collection in Milvus is similar to a table in a relational database, and is used for storing all the vectors. To create a collection, we first must select a name, the dimension of the vectors being stored within, the index_file_size, and metric_type. The index_file_size corresponds to how large each data segmet will be within the collection. More information on this can be found here. The metric_type is the distance formula being used to calculate similarity. In this example we are using the Euclidean distance. 

```python
collection_param = {
            'collection_name': collection_name,
            'dimension': 512,
            'index_file_size': 1024,  # optional
            'metric_type': milvus.MetricType.L2  # optional
            }

status, ok = milv.has_collection(collection_name)
if not ok:
    status = milv.create_collection(collection_param)
```

After creating the collection we want to assign it an index type. This can be done before or after inserting the data. When done before, indexes will be made as data comes in and fills the data segments. In this example we are using IVF_SQ8 which requires the 'nlist' parameter. Each index types carries its own parameters. More info about this param can be found here.

```python
index_param = {
    'nlist': 512
}

status = milv.create_index(collection_name, milvus.IndexType.IVF_SQ8, index_param)
status, index = milv.get_index_info(collection_name)
```
### Processing and Storing Images
In order to store the images in Milvus, we must first run them through the ResNet model. In this case, we are using the pretrained ResNet-18 model provided by PyTorch. In order to get the feature vectors, we must remove the classifying layer that comes at the end. 
```python
model = torch.hub.load('pytorch/vision:v0.9.0', 'resnet18', pretrained=True)
encoder = torch.nn.Sequential(*(list(model.children())[:-1]))
encoder.eval()
```
In this example we are also using a slightly modified dataloader that also returns the file path of the image. With this dataloader we are also transforming the images into what ResNet model takes as input. 

```python
class ImageFolderWithPaths(datasets.ImageFolder):
    def __getitem__(self, index):
        return super(ImageFolderWithPaths, self).__getitem__(index) + (self.imgs[index][0],)

dataset = ImageFolderWithPaths(data_dir, transform=transforms.Compose([
                                                transforms.Resize(256),
                                                transforms.CenterCrop(224),
                                                transforms.ToTensor(),
                                                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]))

dataloader = torch.utils.data.DataLoader(dataset, num_workers=0, batch_size = 256)
```

Inputting the data involves three major steps. First, the images need to be run through the model. This outputs vectors for each image. Second, these vectors are pushed into Milvus. Milvus then returns the corresponding IDs for the vectors. Third, these IDs and the image filepaths are used as the key and value for storage in Redis. Redis is used so that we can return the original image as a result. 

```python
for inputs, labels, paths in dataloader:
    with torch.no_grad():
        output = encoder(inputs).squeeze()
        output = output.numpy()

    status, ids = milv.insert(collection_name=collection_name, records=output)

    if not status.OK():
        print("Insert failed: {}".format(status))
    else:
        for x in range(len(ids)):
            red.set(str(ids[x]), paths[x])
```

### Searching
When searching for an image, we first put the image through the same transformations as the ones used for storing the images. Once transformed, we run the image through the ResNet to get the corresponding vectors. 

```python
transform_ops = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    
embeddings = [transform_ops(Image.open(x)) for x in search_images]
embeddings = torch.stack(embeddings, dim=0)
    
with torch.no_grad():
    embeddings = encoder(embeddings).squeeze().numpy()
```

Then we can use these embeddings in a search. The search requires a few arguments. It needs the name of the collection, the vectors being searched for, how many closest vectors to be returned, and the parameters for the index, in this case nprobe. 

```python
search_sub_param = {
        "nprobe": 16
    }

search_param = {
    'collection_name': collection_name,
    'query_records': embeddings,
    'top_k': 3,
    'params': search_sub_param,
    }

status, results = milv.search(**search_param)
```

The result of this search contains the IDs and corresponding distances of the top_k closes vectors. We can use the IDs in Redis to get the original image. 

```python
if status.OK():
    for x in range(len(results)):
        query_file = search_images[x]
        result_files = [red.get(y.id).decode('utf-8') for y in results[x]]
        distances = [y.distance for y in results[x]]
        show_results(query_file, result_files, distances)
```

This is the basic way to do a reverse image search. Included in the notebook is a way to display the results. 
