# Image Similarity Search with Milvus 🖼️

This demo implements an image similarity search application using Streamlit, Milvus, and a pre-trained ResNet model. Users can upload an image, crop it to focus on the region of interest, and search for similar images from a pre-built database.

## Features
- Upload and crop images to define the region of interest.
- Extract features using a pre-trained ResNet model.
- Search for similar images using Milvus for efficient similarity search.
- Display search results along with similarity scores.

## Code Structure
```text
image_search_with_milvus/
│
├── app.py                  # Main Streamlit application
├── insert.py               # Script to download and unzip image data
├── milvus_utils.py         # Milvus-related operations
├── encoder.py              # Feature extraction and model loading
├── requirements.txt        # List of dependencies
```

- app.py: The main Streamlit application file where the user interface is defined and the image similarity search is performed.
- insert.py: This script handles the downloading and unzipping of image data required for the application as well as inserting image embeddings.
- milvus_utils.py: Includes functions for interacting with the Milvus database, such as setting up Milvus client.
- encoder.py: Contains the FeatureExtractor class, which is responsible for extracting feature vectors from images using a pre-trained ResNet model.

## Quick Deploy

Follow these steps to quickly deploy the application locally:

### Installation

#### Prerequisites
- Python 3.8 or higher

#### Install Dependencies
```sh
pip3 install -r requirements.txt
```

#### Clone the Repository
```sh
git clone <https://github.com/milvus-io/bootcamp.git>
cd bootcamp/bootcamp/tutorials/quickstart/app/image_search_with_milvus
```

### Dataset Preparation
We are using a diverse dataset for this demo, which includes approximately 200 categories with images of animals, objects, buildings, and more: <https://github.com/milvus-io/pymilvus-assets/releases/download/imagedata/reverse_image_search.zip>. <br>
We will load and unzip the dataset for further processing and inserting image embeddings by running the `insert.py` file.
```sh
python3 insert.py
```

### Usage
#### Run the Streamlit application
```sh
streamlit run app.py
```
#### Steps:
<div style="text-align: center;">
  <figure>
    <img src="./pics/step1.png" alt="Description of Image" width="700"/>
    <figcaption>Step 1: Choose an image file to upload (JPEG format).</figcaption>
  </figure>
</div>

<div style="text-align: center;">
  <figure>
    <img src="./pics/step2_and_3.jpg" alt="Description of Image" width="700"/>
    <figcaption>Step 2: Crop the image to focus on the region of interest.</figcaption>
    <figcaption>Step 3: Set the desired number of top-k results to display using the slider.</figcaption>
  </figure>
</div>

<div style="text-align: center;">
  <figure>
    <img src="./pics/step4.jpg" alt="Description of Image" width="700"/>
    <figcaption>Step 4: View the search results along with similarity scores.</figcaption>
  </figure>
</div>