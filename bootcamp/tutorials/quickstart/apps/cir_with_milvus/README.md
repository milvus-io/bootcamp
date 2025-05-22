# Composed Image Retrieval with Milvus 🖼️

<div style="text-align: center;">
  <figure>
    <img src="./pics/cir_demo.jpg" alt="Description of Image" width="700"/>
  </figure>
</div>

This Composed Image Retrieval demo showcases the integration of Milvus with [MagicLens](https://open-vision-language.github.io/MagicLens/) for advanced image searching based on user instructions. Users can upload an image and provide editing instructions, which are processed by MagicLens's composed retrieval model to search for candidate images. This powerful combination enables a seamless and intuitive image search experience, leveraging Milvus for efficient retrieval and MagicLens for precise image processing and matching.

## Quick Deploy

Follow these steps to quickly deploy the application locally:

### Preparation

> Prerequisites: Python 3.8 or higher

**1. Download Codes**
```bash
$ git clone <https://github.com/milvus-io/bootcamp.git>
$ cd bootcamp/bootcamp/tutorials/quickstart/app/cir_with_milvus
```

**2. Set Environment**

- Install dependencies

```bash
  $ pip install -r requirements.txt
```

**3. Prepare MagicLens Model** <br>

More detailed information can be found at <https://github.com/google-deepmind/magiclens>

- Setup

  ```bash
  conda create --name magic_lens python=3.9
  conda activate magic_lens
  git clone https://github.com/google-research/scenic.git
  cd scenic
  pip install .
  pip install -r scenic/projects/baselines/clip/requirements.txt
  # you may need to install corresponding GPU version of jax following https://jax.readthedocs.io/en/latest/installation.html
  # e.g.,
  # # CUDA 12 installation
  # Note: wheels only available on linux.
  # pip install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

  # # CUDA 11 installation
  # Note: wheels only available on linux.
  # pip install --upgrade "jax[cuda11_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
  ```

- Model Download
  
  ```bash
  cd .. # in main folder of demo.
  # you may need to use `gcloud auth login` for access, any gmail account should work.
  gsutil cp -R gs://gresearch/magiclens/models ./
  ```

**4. Prepare Data**

We are using a subset of https://github.com/hyp1231/AmazonReviews2023 which includes approximately 5000 images in 33 different categories, such as applicances, beauty and personal care, clothing, sports and outdoors, etc.<br>

Download image set by running [download_images.py](./download_images.py).
```bash
$ python download_images.py
```

Create a collection and load image data from the dataset to get the knowledge ready by running [index.py](./index.py).

```bash
$ python index.py
```

### Start Service

Run the Streamlit application:

```bash
$ streamlit run ui.py
```

There have some options you can set in `cfg.py`.


### Example Usage:

**Step 1:** Choose an image file to upload (JPEG format), and give user instruction as a text input.

<div style="text-align: center;">
  <figure>
    <img src="./pics/edit.jpg" alt="Description of Image" width="700"/>
  </figure>
</div>

**Step 2:** Click on the 'Search' button to see top 100 candidate images generated based on both query image and user instruction.

<div style="text-align: center;">
  <figure>
    <img src="./pics/cir_demo.jpg" alt="Description of Image" width="700"/>
  </figure>
</div>

