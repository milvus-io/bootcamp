# Multimodal RAG with Milvus 🖼️

<div style="text-align: center;">
  <figure>
    <img src="./pics/cir_demo.jpg" alt="Description of Image" width="700"/>
  </figure>
</div>

This multi-modal RAG (Retrieval-Augmented Generation) demo showcases the integration of Milvus with [MagicLens](https://open-vision-language.github.io/MagicLens/) and [GPT-4o](https://openai.com/index/hello-gpt-4o/) for advanced image searching based on user instructions. Users can upload an image and edit instructions, which are processed by MagicLens's composed retrieval model to search for candidate images. GPT-4o then acts as a reranker, selecting the most suitable image and providing the rationale behind the choice. This powerful combination enables a seamless and intuitive image search experience.

## Quick Deploy

Follow these steps to quickly deploy the application locally:

### Preparation

> Prerequisites: Python 3.8 or higher

**1. Download Codes**
```bash
$ git clone <https://github.com/milvus-io/bootcamp.git>
$ cd bootcamp/bootcamp/tutorials/quickstart/app/multimodal_rag_with_milvus
```

**2. Set Environment**

- Install dependencies

```bash
  $ pip install -r requirements.txt
```

- Set environment variables

  Modify the environment file [.env](./.env) to change environment variables for either OpenAI or Azure OpenAI service, and only keep the variables relevant to the service chosen:

  ```bash
  # Fill out and keep the following if using OpenAI service
  API_KEY=**************

  # Fill out and keep the following if using Azure OpenAI service
  AZURE_OPENAI_API_KEY=**************
  AZURE_OPENAI_ENDPOINT=https://*******.com    
  AZURE_DEPLOYMENT=******-***-**
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
    <img src="./pics/step1.jpg" alt="Description of Image" width="700"/>
  </figure>
</div>

**Step 2:** Click on the 'Search' button to see top 100 candidate images generated based on both query image and user instruction.

<div style="text-align: center;">
  <figure>
    <img src="./pics/step2.jpg" alt="Description of Image" width="700"/>
  </figure>
</div>

**Step 3:** Click on the 'Ask GPT' button to get the best item chosen by GPT-4o after reranking along with detailed explanation.

<div style="text-align: center;">
  <figure>
    <img src="./pics/step3.jpg" alt="Description of Image" width="700"/>
  </figure>
</div>
