<!-- PROJECT LOGO -->

<p align="center">
  <a href="https://github.com/milvus-io/bootcamp">
    <img src="images/logo.png" alt="Logo">
  </a>
  <p align="center" style="padding-left: 100px; padding-right: 100px">
      Dealing with all unstructured data such as reverse image search, audio search, molecular search, video analysis, question and answer systems, NLP, etc.
    <br />
    <br />
    <a href="https://github.com/milvus-io/bootcamp/issues">Report Bug or Request Feature</a>
  </p>
<!-- TABLE OF CONTENTS -->
<details open="open">
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#mega-about-milvus-bootcamp">About Milvus Bootcamp</a>
    </li>
    <li>
      <a href="#pencil-solutions">Solutions</a>
      <ul>
        <li><a href="#icecream-run-locally">Run locally</a></li>
        <li><a href="#clapper-live-demo">Live demo</a></li>
      </ul>
    </li>
    <li>
      <a href="#mag-benchmark-tests">Benchmark Tests</a>
      <ul>
        <li><a href="#dart-1-million-benchmark-testing">1 million benchmark testing</a></li>
        <li><a href="#art-100-million-benchmark-testing">100 million benchmark testing</a></li>
      </ul>
    </li>
    <li><a href="#two_women_holding_hands-collaborations">Collaborations</a></li>
      <ul>
        <li><a href="#clap-milvus_and_onnx">Milvus_and_ONNX</a></li>
      </ul>
    <li><a href="#pencil-contributing">Contributing</a></li>
    <li><a href="#fire-supports">Supports</a></li>
    </ol>
</details>
<table>
  <tr>
    <td width="30%">
      <a href="https://zilliz.com/milvus-demos">
        <img src="https://zilliz-cms.s3.us-west-2.amazonaws.com/image_search_59a64e4f22.gif" />
      </a>
    </td>
    <td width="30%">
<a href="https://zilliz.com/milvus-demos">
<img src="https://zilliz-cms.s3.us-west-2.amazonaws.com/qa_df5ee7bd83.gif" />
</a>
    </td>
    <td width="30%">
<a href="https://zilliz.com/milvus-demos">
<img src="https://zilliz-cms.s3.us-west-2.amazonaws.com/mole_search_76f8340572.gif" />
</a>
    </td>
  </tr>
  <tr>
    <th align="center">
      <a href="https://zilliz.com/milvus-demos">Reverse Image search</a>
    </th>
    <th align="center">
      <a href="https://zilliz.com/milvus-demos">Chatbots</a>
    </th>
    <th align="center">
      <a href="https://zilliz.com/milvus-demos">Chemical structure search</a>
    </th>
  </tr>
</table>
<!-- ABOUT MILVUS Bootcamp -->

## :mega: About Milvus Bootcamp

**Embedding in everything**, thanks to AI, we can use neural networks to extract feature vectors from unstructured data, such as image, audio and vide etc. Then analyse the unstructured data by calculating the feature vectors, for example calculating the Euclidean or Cosine distance of the vectors to get the similarity.

[Milvus Bootcamp](https://github.com/milvus-io/bootcamp) is designed to expose users to both the simplicity and depth of the [**Milvus**](https://milvus.io/) vector database. Discover how to run **benchmark tests** as well as build similarity search applications spanning **chatbots**, **recommendation systems**, **reverse image search**, **molecular search**, **video search**, **audio search**, and more.

<!--ALL SOLUTIONS-->

## :pencil: Solutions

### :icecream: Run locally

Here are several solutions for a wide range of scenarios. Each solution contains a Jupyter Notebook and a Docker deployable solution, meaning anyone can run it on their local machine. In addition to this there are also some related technical articles and live streams.

| <br />Solutions<img width=600/> | <br />Have fun with it<img width=300/>                    | <br />Article<img width=200/>                              | <br />Video<img width=200/>                          |
| ----------------------------------------------------------- | ----------------------------------------------------------- | ----------------------------------------------------------- | ----------------------------------------------------- |
| [**Reverse Image Search**](./solutions/reverse_image_search)<br />Build a reverse image search system using Milvus paired with YOLOv3 for object detection and ResNet-50 for feature extraction. | - [Jupyter notebook](solutions/reverse_image_search/reverse_image_search.ipynb)<br />- [Quick deploy](solutions/reverse_image_search/quick_deploy)<br />- [Object detection](solutions/reverse_image_search/object_detection) | - [Chinese](https://mp.weixin.qq.com/s/7lNuaI-eL3lsQlOq0eolkw)<br />- [English](https://blog.milvus.io/milvus-application-1-building-a-reverse-image-search-system-based-on-milvus-and-vgg-aed4788dd1ea) | - [Chinese](https://www.bilibili.com/video/BV1SN411o79n) |
| [**Question Answering System**](./solutions/question_answering_system)<br />Build an intelligent chatbot using Milvus and the BERT model for natural language processing (NLP). | - [Jupyter notebook](solutions/question_answering_system/question_answering.ipynb)<br />- [Quick deploy](solutions/question_answering_system/quick_deploy) | - [Chinese](https://mp.weixin.qq.com/s/BZp4CMv2yuVb0oEyuDKNkw)<br />- [English](https://medium.com/voice-tech-podcast/building-an-intelligent-qa-system-with-nlp-and-milvus-75b496702490) | - [Chinese](https://www.bilibili.com/video/BV1ki4y1t72o) |
| [**Recommendation System**](./solutions/recommendation_system)<br />Build an AI-powered movie recommendation system using Milvus paired with PaddlePaddle’s deep learning framework. | - [Jupyter notebook](solutions/recommendation_system/recommendation_system.ipynb)<br />- [Quick deploy](solutions/recommendation_system/quick_deploy) | - [Chinese](https://mp.weixin.qq.com/s/nAr45u-ruvhWQ8LcVxbhOg) |  |
| [**Molecular Similarity Search**](./solutions/molecular_similarity_search)<br />Build a molecular similarity search system using Milvus paired with RDKit for cheminformatics. | - [Jupyter notebook](solutions/molecular_similarity_search/molecular_search.ipynb)<br />- [Quick deploy](solutions/molecular_similarity_search/quick_deploy) | - [Chinese](https://mp.weixin.qq.com/s/ZIH_zYltT6aJNQYMhOSsAg) | - [Chinese](https://www.bilibili.com/video/BV1dD4y1D7zS) |
| [**Video Similarity Search**](./solutions/video_similarity_search)<br />Build a video similarity search engine using Milvus and a VGG neural network. Also paired with YOLOV2 & ResNet-50 to detect object in video. | - [Jupyter notebook](solutions/video_similarity_search/video_similarity_search.ipynb)<br />- [Quick deploy](solutions/video_similarity_search/quick_deploy)<br />- [Object detection](solutions/video_similarity_search/object_detection) | - [Chinese](https://mp.weixin.qq.com/s/DOfiGP5BG_9sD7zZair4ew)<br />- [English](https://blog.milvus.io/4-steps-to-building-a-video-search-system-5a3ced633308) |                                                        |
| [**Audio Similarity Search**](./solutions/audio_similarity_search)<br />Build an audio search engine using Milvus paired with PANNs for audio pattern recognition. | - [Jupyter notebook](solutions/audio_similarity_search/audio_similarity_search.ipynb)<br />- [Quick deploy](solutions/audio_similarity_search/quick_deploy) | - [Chinese](https://mp.weixin.qq.com/s/PJfO71YOTW2gXO6SL-OOuA) |                                                        |
| [**Text Search Engine**](./solutions/text_search_engine)<br />Build a text search engine using Milvus and BERT model. | - [Jupyter notebook](solutions/text_search_engine/text_search_engine.ipynb)<br />- [Quick deploy](solutions/text_search_engine/quick_deploy) | - [Chinese](https://mp.weixin.qq.com/s/OUrBSCqnLuh9btyK3SxWgQ) | - [Chinese](https://www.bilibili.com/video/BV1Xi4y1E7Tb) |
| [**DNA Sequence Classification**](./solutions/dna_sequence_classification)<br />Build a DNA sequence classification system using Milvus with k-mers & CountVectorizer. | - [Jupyter notebook](solutions/dna_sequence_classification/dna_sequence_classification.ipynb)<br />- [Quick deploy](solutions/text_search_engine/quick_deploy) | - [Chinese](https://my.oschina.net/u/4209276/blog/5191465)  |  |

### :clapper: Live Demo

We have built [online demos](https://zilliz.com/milvus-demos?isZilliz=true) for reverse image search, chatbot and molecular search that everyone can have fun with.

<!-- BENCHMARK TESTS-->

## :mag: Benchmark Tests

The [Benchmark Test](./benchmark_test) contains 1 million and 100 million vector tests that indicate how your system will react to differently sized datasets.

 ### :dart: [1 million benchmark testing](https://github.com/milvus-io/bootcamp/blob/master/benchmark_test/lab1_sift1b_1m.md)

We extracted one million vectors from the [SIFT1B Dataset](http://corpus-texmex.irisa.fr/) for **accuracy tests** and **performance tests**. Through [this test](./benchmark_test/lab1_sift1b_1m.md), you can learn the basic operations of Milvus, including creating collections, inserting data, building indexes, searching, etc.

 ### :art: [100 million benchmark testing](https://github.com/milvus-io/bootcamp/blob/master/benchmark_test/lab2_sift1b_100m.md)

We extracted 100 million vectors from the [SIFT1B Dataset](http://corpus-texmex.irisa.fr/) for **accuracy tests** and **performance tests**. Through [this test](./benchmark_test/lab2_sift1b_100m.md), you can learn the basic operations of Milvus, including creating collections, inserting data, building indexes, searching, etc.

<!--THE COLLABORATIONS-->

## :two_women_holding_hands: Collaborations

### :clap: [Milvus_and_ONNX](etc/onnx_and_milvus)

Build a reverse image search system with Milvus using various AI models in collaboration with the Open Neural Network Exchange (ONNX).


## :pencil: Contributing

Contributions to Milvus Bootcamp are welcome from everyone. See [Guidelines for Contributing](./contributing.md) for details. 


## :fire: Supports

Join the Milvus community on [Slack](https://join.slack.com/t/milvusio/shared_invite/zt-e0u4qu3k-bI2GDNys3ZqX1YCJ9OM~GQ) to give feedback, ask for advice, and direct questions to our engineering team. We also have a [WeChat group](images/wechat_group.png).
