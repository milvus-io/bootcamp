<!-- PROJECT LOGO -->

<p align="center">
  <a href="https://github.com/milvus-io/bootcamp">
    <img src="images/logo.png" alt="Logo">
  </a>
  <p align="center" style="padding-left: 100px; padding-right: 100px">
      Working with all unstructured data, such as reverse image search, audio search, molecular search, video analysis, question and answer systems, NLP, etc.
    <br />
    <br />
    <a href="https://github.com/milvus-io/bootcamp/issues">Report Bug or Request Feature</a>
  </p>
<!-- DEMO -->
<table>
  <tr>
    <td width="30%">
      <a href="https://milvus.io/milvus-demos/">
        <img src="https://zilliz-cms.s3.us-west-2.amazonaws.com/image_search_59a64e4f22.gif" />
      </a>
    </td>
    <td width="30%">
<a href="https://milvus.io/milvus-demos/">
<img src="https://zilliz-cms.s3.us-west-2.amazonaws.com/qa_df5ee7bd83.gif" />
</a>
    </td>
    <td width="30%">
<a href="https://milvus.io/milvus-demos/">
<img src="https://zilliz-cms.s3.us-west-2.amazonaws.com/mole_search_76f8340572.gif" />
</a>
    </td>
  </tr>
  <tr>
    <th align="center">
      <a href="https://milvus.io/milvus-demos/">Reverse Image search</a>
    </th>
    <th align="center">
      <a href="https://milvus.io/milvus-demos/">Chatbots</a>
    </th>
    <th align="center">
      <a href="https://milvus.io/milvus-demos/">Chemical structure search</a>
    </th>
  </tr>
</table>

<!-- TABLE OF CONTENTS -->
<details open="open">
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="https://github.com/milvus-io/bootcamp/tree/master/bootcamp">Bootcamp tutorials</a>
    </li>
    <ul>
        <li><a href="https://github.com/milvus-io/bootcamp/tree/master/bootcamp/Retrieval/">Retrieval</a></li>
        <li><a href="https://github.com/milvus-io/bootcamp/tree/master/bootcamp/RAG/">RAG</a></li>
        <li><a href="https://github.com/milvus-io/bootcamp/tree/master/bootcamp/OpenAIAssistants/">OpenAIAssistants</a></li>
      </ul>
    <li>
      <a href="#pencil-applications">Applications</a>
      <ul>
        <li><a href="#icecream-run-locally">Run locally</a></li>
        <li><a href="#clapper-live-demo">Live demo</a></li>
      </ul>
    </li>
    <li>
      <a href="#mag-benchmark-tests">Benchmark Tests</a>
    </li>
    <li><a href="#pencil-contributing">Contributing</a></li>
    <li><a href="#fire-community">Community</a></li>
    </ol>
</details>

<!-- ABOUT MILVUS Bootcamp -->

## :mega: About Milvus Bootcamp

**Embed everything**, thanks to AI, we can use neural networks to extract feature vectors from unstructured data, such as image, audio and vide etc. Then analyse the unstructured data by calculating the feature vectors, for example calculating the Euclidean or Cosine distance of the vectors to get the similarity.

[Milvus Bootcamp](https://github.com/milvus-io/bootcamp) is designed to expose users to both the simplicity and depth of the [**Milvus**](https://milvus.io/) vector database. Discover how to run **benchmark tests** as well as build similarity search applications like **chatbots**, **recommender systems**, **reverse image search**, **molecular search**, **video search**, **audio search**, and more.

<!--ALL APPLICATIONS-->

## :pencil: Applications

### :icecream: Run locally

Here are several applications for a wide range of scenarios. Each application contains a Jupyter Notebook or a Docker deployable solution, meaning anyone can run it on their local machine. In addition to this there are also some related technical articles and live streams.

Look here for more application [**Examples**](https://github.com/towhee-io/examples).

💡 Please refer to the [Bootcamp FAQ](applications/applications_faq.md) for troubleshooting.<br>
💡 For Chinese links below, try using Google Translate.

<table>
   <tr>
        <td width="40%"><b>Applications</b></td>
     		<td><b>Have fun with it</b></td>
        <td><b>Article</b></td>
        <td><b>Video</b></td>
   </tr>
   <tr>
     <td><b>Reverse Image Search using Images</b>
        <p>Build a reverse image search system using Milvus paired with Towhee for feature extraction.</p>
     </td>
      <td>
        <p>- <a href="https://github.com/towhee-io/examples/tree/main/image/reverse_image_search">Jupyter notebook</a></p>
        <p>- <a href="applications/image/reverse_image_search">Quick deploy</a></p>
     </td>
      <td>
        <p>- <a href="https://zhuanlan.zhihu.com/p/517360724">10 lines of code for reverse image search</a></p>
        <p>- <a href="https://milvus.io/blog/building-a-search-by-image-shopping-experience-with-vova-and-milvus.md">Reverse Image Search Shopping Experience with VOVA and Milvus</a></p>
     </td>
     <td>
        <p>- <a href="https://www.bilibili.com/video/BV1SN411o79n">VOVA video demo (in Chinese)</a></p>
     </td>
   </tr>
   <tr>
      <td>
        <p><b>Reverse Image Search using Text</b></p>
        <p>Using Milvus and Towhee. </p></td>
      <td>- <a href="https://github.com/towhee-io/examples/tree/main/image/text_image_search">Jupyter notebook</a></td>
      <td>
        <p>- <a href="https://zhuanlan.zhihu.com/p/531951677">1. CLIP text-based image search）</a></p>
        <p>- <a href="https://zhuanlan.zhihu.com/p/537123858">2. Implement the prototype in 5 minutes</a></p>
      </td>
      <td></td>
   </tr>
   <tr>
      <td rowspan="2">
        <p><b>RAG</b></p>
        <p>Question answering chatbot using Milvus and Towhee for natural language processing (NLP).</p>
     </td>
      <td>
        <p>-<a href="https://github.com/towhee-io/examples/tree/main/nlp/question_answering"> Jupyter notebook</a></p>
        <p>-<a href="applications/nlp/question_answering_system"> Quick deploy</a></p>
     </td>
      <td>
          <p>-<a href="https://mp.weixin.qq.com/s/BZp4CMv2yuVb0oEyuDKNkw">Quickly build a conversational chatbot</a></p>
        <p>-<a href="https://medium.com/voice-tech-podcast/building-an-intelligent-qa-system-with-nlp-and-milvus-75b496702490">Building an Intelligent QA System with NLP and Milvus</a></p>
     </td>
     <td>
        <p>-<a href="https://www.bilibili.com/video/BV1ki4y1t72o">video demo (in Chinese)</a></p>
     </td>
   </tr>
   <tr>
      <td>
        <p>-<a href="https://github.com/PaddlePaddle/PaddleNLP/tree/develop/pipelines/examples/question-answering">PaddlePaddle (Chinese bot)</a></p>
        <p>-<a href="https://github.com/PaddlePaddle/PaddleNLP/tree/develop/pipelines/examples/FAQ">PaddlePaddle FAQ (in Chinese)</a></p>
     </td>
      <td></td>
      <td></td>
   </tr>
   <tr>
      <td rowspan="2">
        <p><b>Retrieval</b></p>
        <p>Build a text search engine using Milvus and BERT model. </p>
     </td>
      <td>
        <p>- <a href="https://github.com/towhee-io/examples/tree/main/nlp/text_search">Jupyter notebook</a></p>
     </td>
      <td>- <a href="https://mp.weixin.qq.com/s/OUrBSCqnLuh9btyK3SxWgQ">Using Milvus and BERT</a></td>
      <td>- <a href="https://www.bilibili.com/video/BV1Xi4y1E7Tb">video demo (in Chinese)</a></td>
   </tr>
     <tr>
      <td>
        <p>- <a href="https://github.com/PaddlePaddle/PaddleNLP/tree/develop/applications/neural_search">PaddlePaddle Hybrid Search with neural cross-encoder (in Chinese)</a></p>
     </td>
      <td></td>
      <td></td>
   </tr>
   <tr>
      <td>
        <p><b>Recommender System </b></p>
        <p>Build an AI-powered movie recommender system using Milvus paired with PaddlePaddle’s deep learning framework. </p>
     </td>
      <td>
        <p>- <a href="applications/nlp/recommender_system/recommender_system.ipynb">Jupyter notebook</a></p>
      </td>
      <td>- <a href="https://mp.weixin.qq.com/s/nAr45u-ruvhWQ8LcVxbhOg">Milvus and PaddlePaddle (in Chinese)</a></td>
      <td></td>
   </tr>
   <tr>
      <td><p><b>Video Search by Image</b></p>
        <p>Build a video similarity search engine using Milvus and Towhee. </p>
     </td>
      <td>
        <p>- <a href="https://github.com/towhee-io/examples/tree/main/video/reverse_video_search">Jupyter notebook</a></p>
     </td>
      <td>
        <p>- <a href="https://mp.weixin.qq.com/s/DOfiGP5BG_9sD7zZair4ew">Milvus video search by image</a></p>
  		<p>- <a href="https://milvus.io/blog/2021-10-10-milvus-helps-analyze-videos.md">Building a Video Analysis System with Milvus Vector Database</a></p>
	  </td>
	  <td></td>
   </tr>
   <tr>
      <td>
        <p><b>Video Deduplication</b></p>
        <p>Build a video deduplication system to detect copied video sharing duplicate segments.</p>
     </td>
     <td>
        <p>- <a href="https://github.com/towhee-io/examples/tree/main/video/video_copy_detection">Jupyter notebook</a></p>
      </td>
      <td></td>
      <td></td>
   </tr>
   <tr>
      <td>
        <p><b>Video Search by Text</b></p>
        <p>Search for matched or related videos given an input text.  Uses Milvus and Towhee. </p>
     </td>
      <td>- <a href="https://github.com/towhee-io/examples/tree/main/video/text_video_retrieval">Jupyter notebook</a></td>
      <td>- <a href="https://zhuanlan.zhihu.com/p/584641533">Implement video search in 5 minutes no tags required</a></td>
      <td></td>
   </tr>
   <tr>
      <td>
        <p><b>Audio Classification</b></p>
        <p>Build an audio classification engine using Milvus & Towhee to classify audio.</p>
     </td>
      <td>
        <p>- <a href="https://github.com/towhee-io/examples/tree/main/audio/audio_classification">Jupyter notebook</a></p>
     </td>
      <td></td>
      <td></td>
   </tr>
   <tr>
      <td>
        <p><b>Audio Fingerprinting</b> </p>
        <p>Build engines based on audio fingerprints using Milvus & Towhee, such as music detection system.</p>
     </td>
      <td>
        <p>- <a href="https://github.com/towhee-io/examples/tree/main/audio/audio_fingerprint">Jupyter notebook</a></p>
     </td>
      <td></td>
      <td></td>
   </tr>
   <tr>
      <td>
        <p><b>Molecular Similarity Search </b></p>
        <p>Build a molecular similarity search system using Milvus paired with RDKit for cheminformatics. </p>
     </td>
      <td><p>- <a href="https://github.com/towhee-io/examples/tree/main/medical/molecular_search">Jupyter notebook</a></p>
      </td>
      <td>- <a href="https://mp.weixin.qq.com/s/ZIH_zYltT6aJNQYMhOSsAg">Milvus powers AI drug research (in Chinese)</a></td>
      <td>- <a href="https://www.bilibili.com/video/BV1dD4y1D7zS">demo video (in Chinese)</a></td>
   </tr>
</table>

### :clapper: Live Demo

We have built [online demos](https://milvus.io/milvus-demos/) for reverse image search, chatbot and molecular search that everyone can have fun with.

## :mag: Benchmark Tests

The [VectorDBBench](https://github.com/zilliztech/VectorDBBench) is not just an offering of benchmark results for mainstream vector databases and cloud services, it's your go-to tool for the ultimate performance and cost-effectiveness comparison.

## :pencil: Contributing

Contributions to Milvus Bootcamp are welcome from everyone. See [Guidelines for Contributing](./contributing.md) for details. 


## :fire: Community

- 🤖 Join the Milvus community on [Discord](https://discord.gg/8uyFbECzPX) to chat with the Milvus team and community.

- #️⃣ Enterprise Zilliz customers, join us on Slack (ask your Zilliz contact for an invitation) for technical support.

- 😺 For all other open source Milvus technical support, to discuss, and report bugs, join us on [GitHub Discussions](https://github.com/milvus-io/milvus/discussions).

- 🧧 We also have a [Chinese WeChat group](images/wechat_group.png).
