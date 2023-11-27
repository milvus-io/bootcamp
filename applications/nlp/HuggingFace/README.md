# HuggingFace Text Search Example
This solution leverages HuggingFace transformer models and Milvus in order to make a question answering system. The system generates a database of question answer pairs first pumping questions through BERT and storing them in Milvus. When searching, the same processing is done on the search question, but instead of being inserted, is searched against the Milvus collection.
## Try notebook
In this [notebook](milvus.ipynb) we will be going over the code required to create the Milvus collection, insert the data, and search the data using only Hugging Face and Milvus.

In this [notebook](towhee.ipynb) we will be going over the code required to create the Milvus collection, insert the data, and search the data using a simplified pipeline provided by Towhee.

