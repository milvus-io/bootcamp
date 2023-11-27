# OpenAI Text Search Example
This solution leverages OpenAI's embedding models, Towhee, and Milvus in order to make a book title search system. The system generates a database of title embeddings by first pumping titles through OpenAI and storing them in Milvus. When searching, the same processing is done on the search text, but instead of being inserted, is searched against the Milvus collection.
## Try notebook
In this [notebook](openai_text_search.ipynb) we will be going over the code required to create the Milvus collection, insert the data, and search the data.
