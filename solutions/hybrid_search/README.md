# Hybrid Search 

This is an example of Milvus combined with Postgres for hybird search, using feature vectors and structural data to model face attributes. In this example, for a given vector (which can be seen as a given face image), and its attributes (gender, time, whether or not to wear glasses), Milvus is combined with Milvus to query the vector that is most similar to it and its Euclidean distance

## Try notebook
- In this [notebook](Hybird_Search.ipynb) we will be going over the code required to perform hybrid search. This example combines a relational database, Postgres, and then uses it with Milvus for hybird search

## How to deploy
- Here is the [quick start](QUICK_START.md) for a deployable version of hybrid search.


