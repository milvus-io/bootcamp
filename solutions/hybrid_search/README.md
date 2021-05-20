# Hybrid Search 
Hybrid search  are something that most systems require. In the current version of Milvus, this is done by leveraging a second metadata service in order to perform the second half of the searching. 

In this solution we will be demonstrating how to perform hybrid seraches using Postgres and Milvus. The example shows how to perform searches based on multiple fields, such as sex, age, etc., but uses randomly generated data as a placeholder for actual vectors. 


## Try notebook
- In this [notebook](hybrid_search.ipynb) we will be going over the code required to perform the hybrid search. This example combines a relational database, Postgres, and then uses it with Milvus for hybird search

## How to deploy
- Here is the [quick start](./quick_deploy/QUICK_START.md) for a deployable version of hybrid search.


