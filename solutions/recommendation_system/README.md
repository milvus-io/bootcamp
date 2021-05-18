#Recommendation System
Recommendation systems and related technologies have seen rapid development in recent years. Recall is the first stage of any recommendation system, which handles the quick retrieval of items of potential interest to a user. We then push these results into a sorting process. The most time spent in this process flow is spent on finding the potential items, as they must be pulled out from large collections, sometimes in the billion item scale. This is where Milvus comes into play.

This solution builds around a movie recommendation system, specifically tackling the recall step of the pipeline. The movie recommendation system recalls other movies that may be of interest to a given user based on that user's historical movie rating data.


## Try notebook
In this [notebook](recommendation_system.ipynb) we will be going over the code required to perform the recall portion in a movie recommendation system. 

## How to deploy
Here is the [quick start](./quick_deploy/QUICK_START.md) for a deployable version of a movie recommendation system.

