Recommendation systems and related technologies have seen rapid development in recent years. A recommendation system is to determine what kind of specific content to provide to the user among the vast amount of information based on the individual needs.

How to recommend based on the existing user profile and content profile involves two key issues: recall and sorting. Recall is the first stage of the recommendation system, which is to quickly retrieve a number of items of potential interest to users from a large library of items, and then hand them over to the sorting process. The amount of data to be processed in this part is usually very large, and the speed required is fast. The sorting involves scoring and sorting all the recalled content and selecting the top scoring results to recommend to the user.

In this projectk, we build a movie recommendation scenario that shows how to implement a recall service for a recommendation system with Milvus. The movie recommendation system is to recall other movies that may be of interest to a given user in a large library of movies based on that user's historical movie rating data.



## Try notebook

In this [notebook](TUTORIAL.ipynb) we will be going over the code required to perform the recall service in a movie recommendation system. 

## How to deploy

Here is the [quick start](QUICK_START.md) for a deployable version of a movie recommendation system.

