# Question Answering System
Question answering is a classic problem in the field of natural language processing. While it sounds like an easy problem to solve, there is still a lot of research going on to improve the techniques that we have now. A large part of solving questions is finding questions that are similar to the one being asked. This is where Milvus comes into play, as Milvus can handle searching through billions of previously seen questions in a matter of milliseconds. Some use cases that can be seen today include: intelligent voice interaction, online customer service, knowledge acquisition, emotion-based chat, etc.

In this solution we will be creating a question answering system that deals with domaine specific questions, in this case insurance.

## Try notebook

In this [notebook](question_answering.ipynb) we will be going over the code required to create a question-answering system. This example uses BERT to extract features of questions that are then used with Milvus to build a system that can perform the searches. 

## How to deploy

Here is the [quick start](QUICK_START.md) for a deployable version of a question-answering system.

