# DNA Sequence Classification
DNA sequence is a text sequence ordering of the bases (A, T, G, C). A classification model build by a bunch of DNA sequences and corresponding classes can determine the class of unknown sequences.

This example uses Milvus and Mysql to build a DNA sequence classification model, which uses CountVectorizer to extract features and vectorize DNA sequences. Vectors are stored in Milvus and corresponding classes are stored in Mysql. Searching similar sequences in Milvus and recall classes from Mysql can classify input DNA sequences.

## Try notebook
- In this [notebook](dna_sequence_classification.ipynb) we will be going over the code required to build and test the DNA sequence classification model.

## How to deploy
- Here is the [quick start](./quick_deploy/README.md) for a deployable version of DNA sequence classification model build with sample dataset.
