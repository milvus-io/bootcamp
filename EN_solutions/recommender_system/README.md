# Personalized Recommender System Based on Milvus

## Prerequisites

### Environment requirements

The following table lists recommended configurations, which have been tested:

| Component | Recommended Configuration                |
| --------- | ---------------------------------------- |
| CPU       | Intel(R) Core(TM) i7-7700K CPU @ 4.20GHz |
| GPU       | GeForce GTX 1050 Ti 4GB                  |
| Memory    | 32GB                                     |
| OS        | Ubuntu 18.04                             |
| Software  | [Milvus 0.5.3](https://milvus.io/docs/en/userguide/install_milvus/) <br /> [pymilvus 0.2.5](https://pypi.org/project/pymilvus/)  <br /> [PaddlePaddle 1.6.1](https://www.paddlepaddle.org.cn/documentation/docs/en/beginners_guide/install/index_en.html)   |

> For versions before Milvus 0.6.0, GPU is required. For CPU-only version, please use Milvus 0.6.0.

### Data source

The data source is [MovieLens million-scale dataset (ml-1m)](http://files.grouplens.org/datasets/movielens/ml-1m.zip), created by GroupLens Research. Refer to [ml-1m-README](http://files.grouplens.org/datasets/movielens/ml-1m-README.txt) for more information.

## Build a personalized recommender system based on Milvus

Follow the steps below to build a recommender system:

1. Train the model.

   ```bash
   # run train.py
   $ python3 train.py
   ```

   This command generates a model file `recommender_system.inference.model` in the same folder.

2. Generate test data.

   ```bash
   # Download movie data movies_origin.txt to the same folder
   $ wget https://raw.githubusercontent.com/milvus-io/bootcamp/0.5.3/demo/recommender_system/movies_origin.txt
   # Generate test data. The -f parameter is followed by the movie data filename.
   $ python3 get_movies_data.py -f movies_origin.txt
   ```

   The above commands generate `movies_data.txt` in the same folder.

3. Use Milvus for personalized recommendation by running the following command:

   ```bash
   # Milvus performs personalized recommendation based on user status
   $ python3 infer_milvus.py -a <age> -g <gender> -j <job> [-i]
   # Example 1
   $ python3 infer_milvus.py -a 0 -g 1 -j 10 -i
   # Example 2
   $ python3 infer_milvus.py -a 6 -g 0 -j 16
   ```
   
   The following table describes arguments of `infer_milvus.py`.

   | Parameter        | Description                                                         |
   | ----------- | ------------------------------------------------------------ |
   | `-a`/`--age`    | Age distribution <br />0: "Under 18" <br />1: "18-24" <br />2: "25-34" <br />3: "35-44" <br />4: "45-49" <br />5: "50-55" <br />6: "56+" |
   | `-g`/`--gender` | Gender <br />0:male <br />1:female                                         |
   | `-j`/`--job`    | Job <br />0: "other" or not specified <br />1: "academic/educator" <br />2: "artist" <br />3: "clerical/admin" <br />4: "college/grad student" <br />5: "customer service" <br />6: "doctor/health care" <br />7: "executive/managerial" <br />8: "farmer" <br />9: "homemaker" <br />10: "K-12 student" <br />11: "lawyer" <br />12: "programmer" <br />13: "retired" <br />14: "sales/marketing" <br />15: "scientist" <br />16: "self-employed" <br />17: "technician/engineer" <br />18: "tradesman/craftsman" <br />19: "unemployed" <br />20: "writer" |
   | `-i`/`--infer`  | (Optional) Converts test data to vectors and import to Milvus. |

   > Note: `-i`/`--infer` is required when you use Milvus for personalized recommendation for the first time or when you start another training and regenerate the model.

    The result displays top 5 movies that the specified user might be interested in:

   ```bash
   get infer vectors finished!
   Server connected.
   Status(code=0, message='Create table successfully!')
   rows in table recommender_demo: 3883
   Top      Ids     Title   Score
   0        3030    Yojimbo         2.9444923996925354
   1        3871    Shane           2.8583481907844543
   2        3467    Hud     2.849525213241577
   3        1809    Hana-bi         2.826111316680908
   4        3184    Montana         2.8119677305221558
   ```

   > Run `python3 infer_paddle.py`. You can see that Paddle and Milvus generate the same result.
