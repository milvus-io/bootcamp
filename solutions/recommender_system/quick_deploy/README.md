# Quick Start

This project combines Milvus and [PaddleRec](https://aistudio.baidu.com/aistudio/projectdetail/1481839?channel=0&channelType=0&lang=en) to build the recall service of a movie recommender system.

## Data description

[MovisLens](https://grouplens.org/datasets/movielens/) is a dataset on movie ratings, with data from movie rating sites such as IMDB. The dataset contains information about users' ratings of movies, users' demographic characteristics and descriptive features of movies, which is suitable for getting started with recommender systems.

In this project, we use one of the sub-datasets — [MovieLens 1M](https://grouplens.org/datasets/movielens/1m/). This dataset contains 1,000,209 anonymous ratings of approximately 3,900 movies  made by 6,040 MovieLens users. 

The model already trained only uses  `user.dat` to query existed users. In addtition, the model generates vectors of movies in `movie.dat`, and we load these vectors into Milvus. After passing in new user information(gender, age, and occupation), the next step is to extract features and recall in Milvus. Then the system goes through positive sorting(checking movies), and sorting to recommend the most suitable movies.

#### users.dat

UserID::Gender::Age::Occupation::Zip-code

All demographic information is provided voluntarily by the users and is
not checked for accuracy.  Only users who have provided some demographic
information are included in this data set.

- Gender is denoted by a "M" for male and "F" for female
- Age is chosen from the following ranges:

	*  1:  "Under 18"
	* 18:  "18-24"
	* 25:  "25-34"
	* 35:  "35-44"
	* 45:  "45-49"
	* 50:  "50-55"
	* 56:  "56+"
- Occupation is chosen from the following choices:

	*  0:  "other" or not specified
	*  1:  "academic/educator"
	*  2:  "artist"
	*  3:  "clerical/admin"
	*  4:  "college/grad student"
	*  5:  "customer service"
	*  6:  "doctor/health care"
	*  7:  "executive/managerial"
	*  8:  "farmer"
	*  9:  "homemaker"
	* 10:  "K-12 student"
	* 11:  "lawyer"
	* 12:  "programmer"
	* 13:  "retired"
	* 14:  "sales/marketing"
	* 15:  "scientist"
	* 16:  "self-employed"
	* 17:  "technician/engineer"
	* 18:  "tradesman/craftsman"
	* 19:  "unemployed"
	* 20:  "writer"

#### movies.dat

MovieID::Title::Genres

- Titles are identical to titles provided by the IMDB (including
year of release)
- Genres are pipe-separated

- Some MovieIDs do not correspond to a movie due to accidental duplicate
entries and/or test entries
- Movies are mostly entered by hand, so errors and inconsistencies may exist

## Environments

1. Python 3.6/3.7
2. [Milvus 2.0.0](https://milvus.io/docs/v2.0.0/install_standalone-docker.md)

## How to start

1. Start servers: milvus2.0 & redis
 
3. Pull the source code.

   ```shell
   $ git clone https://github.com/milvus-io/bootcamp.git
   $ cd solutions/recommender_system
   ```

3. Install requirements.

   ```shell
   $ pip install -r requirements.txt
   ```

4. Modify config in `milvus_tool/config.py`

   ```
   MILVUS_HOST = 'localhost'
   MILVUS_PORT = 19530

   dim = 32
   pk = FieldSchema(name='pk', dtype=DataType.INT64, is_primary=True)
   field = FieldSchema(name='embedding', dtype=DataType.FLOAT_VECTOR, dim=dim)
   schema = CollectionSchema(fields=[pk, field], description="movie recommender: demo films")

   index_param = {
       "metric_type": "L2",
       "index_type":"IVF_FLAT",
       "params":{"nlist":128}
       }
   
   top_k = 10
   search_params = {
       "metric_type": "L2",
       "params": {"nprobe": 10}
       }

   ```

5. Prepare data (movie_vectors.txt, users.dat, movies.dat) & download models (rank_model, user_vector_model).

   ```shell
   $ cd quick_deploy/movie_recommender
   $ sh get_data.sh
   ```

6. Start recall and sorting service.

   ```shell
   $ sh start_server.sh
   ```
   (May take a few seconds to start the service.)

## How to use

1. Recommend movies.

   ```shell
   $ export PYTHONPATH=$PYTHONPATH:$PWD/proto
   $ python test_client.py as M 32 5 # gender, age, and occupation
   # Expected outputs
   error {
   code: 200
   }
   item_infos {
     movie_id: "760"
     title: "Stalingrad (1993)"
     genre: "War"
   }
   item_infos {
     movie_id: "632"
     title: "Land and Freedom (Tierra y libertad) (1995)"
     genre: "War"
   }
   item_infos {
     movie_id: "1275"
     title: "Highlander (1986)"
     genre: "Action, Adventure"
   }
   ...
   ```

2. Search movie information.

   ```shell
   $ python test_client.py cm 600
   # Expected outputs
   error {
     code: 200
   }
   item_infos {
     movie_id: "600"
     title: "Love and a .45 (1994)"
     genre: "Thriller"
   }
   ```

3. Search user information.

   ```shell
   $ python test_client.py um 10
   # Expected outputs
   error {
     code: 200
   }
   user_info {
     user_id: "10"
     gender: "F"
     age: 35
     job: "1"
     zipcode: "95370"
   }
   ```

   
