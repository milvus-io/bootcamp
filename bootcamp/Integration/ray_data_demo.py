# script.py
import ray, pprint
from ray.data import read_parquet

@ray.remote
def hello_world():
    return "hello world"

# Define a regular python function.
def read_data(file_path):
    # Ray data with parquet file works.
    ds = ray.data.read_parquet(file_path)
    print(ds.schema())
    pprint.pprint(ds.take(2))
    return ds

# Convert regular python functions to ray remote functions
read_data_remote = ray.remote(read_data).options(num_returns=1)

# Automatically connect to the running Ray cluster.
ray.init()
FILE_PATH = "local://./data/kaggle_imdb.parquet"

# Call the remote function.
# print(ray.get(hello_world.remote()))
ray.get(read_data_remote.remote(FILE_PATH))