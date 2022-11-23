wget https://paddlerec.bj.bcebos.com/aistudio/aistudio_paddlerec_rank.tar.gz --no-check-certificate
tar xf aistudio_paddlerec_rank.tar.gz
wget https://paddlerec.bj.bcebos.com/aistudio/user_vector.tar.gz --no-check-certificate
mkdir user_vector_model
tar xf user_vector.tar.gz -C user_vector_model/
wget https://paddlerec.bj.bcebos.com/aistudio/movie_vectors.txt --no-check-certificate
wget https://paddlerec.bj.bcebos.com/aistudio/users.dat --no-check-certificate
wget https://paddlerec.bj.bcebos.com/aistudio/movies.dat --no-check-certificate
python3 to_redis.py
python3 to_milvus.py
