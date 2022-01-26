# Milvus3D


Start a Milvus Server. See tutorial [Here](https://milvus.io/docs/v2.0.0/install_standalone-docker.md).

Start MySQL.
```bash
docker run -p 3306:3306 -e MYSQL_ROOT_PASSWORD=123456 -d mysql:5.7
```
Install the dependencies
```bash
pip install -r requirements.txt
```
Download & decompress the ModelNet40 dataset and weights of the deep learning model.
```bash
chmod +x download_data.sh
./download_data.sh
```
Create two directories to store the pre-processed data for load and search respectively.
```bash
mkdir search_features
mkdir load_features
```
Batch pre-process the data. Takes ~1.5 hrs with ModelNet40 (**You can skip this by downloading the pre-processed data directly, 
see next step**). This operation will first compress the 3d models to 1024 faces and then do certain pre-processing steps. load_features directory will be populated.
```bash
chmod +x preprocess.sh
./preprocess.sh true
```
**Only do this if you skipped the last step!!** Download the pre-processed data from Google Drive.
```bash
gdown "https://drive.google.com/uc?id=1XFonx5ubCSTzEQGvGkpX5LXgdAK3yHQX"
tar -xvf load_feature.tar.gz
```
Start the FASTAPI server
```bash
python3 main.py
```
If you see this in your terminal, Milvus3D has been started successfully.

```
$$\      $$\ $$\ $$\                                       $$$$$$\  $$$$$$$\  
$$$\    $$$ |\__|$$ |                                     $$ ___$$\ $$  __$$\ 
$$$$\  $$$$ |$$\ $$ |$$\    $$\ $$\   $$\  $$$$$$$\       \_/   $$ |$$ |  $$ |
$$\$$\$$ $$ |$$ |$$ |\$$\  $$  |$$ |  $$ |$$  _____|        $$$$$ / $$ |  $$ |
$$ \$$$  $$ |$$ |$$ | \$$\$$  / $$ |  $$ |\$$$$$$\          \___$$\ $$ |  $$ |
$$ |\$  /$$ |$$ |$$ |  \$$$  /  $$ |  $$ | \____$$\       $$\   $$ |$$ |  $$ |
$$ | \_/ $$ |$$ |$$ |   \$  /   \$$$$$$  |$$$$$$$  |      \$$$$$$  |$$$$$$$  |
\__|     \__|\__|\__|    \_/     \______/ \_______/        \______/ \_______/ 

Welcome to Milvus 3D! :)

Author: Sida Shen
```
