
from src.milvus import milvus_client
from src.insert import do_insert
from src.mysql_toolkits import connect_mysql
from src.config import data_path



index_client = milvus_client()
conn = connect_mysql()
cursor = conn.cursor()
status = do_insert(data_path,index_client, conn, cursor)
cursor.close()
conn.close()
print(status)