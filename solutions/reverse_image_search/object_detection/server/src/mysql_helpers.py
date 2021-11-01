import sys
import pymysql
from config import MYSQL_HOST, MYSQL_PORT, MYSQL_USER, MYSQL_PWD, MYSQL_DB
from logs import LOGGER


class MySQLHelper():
    """
    Say something about the ExampleCalass...

    Args:
        args_0 (`type`):
        ...
    """
    def __init__(self):
        self.conn = pymysql.connect(host=MYSQL_HOST, user=MYSQL_USER, port=MYSQL_PORT, password=MYSQL_PWD,
                                    database=MYSQL_DB,
                                    local_infile=True)
        self.cursor = self.conn.cursor()

    def test_connection(self):
        try:
            self.conn.ping()
        except Exception:
            self.conn = pymysql.connect(host=MYSQL_HOST, user=MYSQL_USER, port=MYSQL_PORT, password=MYSQL_PWD,
                                    database=MYSQL_DB,local_infile=True)
            self.cursor = self.conn.cursor()

    def create_mysql_table(self, table_name):
        # Create mysql table if not exists
        self.test_connection()
        sql = "create table if not exists " + table_name + "(milvus_id TEXT, image_path TEXT);"
        try:
            self.cursor.execute(sql)
            LOGGER.debug(f"MYSQL create table: {table_name} with sql: {sql}")
        except Exception as e:
            LOGGER.error(f"MYSQL ERROR: {e} with sql: {sql}")
            sys.exit(1)

    def load_data_to_mysql(self, table_name, data):
        # Batch insert (Milvus_ids, img_path) to mysql
        self.test_connection()
        sql = "insert into " + table_name + " (milvus_id,image_path) values (%s,%s);"
        try:
            self.cursor.executemany(sql, data)
            self.conn.commit()
            LOGGER.debug(f"MYSQL loads data to table: {table_name} successfully")
        except Exception as e:
            LOGGER.error(f"MYSQL ERROR: {e} with sql: {sql}")
            sys.exit(1)

    def search_by_milvus_ids(self, ids, table_name):
        # Get the img_path according to the milvus ids
        self.test_connection()
        str_ids = str(ids).replace('[', '').replace(']', '')
        sql = "select image_path from " + table_name + " where milvus_id in (" + str_ids + ") order by field (milvus_id," + str_ids + ");"
        try:
            self.cursor.execute(sql)
            results = self.cursor.fetchall()
            results = [res[0] for res in results]
            LOGGER.debug("MYSQL search by milvus id.")
            return results
        except Exception as e:
            LOGGER.error(f"MYSQL ERROR: {e} with sql: {sql}")
            sys.exit(1)

    def delete_table(self, table_name):
        # Delete mysql table if exists
        self.test_connection()
        sql = "drop table if exists " + table_name + ";"
        try:
            self.cursor.execute(sql)
            LOGGER.debug(f"MYSQL delete table:{table_name}")
        except Exception as e:
            LOGGER.error(f"MYSQL ERROR: {e} with sql: {sql}")
            sys.exit(1)

    def delete_all_data(self, table_name):
        # Delete all the data in mysql table
        self.test_connection()
        sql = 'delete from ' + table_name + ';'
        try:
            self.cursor.execute(sql)
            self.conn.commit()
            LOGGER.debug(f"MYSQL delete all data in table:{table_name}")
        except Exception as e:
            LOGGER.error(f"MYSQL ERROR: {e} with sql: {sql}")
            sys.exit(1)

    def count_table(self, table_name):
        # Get the number of mysql table
        self.test_connection()
        sql = "select count(milvus_id) from " + table_name + ";"
        try:
            self.cursor.execute(sql)
            results = self.cursor.fetchall()
            LOGGER.debug(f"MYSQL count table:{table_name}")
            return results[0][0]
        except Exception as e:
            LOGGER.error(f"MYSQL ERROR: {e} with sql: {sql}")
            sys.exit(1)
