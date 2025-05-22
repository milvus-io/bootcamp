import io.milvus.client.{MilvusClient, MilvusServiceClient}
import io.milvus.grpc.{DataType, ImportResponse}
import io.milvus.param.bulkinsert.{BulkInsertParam, GetBulkInsertStateParam}
import io.milvus.param.collection.{CreateCollectionParam, FieldType}
import io.milvus.param.{ConnectParam, R, RpcStatus}
import org.apache.spark.SparkConf
import org.apache.spark.sql.types._
import org.apache.spark.sql.{SaveMode, SparkSession}
import org.slf4j.LoggerFactory
import zilliztech.spark.milvus.MilvusOptions.{MILVUS_COLLECTION_NAME, MILVUS_HOST, MILVUS_PORT, MILVUS_TOKEN, MILVUS_URI}
import org.apache.hadoop.fs.{FileSystem, Path}
import java.net.URI
import org.apache.log4j.Logger

import scala.collection.JavaConverters._

import java.util

var logger = Logger.getLogger(this.getClass())

val sparkConf = new SparkConf().setMaster("local")
val spark = SparkSession.builder().config(sparkConf).getOrCreate()
// Fill in user's Milvus instance credentials.
val host = "127.0.0.1"
val port = 19530
val username = "root"
val password = "Milvus"
// Specify the target Milvus collection name.
val collectionName = "spark_milvus_test"
// This file simulates a dataframe from user's vector generation job or a Delta table that contains vectors.
val filePath = "/Volumes/zilliz_test/default/sample_vectors/dim32_1k.json"
// The S3 bucket is an internal bucket of the Milvus instance, which the user has full control of.
// The user needs to set up this bucket as "storage crenditial" by following
// the instruction at https://docs.databricks.com/en/connect/unity-catalog/storage-credentials.html#step-2-give-databricks-the-iam-role-details
// Here the user can specify the directory in the bucket to store vector data.
// The vectors will be output to the s3 bucket in specific format that can be loaded to Zilliz Cloud efficiently.
val outputPath = "s3://your-s3-bucket-name/filesaa/spark_output"

// 1. Create Milvus collection through Milvus SDK
val connectParam: ConnectParam = ConnectParam.newBuilder
  .withHost(host)
  .withPort(port)
  .withAuthorization(username, password)
  .build

val client: MilvusClient = new MilvusServiceClient(connectParam)

val field1Name: String = "id_field"
val field2Name: String = "str_field"
val field3Name: String = "float_vector_field"
val fieldsSchema: util.List[FieldType] = new util.ArrayList[FieldType]

fieldsSchema.add(FieldType.newBuilder
  .withPrimaryKey(true)
  .withAutoID(false)
  .withDataType(DataType.Int64)
  .withName(field1Name)
  .build
)
fieldsSchema.add(FieldType.newBuilder
  .withDataType(DataType.VarChar)
  .withName(field2Name)
  .withMaxLength(65535)
  .build
)
fieldsSchema.add(FieldType.newBuilder
  .withDataType(DataType.FloatVector)
  .withName(field3Name)
  .withDimension(32)
  .build
)

// create collection
val createParam: CreateCollectionParam = CreateCollectionParam.newBuilder
  .withCollectionName(collectionName)
  .withFieldTypes(fieldsSchema)
  .build

val createR: R[RpcStatus] = client.createCollection(createParam)

logger.info(s"create collection ${collectionName} resp: ${createR.toString}")

// 2. Read data from file to build vector dataframe. The schema of the dataframe must logically match the schema of vector db.
val df = spark.read
  .schema(new StructType()
    .add(field1Name, IntegerType)
    .add(field2Name, StringType)
    .add(field3Name, ArrayType(FloatType), false))
  .json(filePath)

// 3. Store all vector data in the s3 bucket to prepare for loading. 
df.repartition(1)
  .write
  .format("mjson")
  .mode("overwrite")
  .save(outputPath)

// 4. As the vector data has been stored in the s3 bucket as files, here we list the directory and get the file paths
// to prepare input of Zilliz Cloud Import Data API call.
val hadoopConfig = spark.sparkContext.hadoopConfiguration
val directory = new Path(outputPath)
val fs = FileSystem.get(directory.toUri, hadoopConfig)
val files = fs.listStatus(directory)
val ouputPath = files.filter(file => {
    file.getPath.getName.endsWith(".json")
})(0)
def extractPathWithoutBucket(s3Path: String): String = {
  val uri = new URI(s3Path)
  val pathWithoutBucket = uri.getPath.drop(1)  // Drop the leading '/'
  pathWithoutBucket
}
val ouputFilePathWithoutBucket = extractPathWithoutBucket(ouputPath.getPath.toString)

// 5. Make a call to Milvus bulkinsert API.
val bulkInsertFiles:List[String] = List(ouputFilePathWithoutBucket)
val bulkInsertParam: BulkInsertParam = BulkInsertParam.newBuilder
    .withCollectionName(collectionName)
    .withFiles(bulkInsertFiles.asJava)
    .build

val bulkInsertR: R[ImportResponse] = client.bulkInsert(bulkInsertParam)
logger.info(s"bulkinsert ${collectionName} resp: ${bulkInsertR.toString}")
val taskId: Long = bulkInsertR.getData.getTasksList.get(0)

var bulkloadState = client.getBulkInsertState(GetBulkInsertStateParam.newBuilder.withTask(taskId).build)
while (bulkloadState.getData.getState.getNumber != 1 &&
    bulkloadState.getData.getState.getNumber != 6 &&
    bulkloadState.getData.getState.getNumber != 7 ) {
    bulkloadState = client.getBulkInsertState(GetBulkInsertStateParam.newBuilder.withTask(taskId).build)
    logger.info(s"bulkinsert ${collectionName} resp: ${bulkInsertR.toString} state: ${bulkloadState}")
    Thread.sleep(3000)
}
if (bulkloadState.getData.getState.getNumber != 6) {
    logger.error(s"bulkinsert failed ${collectionName} state: ${bulkloadState}")
}