import io.milvus.client.{MilvusClient, MilvusServiceClient}
import io.milvus.grpc.{DataType, ImportResponse}
import io.milvus.param.bulkinsert.{BulkInsertParam, GetBulkInsertStateParam}
import io.milvus.param.collection.{CreateCollectionParam, FieldType}
import io.milvus.param.{ConnectParam, R, RpcStatus}

import zilliztech.spark.milvus.MilvusOptions.{MILVUS_COLLECTION_NAME, MILVUS_HOST, MILVUS_PORT, MILVUS_TOKEN, MILVUS_URI}

import org.apache.hadoop.fs.s3a.S3AFileSystem
import org.apache.hadoop.fs.{FileSystem, Path}
import org.apache.spark.SparkConf
import org.apache.spark.sql.types._
import org.apache.spark.sql.{SaveMode, SparkSession}

import java.io.{BufferedReader, DataOutputStream, InputStreamReader}
import java.net.{HttpURLConnection, URI, URL}
import java.util

val sparkConf = new SparkConf().setMaster("local")
val spark = SparkSession.builder().config(sparkConf).getOrCreate()

// Fill in user's Zilliz Cloud credentials.
val uri = "https://in01-xxxxxxxxxxxx.aws-us-west-2.vectordb.zillizcloud.com:19535"
val clusterId = "in01-xxxxxx"
val token = "db_admin:xxxx"
val apiKey = "xxxxx"
val region = "aws-us-west-2"
// Specify the target Zilliz Cloud vector database collection name.
val collectionName = "databricks_zilliz_import_demo"
// This file simulates a dataframe from user's vector generation job or a Delta table that contains vectors.
val inputFilePath = "/Volumes/zilliz_test/default/sample_vectors/dim32_1k.json"
// User needs to create an external location on databricks with an S3 bucket and specify the directory in the bucket to store vector data.
// The vectors will be output to the s3 bucket in specific format that can be loaded to Zilliz Cloud efficiently.
val outputDir = "s3://your-s3-bucket-name/zilliz_spark_demo/"
// The AWS access key and private key which grants only read access to the above s3 bucket. They will be used by Zilliz Cloud Import Data API to load data from the bucket.
val s3ak = "xxxxx"
val s3sk = "xxxxx"

// 1. Create Zilliz Cloud vector db collection through SDK, and define the schema of the collection.
val connectParam: ConnectParam = ConnectParam.newBuilder
  .withUri(uri)
  .withToken(token)
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

// 2. Read data from file to build vector dataframe. The schema of the dataframe must logically match the schema of vector db.
val df = spark.read
  .schema(new StructType()
    .add(field1Name, IntegerType)
    .add(field2Name, StringType)
    .add(field3Name, ArrayType(FloatType), false))
  .json(inputFilePath)

// 3. Store all vector data in the s3 bucket to prepare for loading. 
df.repartition(1)
  .write
  .format("mjson")
  .mode("overwrite")
  .save(outputDir)

// 4. As the vector data has been stored in the s3 bucket as files, here we list the directory and get the file paths
// to prepare input of Zilliz Cloud Import Data API call.
val hadoopConfig = spark.sparkContext.hadoopConfiguration
val directory = new Path(outputDir)
val fs = FileSystem.get(directory.toUri, hadoopConfig)
val files = fs.listStatus(directory)
val ouputPath = files.filter(file => {
    file.getPath.getName.endsWith(".json")
})(0)
val completeJsonPath = ouputPath.getPath
println(s"completeJsonPath: ${completeJsonPath}")

// 5. Make a call to Zilliz Cloud Import Data API.
val importApiUrl = s"https://controller.api.${region}.zillizcloud.com/v1/vector/collections/import"
val postData =
  s"""
    |{
    |  "clusterId": "${clusterId}",
    |  "collectionName": "${collectionName}",
    |  "objectUrl": "${completeJsonPath}",
    |  "accessKey": "${s3ak}",
    |  "secretKey": "${s3sk}"
    |}
    |""".stripMargin
val url = new URL(importApiUrl)
val connection = url.openConnection().asInstanceOf[HttpURLConnection]

try {
  // Set up the request method and headers
  connection.setRequestMethod("POST")
  connection.setRequestProperty("Authorization", s"Bearer ${apiKey}")
  connection.setRequestProperty("Content-Type", "application/json")
  connection.setRequestProperty("accept", "application/json")
  connection.setDoOutput(true)

  // Write the POST data to the connection
  val out = new DataOutputStream(connection.getOutputStream)
  out.writeBytes(postData)
  out.flush()
  out.close()

  // Read the response
  val inputStream = new BufferedReader(new InputStreamReader(connection.getInputStream))
  var inputLine: String = null
  val response = new StringBuilder

  while ({inputLine = inputStream.readLine(); inputLine != null}) {
    response.append(inputLine)
  }
  inputStream.close()

  // Print the response
  println(s"Bulkinsert Response code: ${connection.getResponseCode}")
  println(s"Bulkinsert Response body: ${response.toString}")
} finally {
  // Disconnect to release resources
  connection.disconnect()
}
