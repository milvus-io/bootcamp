import io.milvus.client.{MilvusClient, MilvusServiceClient}
import io.milvus.grpc.DataType
import io.milvus.param.collection.{CreateCollectionParam, FieldType}
import io.milvus.param.{ConnectParam, R, RpcStatus}
import org.apache.spark.SparkConf
import org.apache.spark.sql.types._
import org.apache.spark.sql.{SaveMode, SparkSession}
import org.slf4j.LoggerFactory
import zilliztech.spark.milvus.MilvusOptions.{MILVUS_COLLECTION_NAME, MILVUS_HOST, MILVUS_PORT, MILVUS_TOKEN, MILVUS_URI}

import java.util

val sparkConf = new SparkConf().setMaster("local")
val spark = SparkSession.builder().config(sparkConf).getOrCreate()
// Fill in user's Milvus instance credentials.
val host = "127.0.0.1"
val port = 19530
val username = "root"
val password = "Milvus"
// Specify the target Milvus collection name.
val collectionName = "hello_spark_milvus3"
// This file simulates a dataframe from user's vector generation job or a Delta table that contains vectors.
val filePath = "/Volumes/zilliz_test/default/sample_vectors/dim32_1k.json"

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

// log.info(s"create collection ${collectionName} resp: ${createR.toString}")

// 2. Read data from file to build vector dataframe. The schema of the dataframe must logically match the schema of vector db.
val df = spark.read
  .schema(new StructType()
    .add(field1Name, IntegerType)
    .add(field2Name, StringType)
    .add(field3Name, ArrayType(FloatType), false))
  .json(filePath)

// 3. Configure output target
val milvusOptions = Map(
  MILVUS_URI -> uri,
  MILVUS_TOKEN -> token,
  MILVUS_HOST -> host,
  MILVUS_PORT -> port.toString,
  MILVUS_COLLECTION_NAME -> collectionName,
)

// 4. Insert data to milvus collection
df.write
  .options(milvusOptions)
  .format("milvus")
  .mode(SaveMode.Append)
  .save()

// flush data (The following implementation will insert the vector data row by row through Milvus SDK Insert API)
val flushParam: FlushParam = FlushParam.newBuilder
  .addCollectionName(collectionName)
  .build
val flushR: R[FlushResponse] = client.flush(flushParam)
println(flushR)