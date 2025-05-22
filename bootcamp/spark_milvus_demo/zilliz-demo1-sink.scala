import io.milvus.client.{MilvusClient, MilvusServiceClient}
import io.milvus.grpc.DataType
import io.milvus.grpc.{DataType, FlushResponse}
import io.milvus.param.collection.{CreateCollectionParam, FieldType, FlushParam}
import io.milvus.param.{ConnectParam, R, RpcStatus}
import org.apache.spark.SparkConf
import org.apache.spark.sql.types._
import org.apache.spark.sql.{SaveMode, SparkSession}
import org.slf4j.LoggerFactory
import zilliztech.spark.milvus.MilvusOptions.{MILVUS_COLLECTION_NAME, MILVUS_HOST, MILVUS_PORT, MILVUS_TOKEN, MILVUS_URI}

import java.util

val sparkConf = new SparkConf().setMaster("local")
val spark = SparkSession.builder().config(sparkConf).getOrCreate()
// Fill in user's Zilliz Cloud credentials.
val uri = "https://in01-xxxxxxxxxxxx.aws-us-west-2.vectordb.zillizcloud.com:19535"
val token = "db_admin:xxxx"
// Specify the target Zilliz Cloud vector database collection name.
val collectionName = "databricks_milvus_insert_demo"
// This file simulates a dataframe from user's vector generation job or a Delta table that contains vectors.
val filePath = "/Volumes/zilliz_test/default/sample_vectors/dim32_1k.json"

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
  MILVUS_COLLECTION_NAME -> collectionName,
)

// 4. Insert data to Zilliz Cloud vector db collection
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