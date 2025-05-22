

# **MCP + Milvus: Connecting AI with Vector Databases**

## **Introduction**
The **Model Context Protocol (MCP)** is an open protocol that enables AI applications, such as Claude and Cursor, to interact with external data sources and tools seamlessly. Whether you're building custom AI applications, integrating AI workflows, or enhancing chat interfaces, MCP provides a standardized way to connect large language models (LLMs) with relevant contextual data.


This tutorial walks you through **setting up an MCP server for Milvus**, allowing AI applications to perform vector searches, manage collections, and retrieve data using **natural language commands**—without writing custom database queries.


## **Prerequisites**
Before setting up the MCP server, ensure you have:

- Python 3.10 or higher
- A running [Milvus](https://milvus.io/) instance
- [uv](https://github.com/astral-sh/uv) (recommended for running the server)


## **Getting Started**

The recommended way to use this MCP server is to run it directly with uv without installation. This is how both Claude Desktop and Cursor are configured to use it in the examples below.

If you want to clone the repository:

```bash
git clone https://github.com/zilliztech/mcp-server-milvus.git
cd mcp-server-milvus
```

Then you can run the server directly:

```bash
uv run src/mcp_server_milvus/server.py --milvus-uri http://localhost:19530
```


## **Supported Applications**
This MCP server can be used with various AI applications that support the Model Context Protocol, such as:

- **Claude Desktop**: Anthropic's desktop application for Claude
- **Cursor**: AI-powered code editor with MCP support in its Composer feature
- **Other custom MCP clients** Any application implementing the MCP client specification



## **Using MCP with Claude Desktop**

1. Install [Claude Desktop](https://claude.ai/download).
2. Open the Claude configuration file:
   - On macOS: `~/Library/Application Support/Claude/claude_desktop_config.json`
3. Add the following configuration:
```json
{
  "mcpServers": {
    "milvus": {
      "command": "/PATH/TO/uv",
      "args": [
        "--directory",
        "/path/to/mcp-server-milvus/src/mcp_server_milvus",
        "run",
        "server.py",
        "--milvus-uri",
        "http://localhost:19530"
      ]
    }
  }
}
```
4. Restart Claude Desktop to apply the changes.



## **Using MCP with Cursor**
[Cursor](https://docs.cursor.com/context/model-context-protocol) also supports MCP tools through its Agent feature in Composer. You can add the Milvus MCP server to Cursor in two ways:

### **Option 1: Using Cursor Settings UI**
1. Open `Cursor Settings` → `Features` → `MCP`.
2. Click `+ Add New MCP Server`.
3. Fill in:
   - Type: `stdio`
   - Name: `milvus`
   - Command:
     ```bash
     /PATH/TO/uv --directory /path/to/mcp-server-milvus/src/mcp_server_milvus run server.py --milvus-uri http://127.0.0.1:19530
     ```
   - ⚠️ Tip: Use `127.0.0.1` instead of `localhost` to avoid potential DNS resolution issues.

### **Option 2: Using Project-specific Configuration (Recommended)**
1. Create a `.cursor/mcp.json` file in your **project root directory**:
```json
{
  "mcpServers": {
    "milvus": {
      "command": "/PATH/TO/uv",
      "args": [
        "--directory",
        "/path/to/mcp-server-milvus/src/mcp_server_milvus",
        "run",
        "server.py",
        "--milvus-uri",
        "http://127.0.0.1:19530"
      ]
    }
  }
}
```
2. Restart Cursor to apply the configuration.


After adding the server, you may need to press the refresh button in the MCP settings to populate the tool list. The Composer Agent will automatically use the Milvus tools when relevant to your queries.




## **Verifying the Integration**
To ensure the MCP server is correctly set up:

### **For Cursor**
1. Go to `Cursor Settings` → `Features` → `MCP`.
2. Confirm that `"Milvus"` appears in the list of MCP servers.
3. Check if Milvus tools (e.g., `milvus_list_collections`, `milvus_vector_search`) are listed.
4. If errors appear, see the **Troubleshooting** section below.


## **MCP Server Tools for Milvus**
This MCP server provides multiple tools for **searching, querying, and managing vector data in Milvus**. For more details, please refer to the [mcp-server-milvus](https://github.com/zilliztech/mcp-server-milvus) documentation.

### **🔍 Search and Query Tools**
- **`milvus-text-search`** → Search for documents using full text search.
- **`milvus-vector-search`** → Perform vector similarity search on a collection.
- **`milvus-hybrid-search`** → Perform hybrid search combining vector similarity and attribute filtering.
- **`milvus-multi-vector-search`** → Perform vector similarity search with multiple query vectors.
- **`milvus-query`** → Query collection using filter expressions.
- **`milvus-count`** → Count entities in a collection.


### **📁 Collection Management**
- **`milvus-list-collections`** → List all collections in the database.
- **`milvus-collection-info`** → Get detailed information about a collection.
- **`milvus-get-collection-stats`** → Get statistics about a collection.
- **`milvus-create-collection`** → Create a new collection with specified schema.
- **`milvus-load-collection`** → Load a collection into memory for search and query.
- **`milvus-release-collection`** → Release a collection from memory.
- **`milvus-get-query-segment-info`** → Get information about query segments.
- **`milvus-get-collection-loading-progress`** → Get the loading progress of a collection.

### **📊 Data Operations**
- **`milvus-insert-data`** → Insert data into a collection.
- **`milvus-bulk-insert`** → Insert data in batches for better performance.
- **`milvus-upsert-data`** → Upsert data into a collection (insert or update if exists).
- **`milvus-delete-entities`** → Delete entities from a collection based on filter expression.
- **`milvus-create-dynamic-field`** → Add a dynamic field to an existing collection.

### **⚙️ Index Management**
- **`milvus-create-index`** → Create an index on a vector field.
- **`milvus-get-index-info`** → Get information about indexes in a collection.

## **Environment Variables**
- **`MILVUS_URI`** → Milvus server URI (can be set instead of `--milvus-uri`).
- **`MILVUS_TOKEN`** → Optional authentication token.
- **`MILVUS_DB`** → Database name (defaults to "default").

## **Development**
To run the server directly:
```bash
uv run server.py --milvus-uri http://localhost:19530
```

## **Examples**

### **Using Claude Desktop**

#### **Example 1: Listing Collections**

```
What are the collections I have in my Milvus DB?
```
Claude will then use MCP to check this information on our Milvus DB. 
```
I'll check what collections are available in your Milvus database.

> View result from milvus-list-collections from milvus (local)

Here are the collections in your Milvus database:

1. rag_demo
2. test
3. chat_messages
4. text_collection
5. image_collection
6. customized_setup
7. streaming_rag_demo
```

#### **Example 2: Searching for Documents**

```
Find documents in my text_collection that mention "machine learning"
```

Claude will use the full-text search capabilities of Milvus to find relevant documents:

```
I'll search for documents about machine learning in your text_collection.

> View result from milvus-text-search from milvus (local)

Here are the documents I found that mention machine learning:
[Results will appear here based on your actual data]
```

### **Using Cursor**

#### **Example: Creating a Collection**

In Cursor's Composer, you can ask:

```
Create a new collection called 'articles' in Milvus with fields for title (string), content (string), and a vector field (128 dimensions)
```

Cursor will use the MCP server to execute this operation:

```
I'll create a new collection called 'articles' with the specified fields.

> View result from milvus-create-collection from milvus (local)

Collection 'articles' has been created successfully with the following schema:
- title: string
- content: string
- vector: float vector[128]
```

## **Troubleshooting**

### **Common Issues**

#### **Connection Errors**

If you see errors like "Failed to connect to Milvus server":

1. Verify your Milvus instance is running: `docker ps` (if using Docker)
2. Check the URI is correct in your configuration
3. Ensure there are no firewall rules blocking the connection
4. Try using `127.0.0.1` instead of `localhost` in the URI

#### **Authentication Issues**

If you see authentication errors:

1. Verify your `MILVUS_TOKEN` is correct
2. Check if your Milvus instance requires authentication
3. Ensure you have the correct permissions for the operations you're trying to perform

#### **Tool Not Found**

If the MCP tools don't appear in Claude Desktop or Cursor:

1. Restart the application
2. Check the server logs for any errors
3. Verify the MCP server is running correctly
4. Press the refresh button in the MCP settings (for Cursor)

### **Getting Help**

If you continue to experience issues:

1. Check the [GitHub Issues](https://github.com/zilliztech/mcp-server-milvus/issues) for similar problems
2. Join the [Zilliz Community Discord](https://discord.gg/zilliz) for support
3. File a new issue with detailed information about your problem

## **Conclusion**
By following this tutorial, you now have an **MCP server** running, enabling AI-powered vector search in Milvus. Whether you're using **Claude Desktop** or **Cursor**, you can now query, manage, and search your Milvus database using **natural language commands**—without writing database code!