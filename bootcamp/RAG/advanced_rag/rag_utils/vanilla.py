from typing import List

from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_milvus import Milvus
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

PROMPT_TEMPLATE = """
Human: You are an AI assistant, and provides answers to questions by using fact based and statistical information when possible.
Use the following pieces of information to provide a concise answer to the question enclosed in <question> tags.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
<context>
{context}
</context>

<question>
{question}
</question>

The response should be specific and use statistics or numbers when possible.

Assistant:"""

rag_prompt = PromptTemplate(
    template=PROMPT_TEMPLATE, input_variables=["context", "question"]
)

embeddings = OpenAIEmbeddings()

llm = ChatOpenAI(model_name="gpt-4", temperature=0)

vectorstore = Milvus(
    embedding_function=embeddings,
    connection_args={
        "uri": "./milvus_demo.db",
    },
    auto_id=True,
    drop_old=True,
)


def format_docs(docs: List[Document]):
    return "\n\n".join(doc.page_content for doc in docs)
