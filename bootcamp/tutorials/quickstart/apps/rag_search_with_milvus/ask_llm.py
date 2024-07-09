import os
from openai import AzureOpenAI

# fetch the API key and endpoint
api_key = os.getenv("AZURE_OPENAI_API_KEY")
azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
deployment = os.getenv("AZURE_DEPLOYMENT")


# Initialize the AzureOpenAI client
def get_azure_client():
    client = AzureOpenAI(
        api_key=api_key, api_version="2024-02-01", azure_endpoint=azure_endpoint
    )
    return client


def get_llm_answer(client, context, question):
    # Define system and user prompts
    SYSTEM_PROMPT = """
    Human: You are an AI assistant. You are able to find answers to the questions from the contextual passage snippets provided.
    """
    USER_PROMPT = f"""
    Use the following pieces of information enclosed in <context> tags to provide an answer to the question enclosed in <question> tags.
    <context>
    {context}
    </context>
    <question>
    {question}
    </question>
    """

    response = client.chat.completions.create(
        model=deployment,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_PROMPT},
        ],
    )

    answer = response.choices[0].message.content
    return answer
