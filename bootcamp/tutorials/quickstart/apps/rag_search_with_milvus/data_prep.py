import urllib.request
import ssl
import certifi

# Set SSL context
ssl_context = ssl.create_default_context(cafile=certifi.where())

# URL for the file to be retrieved
url = "https://raw.githubusercontent.com/milvus-io/milvus/master/DEVELOPMENT.md"
file_path = "./Milvus_DEVELOPMENT.md"


def get_text():
    # Retrieve the URL content
    with urllib.request.urlopen(url, context=ssl_context) as response:
        with open(file_path, "wb") as file:
            file.write(response.read())

    # Read the downloaded file
    with open(file_path, "r") as file:
        file_text = file.read()

    # Split text into lines
    text_lines = file_text.split("# ")
    return text_lines
