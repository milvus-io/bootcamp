from encoder import TextEncoder
from milvus_client import MilvusClient
import numpy as np

def main():
    encoder = TextEncoder()

    client = MilvusClient(
        host="127.0.0.1",
        port="19530",
        collection_name="text_rag",
        dim=384
    )

    texts = [
        "Milvus makes vector search scalable.",
        "LangChain helps build LLM applications.",
        "Vector databases are essential for AI systems."
    ]

    print("\nEncoding texts and inserting into Milvus...\n")
    vecs = encoder.encode(texts)

    # ⭐ 强制所有向量转为 float32，Milvus 只接受 float32
    vecs = np.array(vecs, dtype=np.float32).tolist()

    client.insert(vecs)
    print("\nInserted documents successfully!\n")

    query = "What can scale vector search?"
    print(f"\nSearching for: {query}\n")

    query_vec = encoder.encode(query)[0]
    query_vec = np.array(query_vec, dtype=np.float32).tolist()   # ⭐ 关键：强制为 float32

    # ⭐ 你的 milvus_client 会自动封装成 [query_vec]，这里千万不能再包一层
    res = client.search(query_vec, topk=3)

    print("\nSearch results:")
    for hits in res:          # res = [[hit1, hit2, hit3]]
        for hit in hits:
            print(f"ID={hit.id}, distance={hit.distance}")

    print("\nDone!")


if __name__ == "__main__":
    main()
