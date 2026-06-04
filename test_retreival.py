import chromadb
from sentence_transformers import SentenceTransformer, CrossEncoder

embedding_model = SentenceTransformer(
    "BAAI/bge-base-en-v1.5"
)

reranker = CrossEncoder(
    "cross-encoder/ms-marco-MiniLM-L-6-v2"
)

client = chromadb.PersistentClient(
    path="./chroma_db"
)

collection = client.get_collection(
    "humanitarian_docs"
)

query = "What are the risks of sharing sensitive data with donors?"

query_for_bge = (
    "Represent this sentence for searching relevant passages: "
    + query
)

query_embedding = embedding_model.encode(
    query_for_bge,
    convert_to_numpy=True
).tolist()

results = collection.query(
    query_embeddings=[query_embedding],
    n_results=20
)

documents = results["documents"][0]
metadatas = results["metadatas"][0]

pairs = [
    [query, doc]
    for doc in documents
]

scores = reranker.predict(pairs)

reranked = sorted(
    zip(scores, documents, metadatas),
    key=lambda x: x[0],
    reverse=True
)

TOP_K = 3

for rank, (score, doc, meta) in enumerate(
    reranked[:TOP_K],
    start=1
):
    print("\n" + "=" * 80)
    print(f"Rank {rank}")
    print("=" * 80)

    print("Relevance Score:", round(float(score), 4))
    print("Section:", meta.get("section"))
    print("Subsection:", meta.get("subsection"))

    print("\nContent:")
    print(doc)