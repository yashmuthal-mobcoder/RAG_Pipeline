import json
import chromadb
from sentence_transformers import SentenceTransformer

with open("data.json", "r", encoding="utf-8") as f:
    chunks = json.load(f)

print(f"Loaded {len(chunks)} chunks")

model = SentenceTransformer(
    "BAAI/bge-base-en-v1.5"
)

client = chromadb.PersistentClient(
    path="./chroma_db"
)

collection = client.get_or_create_collection(
    name="humanitarian_docs"
)

documents = []
metadatas = []
ids = []

for chunk in chunks:
    documents.append(chunk["content"])

    metadatas.append({
        "title": chunk["title"],
        "section": chunk["section"],
        "subsection": chunk["subsection"] or "",
        "char_count": chunk["char_count"],
        "chunk_index": chunk["chunk_index"],
        "total_chunks_in_record": chunk["total_chunks_in_record"],
    })

    ids.append(chunk["chunk_id"])

print("Generating embeddings...")

embeddings = model.encode(
    documents,
    show_progress_bar=True,
    convert_to_numpy=True
).tolist()

collection.add(
    ids=ids,
    documents=documents,
    metadatas=metadatas,
    embeddings=embeddings
)

print(f"Successfully stored {len(documents)} chunks")