import os

import chromadb
import google.generativeai as genai

from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

load_dotenv()

genai.configure(
    api_key=os.getenv("GEMINI_API_KEY")
)

embedding_model = SentenceTransformer(
    "BAAI/bge-base-en-v1.5"
)

client = chromadb.PersistentClient(
    path="./chroma_db"
)

collection = client.get_collection(
    "humanitarian_docs"
)

llm = genai.GenerativeModel(
    "gemini-2.5-flash"
)


def retrieve(
    query: str,
    top_k: int = 5,
):
    query_embedding = embedding_model.encode(
        query,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
    )

    docs = results["documents"][0]
    metas = results["metadatas"][0]

    return list(zip(docs, metas))

def build_context(retrieved_docs):
    context_parts = []

    for i, (doc, meta) in enumerate(
        retrieved_docs,
        start=1,
    ):
        context_parts.append(
            f"""
            [Chunk {i}]
            Section: {meta.get("section")}
            Subsection: {meta.get("subsection")}

            {doc}
            """
        )

    return "\n".join(context_parts)


def generate_answer(
    question: str,
):
    retrieved_docs = retrieve(
        question,
        top_k=5,
    )

    context = build_context(
        retrieved_docs
    )

    prompt = f"""
You are answering questions about a humanitarian policy document.

Rules:
1. Use only the provided context.
2. Do not invent facts.
3. If the answer is not contained in the context, say:
   "I could not find that information in the document."
4. Quote important details when relevant.
5. Give a concise answer.

Context:
{context}

Question:
{question}

Answer:
"""

    response = llm.generate_content(
        prompt
    )

    return response.text



from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class Query(BaseModel):
    question: str

@app.post("/chat")
def chat(query: Query):
    answer = generate_answer(
        query.question
    )

    return {
        "question": query.question,
        "answer": answer,
    }