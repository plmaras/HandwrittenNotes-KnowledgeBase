
import argparse
import os
from pathlib import Path
from dotenv import load_dotenv
import chromadb
from groq import Groq
from sentence_transformers import SentenceTransformer

load_dotenv()
 
EMBED_MODEL_NAME = "BAAI/bge-small-en-v1.5"
GEN_MODEL_NAME = "llama-3.3-70b-versatile"  # text-only model, fast/cheap for generation
 
SYSTEM_PROMPT = """You answer questions using ONLY the handwritten notes provided below.
Rules:
- Base your answer strictly on the note content given. Do not use outside knowledge.
- If the notes don't contain enough information to answer, say so plainly -- don't guess.
- After your answer, on a new line, list which note filenames you used, like:
  Sources: 2026-07-08-budget-meeting.md, 2026-07-05-idea-draft.md
- Some note content may contain [maybe: "word"] markers indicating uncertain handwriting
  transcription -- treat those words as slightly less reliable and say so if relevant.
"""
 
 
def retrieve(embed_model, collection, question, k):
    query_embedding = embed_model.encode([question], normalize_embeddings=True).tolist()
    results = collection.query(query_embeddings=query_embedding, n_results=k)
 
    retrieved = []
    for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
        retrieved.append({"filename": meta["filename"], "text": doc, "date": meta.get("date", "")})
    return retrieved
 
 
def build_context_block(retrieved: list) -> str:
    blocks = []
    for r in retrieved:
        blocks.append(f"--- Note: {r['filename']} (date: {r['date']}) ---\n{r['text']}")
    return "\n\n".join(blocks)
 
 
def answer_question(question: str, db_dir: Path, notes_dir: Path, k: int = 3) -> dict:
    embed_model = SentenceTransformer(EMBED_MODEL_NAME)
    client = chromadb.PersistentClient(path=str(db_dir))
    collection = client.get_or_create_collection("handwritten_notes")
 
    retrieved = retrieve(embed_model, collection, question, k)
    context_block = build_context_block(retrieved)
 
    groq_client = Groq(api_key=os.getenv("GroqAPI"))
    completion = groq_client.chat.completions.create(
        model=GEN_MODEL_NAME,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"NOTES:\n{context_block}\n\nQUESTION: {question}"},
        ],
        temperature=0.3,
        max_completion_tokens=800,
    )
 
    answer_text = completion.choices[0].message.content
 
    return {
        "answer": answer_text,
        "retrieved_notes": retrieved,
    }
 
 
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default="chroma_db")
    parser.add_argument("--notes", default="notes")
    parser.add_argument("--q", required=True, help="Question to ask")
    parser.add_argument("--k", type=int, default=3)
    args = parser.parse_args()
 
    result = answer_question(args.q, Path(args.db), Path(args.notes), args.k)
 
    print("\n=== ANSWER ===\n")
    print(result["answer"])
    print("\n=== RETRIEVED NOTES ===")
    for r in result["retrieved_notes"]:
        print(f"- {r['filename']} ({r['date']})")
 
 
if __name__ == "__main__":
    main()
 