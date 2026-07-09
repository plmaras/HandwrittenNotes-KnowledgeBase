import argparse
import re
from pathlib import Path
 
import chromadb
from sentence_transformers import SentenceTransformer
 
EMBED_MODEL_NAME = "BAAI/bge-small-en-v1.5"
 
 
def parse_note(path: Path) -> dict:
    text = path.read_text(encoding="utf-8")
 
    frontmatter_match = re.search(r"^---\n(.*?)\n---\n(.*)$", text, re.DOTALL)
    if not frontmatter_match:
        return {"body": text, "summary": "", "date": "", "source_image": "", "uncertain_count": 0}
 
    frontmatter_raw, body = frontmatter_match.groups()
 
    def field(name, default=""):
        m = re.search(rf'^{name}:\s*"?(.*?)"?\s*$', frontmatter_raw, re.MULTILINE)
        return m.group(1) if m else default
 
    return {
        "body": body.strip(),
        "summary": field("summary"),
        "date": field("date"),
        "source_image": field("source_image"),
        "uncertain_count": int(field("uncertain_words", "0") or 0),
    }
 
 
def build_index(notes_dir: Path, db_dir: Path):
    print(f"Loading embedding model {EMBED_MODEL_NAME}...")
    model = SentenceTransformer(EMBED_MODEL_NAME)
 
    client = chromadb.PersistentClient(path=str(db_dir))
    collection = client.get_or_create_collection("handwritten_notes")
 
    note_paths = sorted(p for p in notes_dir.glob("*.md"))
    if not note_paths:
        print(f"No .md files found in {notes_dir}")
        return
 
    ids, documents, metadatas = [], [], []
 
    for note_path in note_paths:
        parsed = parse_note(note_path)
 
    
        embed_text = f"{parsed['summary']}\n\n{parsed['body']}".strip()
 
        ids.append(note_path.stem)
        documents.append(embed_text)
        metadatas.append({
            "filename": note_path.name,
            "date": parsed["date"],
            "source_image": parsed["source_image"],
            "uncertain_count": parsed["uncertain_count"],
        })
 
    print(f"Embedding {len(documents)} notes...")
    embeddings = model.encode(documents, normalize_embeddings=True).tolist()
 
    
    collection.upsert(ids=ids, embeddings=embeddings, documents=documents, metadatas=metadatas)
 
    print(f"Indexed {len(documents)} notes into {db_dir}")
 
 
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--notes", default="notes")
    parser.add_argument("--db", default="chroma_db")
    args = parser.parse_args()
 
    build_index(Path(args.notes), Path(args.db))
 
 
if __name__ == "__main__":
    main()
 