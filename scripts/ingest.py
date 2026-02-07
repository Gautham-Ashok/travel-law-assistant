import json
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv
from chromadb import PersistentClient
from tqdm import tqdm
from litellm import completion
from tenacity import retry, wait_exponential

load_dotenv(override=True)

# ---------- CONFIG ----------
MODEL = "gpt-4.1-nano"  
EMBEDDING_MODEL = "text-embedding-3-small"
WORKERS = 1  # Keep to 1 to avoid rate limits/locking issues during debugging

BASE_DIR = Path(__file__).parent.parent
# Ensure this path matches where you saved your markdown files
KNOWLEDGE_BASE_PATH = BASE_DIR / "data/markdown/filtered"
DB_PATH = BASE_DIR / "preprocessed_db"
COLLECTION_NAME = "india_travel_law"

openai = OpenAI()
chroma = PersistentClient(path=str(DB_PATH))

# ----------------------------

def load_documents():
    docs = []
    if not KNOWLEDGE_BASE_PATH.exists():
        print(f"❌ Error: Path {KNOWLEDGE_BASE_PATH} does not exist!")
        return []
    
    # Sort files to ensure consistent processing order
    files = list(KNOWLEDGE_BASE_PATH.glob("*.md"))
    if not files:
        print(f"⚠️ Warning: No .md files found in {KNOWLEDGE_BASE_PATH}")
        return []

    for file in files:
        docs.append({
            "type": file.stem,
            "source": file.name,
            "text": file.read_text(encoding="utf-8")
        })
    print(f"📂 Loaded {len(docs)} markdown documents")
    return docs

def build_prompt(text):
    return f"""
Act as a Legal Data Engineer. Split this Indian Law text into logical chunks.
Each chunk must preserve the original legal wording and Section numbers.

Return a JSON object with this structure:
{{
  "chunks": [
    {{
      "headline": "Section XX: Name",
      "summary": "Brief explanation",
      "original_text": "Full legal text"
    }}
  ]
}}

TEXT:
{text}
"""

@retry(wait=wait_exponential(multiplier=1, min=4, max=10))
def process_document(doc):
    try:
        response = completion(
            model=MODEL,
            messages=[{"role": "user", "content": build_prompt(doc["text"])}],
            response_format={ "type": "json_object" } 
        )
        
        content = response.choices[0].message.content
        data = json.loads(content)
        
        results = []
        for c in data.get("chunks", []):
            # --- THE FIX: USE .get() ---
            # This prevents crashing if 'original_text' is missing
            headline = c.get('headline', 'Section Unknown')
            summary = c.get('summary', '')
            original = c.get('original_text', '') 
            
            # Fallback: If original text is missing, use the summary as the text
            if not original and summary:
                original = summary

            combined_text = f"{headline}\n{summary}\n{original}"
            
            if combined_text.strip():
                results.append({
                    "content": combined_text,
                    "metadata": {"source": doc["source"], "law_type": doc["type"]}
                })
        return results
    except Exception as e:
        print(f"⚠️ Error processing {doc['source']}: {e}")
        return []

def embed_and_store(all_results):
    if not all_results:
        print("❌ No chunks created.")
        return

    # Clear old data
    try:
        if COLLECTION_NAME in [c.name for c in chroma.list_collections()]:
            chroma.delete_collection(COLLECTION_NAME)
    except Exception as e:
        print(f"Note: Collection cleanup skipped ({e})")
    
    collection = chroma.create_collection(COLLECTION_NAME)

    texts = [r["content"] for r in all_results]
    metadatas = [r["metadata"] for r in all_results]
    ids = [f"id_{i}" for i in range(len(texts))]

    print(f"🧠 Generating embeddings for {len(texts)} chunks...")
    emb_response = openai.embeddings.create(
        model=EMBEDDING_MODEL,
        input=texts
    )
    embeddings = [e.embedding for e in emb_response.data]

    collection.add(
        ids=ids,
        documents=texts,
        embeddings=embeddings,
        metadatas=metadatas
    )
    print(f"✅ Success! Database created with {collection.count()} legal chunks.")

if __name__ == "__main__":
    docs = load_documents()
    if docs:
        all_chunks = []
        # Using a loop instead of Pool for better error visibility
        for d in tqdm(docs, desc="Processing Laws"):
            chunks = process_document(d)
            if chunks:
                all_chunks.extend(chunks)
            else:
                print(f"⚠️ Warning: No chunks generated for {d['source']}")
        
        embed_and_store(all_chunks)

