from openai import OpenAI
from dotenv import load_dotenv
from chromadb import PersistentClient
from pathlib import Path

load_dotenv(override=True)

# ---------------- CONFIG ----------------
MODEL = "gpt-4.1-nano"
QUERY_EMBED_MODEL = "text-embedding-3-small"

BASE_DIR = Path(__file__).parent.parent
DB_PATH = BASE_DIR / "preprocessed_db"
COLLECTION_NAME = "india_travel_law"

RETRIEVAL_K = 5
MAX_CONTEXT_CHARS = 3500 

# Hard-coded disclaimer to ensure it ALWAYS appears
LEGAL_DISCLAIMER = """
\n\n---
⚠️ **Disclaimer:** *I am an AI assistant, not a lawyer. This information is for general guidance only and serves as a reference to Indian laws. For serious legal matters, please contact a qualified advocate or the nearest police station.*
"""
# ---------------------------------------

client = OpenAI()
chroma = PersistentClient(path=str(DB_PATH))
collection = chroma.get_collection(COLLECTION_NAME)

# --- UPDATED PROMPT (Natural Flow) ---
SYSTEM_PROMPT = """
You are an India Travel Law Information Assistant. 
Answer the user's question using the provided Context.

**INSTRUCTIONS:**
- Do not use empty fields like "Not specified".
- If a fine/jail term is not listed, explain the general legal consequence (e.g., deportation, arrest, or police action) in a full sentence.
- Use the following structure:

**1. The Law & Section**
[State the Act and Section number clearly]

**2. The Rule**
[Explain what is allowed or prohibited in clear, simple sentences.]

**3. Consequences**
[Describe the penalties, fines, jail time, or other risks (like deportation/blacklisting) in full sentences.]

**4. Traveller Guidance**
[Practical advice on what to do. If the situation is dangerous, advise calling 112.]

**Context:**
{context}
"""

def retrieve_chunks(question: str):
    embed_resp = client.embeddings.create(
        model=QUERY_EMBED_MODEL,
        input=[question]
    )
    query_vector = embed_resp.data[0].embedding

    results = collection.query(
        query_embeddings=[query_vector],
        n_results=RETRIEVAL_K
    )

    chunks = []
    if results["documents"] and len(results["documents"]) > 0:
        for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
            chunks.append({"text": doc, "meta": meta})
    
    return chunks

def answer_question(question: str):
    if not question.strip():
        return "Please enter a valid question.", []

    print(f"▶️ Querying: {question}")
    
    try:
        chunks = retrieve_chunks(question)
        
        # Guard Clause
        if not chunks:
            return "I couldn't find specific legal sections for that in my database. Please consult a local authority." + LEGAL_DISCLAIMER, []

        context_text = "\n\n---\n\n".join([c["text"] for c in chunks])
        context_text = context_text[:MAX_CONTEXT_CHARS]

        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT.format(context=context_text)},
                {"role": "user", "content": question}
            ],
            temperature=0
        )
        
        raw_answer = response.choices[0].message.content
        final_answer = raw_answer + LEGAL_DISCLAIMER
        
        return final_answer, chunks

    except Exception as e:
        print(f"❌ Error in answer_question: {e}")
        return f"System Error: {str(e)}", []