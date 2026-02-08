import os
from openai import OpenAI
from dotenv import load_dotenv
from chromadb import PersistentClient
from pathlib import Path
from sentence_transformers import CrossEncoder

load_dotenv(override=True)

# ---------------- CONFIG ----------------
MODEL = "gpt-4o"
QUERY_EMBED_MODEL = "text-embedding-3-small"

BASE_DIR = Path(__file__).parent.parent
DB_PATH = BASE_DIR / "preprocessed_db"
COLLECTION_NAME = "india_travel_law"

RETRIEVAL_K = 5
MAX_CONTEXT_CHARS = 3500 

LEGAL_DISCLAIMER = """
\n\n---
⚠️ **Disclaimer:** *I am an AI assistant, not a lawyer. This information is for general guidance only and serves as a reference to Indian laws. For serious legal matters, please contact a qualified advocate or the nearest police station.*
"""
# ---------------------------------------

# Initialize Clients
client = OpenAI()
chroma = PersistentClient(path=str(DB_PATH))
collection = chroma.get_collection(COLLECTION_NAME)

# Initialize Reranker
print("⏳ Loading Reranker Model...")
reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
print("✅ Reranker Loaded!")

# --- PROMPTS ---
CONTEXTUALIZE_Q_SYSTEM_PROMPT = """Given a chat history and the latest user question which might reference context in the chat history, formulate a standalone question which can be understood without the chat history. Do NOT answer the question, just reformulate it if needed and otherwise return it as is."""

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

def contextualize_query(question: str, history: list):
    """
    Rewrites query using history (Handles BOTH Old Tuples and New Dicts).
    """
    if not history:
        return question

    messages = [{"role": "system", "content": CONTEXTUALIZE_Q_SYSTEM_PROMPT}]
    
    # DEBUG: See what history looks like
    # print(f"DEBUG: Processing history item 0 type: {type(history[0])}")

    for turn in history:
        # Case A: Dictionary (Gradio 4/5 "messages" format)
        if isinstance(turn, dict):
            role = turn.get("role")
            content = turn.get("content")
            if role and content:
                messages.append({"role": role, "content": content})
        
        # Case B: List/Tuple (Gradio 3/4 standard format)
        elif isinstance(turn, (list, tuple)) and len(turn) >= 2:
            user_msg = str(turn[0]) if turn[0] is not None else ""
            bot_msg = str(turn[1]) if turn[1] is not None else ""
            
            if user_msg:
                messages.append({"role": "user", "content": user_msg})
            if bot_msg:
                messages.append({"role": "assistant", "content": bot_msg})

    messages.append({"role": "user", "content": question})

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            temperature=0.3 
        )
        rewritten_query = response.choices[0].message.content
        return rewritten_query
    except Exception as e:
        print(f"⚠️ Rewrite failed: {e}")
        return question

def retrieve_chunks(question: str):
    """
    Retrieves chunks with Semantic Reranking.
    """
    # 1. Fetch broad candidates
    embed_resp = client.embeddings.create(
        model=QUERY_EMBED_MODEL,
        input=[question]
    )
    query_vector = embed_resp.data[0].embedding

    results = collection.query(
        query_embeddings=[query_vector],
        n_results=30
    )

    initial_chunks = []
    if results["documents"] and len(results["documents"]) > 0:
        for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
            initial_chunks.append({"text": doc, "meta": meta})

    if not initial_chunks:
        return []

    # 2. RERANKING STEP
    pairs = [[question, chunk["text"]] for chunk in initial_chunks]
    scores = reranker.predict(pairs)

    scored_chunks = []
    for i, chunk in enumerate(initial_chunks):
        chunk["score"] = scores[i]
        scored_chunks.append(chunk)

    scored_chunks.sort(key=lambda x: x["score"], reverse=True)

    # 3. Return Top K
    final_top_k = scored_chunks[:RETRIEVAL_K]
    
    return final_top_k

def answer_question(question: str, history: list = None):
    """
    Main function to generate an answer.
    """
    if not question.strip():
        return "Please enter a valid question.", []

    # 1. REWRITE THE QUESTION
    search_query = contextualize_query(question, history)
    
    # PRINT DEBUG INFO TO TERMINAL
    print(f"\n💬 History Length: {len(history) if history else 0}")
    print(f"▶️ User Question: {question}")
    print(f"🔄 Rewritten Query: {search_query}")
    
    try:
        # 2. Retrieve using RERANKER
        chunks = retrieve_chunks(search_query)
        
        if not chunks:
            return "I couldn't find specific legal sections for that in my database. Please consult a local authority." + LEGAL_DISCLAIMER, []

        context_text = "\n\n---\n\n".join([c["text"] for c in chunks])
        context_text = context_text[:MAX_CONTEXT_CHARS]

        # 3. Generate Answer
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT.format(context=context_text)},
                {"role": "user", "content": search_query}
            ],
            temperature=0
        )
        
        raw_answer = response.choices[0].message.content
        final_answer = raw_answer + LEGAL_DISCLAIMER
        
        return final_answer, chunks

    except Exception as e:
        print(f"❌ Error in answer_question: {e}")
        return f"System Error: {str(e)}", []