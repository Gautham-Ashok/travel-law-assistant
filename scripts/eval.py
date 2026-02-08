import sys
import math
import json
import asyncio
from pathlib import Path
from pydantic import BaseModel, Field
from openai import OpenAI
from dotenv import load_dotenv

# Import your actual project logic
from scripts.answer import answer_question, retrieve_chunks

load_dotenv(override=True)

MODEL = "gpt-4.1-nano"
TEST_FILE = str(Path(__file__).parent.parent / "data" / "tests.jsonl")

client = OpenAI()

# --- DATA MODELS ---

class TestQuestion(BaseModel):
    """A test question with expected keywords and reference answer."""
    question: str
    keywords: list[str]
    reference_answer: str
    category: str

class RetrievalEval(BaseModel):
    """Metrics for retrieval performance."""
    mrr: float
    ndcg: float
    keywords_found: int
    total_keywords: int
    keyword_coverage: float

class AnswerEval(BaseModel):
    """LLM-as-a-judge evaluation."""
    feedback: str = Field(description="Concise feedback on answer quality vs reference.")
    accuracy: float = Field(description="1 (wrong) to 5 (perfect).")
    completeness: float = Field(description="1 (missing info) to 5 (comprehensive).")
    relevance: float = Field(description="1 (off-topic) to 5 (direct answer).")

# --- HELPER FUNCTIONS ---

def load_tests() -> list[TestQuestion]:
    tests = []
    with open(TEST_FILE, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data = json.loads(line.strip())
                tests.append(TestQuestion(**data))
    return tests

def calculate_mrr(keyword: str, retrieved_docs: list) -> float:
    """Calculate Reciprocal Rank (1/Rank)."""
    keyword_lower = keyword.lower()
    for rank, doc in enumerate(retrieved_docs, start=1):
        # ADAPTATION: Accessing dictionary key 'text' instead of object attribute
        content = doc.get("text", "").lower()
        if keyword_lower in content:
            return 1.0 / rank
    return 0.0

def calculate_ndcg(keyword: str, retrieved_docs: list, k: int = 5) -> float:
    """Calculate nDCG (Normalized Discounted Cumulative Gain)."""
    keyword_lower = keyword.lower()
    
    # Binary relevance: 1 if found, 0 if not
    relevances = [
        1 if keyword_lower in doc.get("text", "").lower() else 0 
        for doc in retrieved_docs[:k]
    ]
    
    # DCG
    dcg = 0.0
    for i in range(min(k, len(relevances))):
        dcg += relevances[i] / math.log2(i + 2)

    # Ideal DCG (Sorted)
    ideal_relevances = sorted(relevances, reverse=True)
    idcg = 0.0
    for i in range(min(k, len(ideal_relevances))):
        idcg += ideal_relevances[i] / math.log2(i + 2)

    return dcg / idcg if idcg > 0 else 0.0

# --- CORE EVALUATION FUNCTIONS ---

def evaluate_retrieval(test: TestQuestion) -> RetrievalEval:
    # 1. Retrieve using your actual script
    # Note: retrieve_chunks returns [{'text':..., 'meta':...}, ...]
    retrieved_docs = retrieve_chunks(test.question)
    
    # 2. Calculate Metrics
    mrr_scores = [calculate_mrr(kw, retrieved_docs) for kw in test.keywords]
    avg_mrr = sum(mrr_scores) / len(mrr_scores) if mrr_scores else 0.0

    ndcg_scores = [calculate_ndcg(kw, retrieved_docs) for kw in test.keywords]
    avg_ndcg = sum(ndcg_scores) / len(ndcg_scores) if ndcg_scores else 0.0

    keywords_found = sum(1 for s in mrr_scores if s > 0)
    total = len(test.keywords)
    coverage = (keywords_found / total * 100) if total > 0 else 0.0

    return RetrievalEval(
        mrr=avg_mrr,
        ndcg=avg_ndcg,
        keywords_found=keywords_found,
        total_keywords=total,
        keyword_coverage=coverage
    )

def evaluate_answer(test: TestQuestion) -> tuple[AnswerEval, str]:
    # 1. Generate Answer (Pass history=[])
    generated_answer, chunks = answer_question(test.question, history=[])

    # 2. LLM Judge Prompt
    judge_prompt = f"""
    You are an expert legal evaluator. Compare the Generated Answer to the Reference Answer.
    
    Question: {test.question}
    Reference Answer: {test.reference_answer}
    Generated Answer: {generated_answer}
    
    Evaluate on:
    1. Accuracy (1-5): Is the legal information (fines, jail terms, sections) correct?
    2. Completeness (1-5): Did it miss key details from the reference?
    3. Relevance (1-5): Did it answer the specific question?
    
    Return JSON format matching the AnswerEval schema.
    """

    response = client.beta.chat.completions.parse(
        model=MODEL,
        messages=[{"role": "user", "content": judge_prompt}],
        response_format=AnswerEval,
        temperature=0
    )

    return response.choices[0].message.parsed, generated_answer

# --- GENERATORS FOR DASHBOARD ---

def run_all_retrieval_tests():
    tests = load_tests()
    total = len(tests)
    for i, test in enumerate(tests):
        res = evaluate_retrieval(test)
        yield test, res, (i+1)/total

def run_all_answer_tests():
    tests = load_tests()
    total = len(tests)
    for i, test in enumerate(tests):
        res, ans = evaluate_answer(test)
        yield test, res, (i+1)/total