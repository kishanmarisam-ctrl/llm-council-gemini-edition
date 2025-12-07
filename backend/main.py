import os
import json
import re
from collections import defaultdict
from typing import List, Dict

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
import google.generativeai as genai

# -----------------------------
# Load env + Gemini client
# -----------------------------
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise RuntimeError("GEMINI_API_KEY not set in .env")

genai.configure(api_key=api_key)

# -----------------------------
# Council config (Gemini models)
# -----------------------------
# If you hit quota, switch all to "gemini-1.5-flash".
COUNCIL_MODELS = [
    "gemini-2.5-flash-lite",
    "gemini-2.5-flash-lite",
    "gemini-2.5-flash-lite",
]

CHAIRMAN_MODEL = "gemini-2.5-flash-lite"




# -----------------------------
# Pydantic models
# -----------------------------
class Question(BaseModel):
    question: str


class ModelAnswer(BaseModel):
    model_name: str
    answer: str


class ModelReview(BaseModel):
    reviewer_model: str
    scores: Dict[str, float]


class CouncilResult(BaseModel):
    final_answer: str
    all_answers: List[ModelAnswer]
    reviews: List[ModelReview]
    ranking: List[str]


# -----------------------------
# Helper: call a Gemini model
# -----------------------------
def call_model(model: str, system_prompt: str, user_prompt: str) -> str:
    """
    Simple wrapper around Gemini generate_content.
    We just concatenate system + user into one prompt string.
    """
    prompt = f"{system_prompt}\n\n{user_prompt}"
    gm = genai.GenerativeModel(model)
    response = gm.generate_content(prompt)
    # response may have candidates/parts; .text gives the combined text
    return (response.text or "").strip()


# -----------------------------
# Council logic
# -----------------------------
def get_answers(question: str) -> List[ModelAnswer]:
    answers: List[ModelAnswer] = []

    for model in COUNCIL_MODELS:
        text = call_model(
            model,
            "You are a precise expert. Answer clearly and honestly.",
            f"Question:\n{question}\n\nAnswer:",
        )
        answers.append(ModelAnswer(model_name=model, answer=text.strip()))

    return answers


def get_reviews(question: str, answers: List[ModelAnswer]) -> List[ModelReview]:
    reviews: List[ModelReview] = []

    answers_text = ""
    for i, ans in enumerate(answers):
        answers_text += f"[{i}] Model: {ans.model_name}\n{ans.answer}\n\n"

    for reviewer_model in COUNCIL_MODELS:
        system_prompt = (
            "You are a strict reviewer. You will give each answer one score from 1 to 10 "
            "for correctness, completeness, clarity, and reasoning."
        )

        user_prompt = f"""
Question:
{question}

Here are several answers to the same question:

{answers_text}

For each answer index [i], give ONE overall score from 1 to 10.

Respond ONLY with JSON in this exact format (no extra text):

{{
  "scores": {{
    "0": <score_for_answer_0>,
    "1": <score_for_answer_1>,
    ...
  }}
}}
"""

        raw = call_model(reviewer_model, system_prompt, user_prompt)

        # Try to extract JSON from output
        try:
            match = re.search(r"\{.*\}", raw, re.DOTALL)
            obj = json.loads(match.group(0)) if match else {"scores": {}}
        except Exception:
            obj = {"scores": {}}

        scores: Dict[str, float] = {}
        for idx_str, score_val in obj.get("scores", {}).items():
            try:
                idx = int(idx_str)
                if 0 <= idx < len(answers):
                    model_name = answers[idx].model_name
                    scores[model_name] = float(score_val)
            except Exception:
                continue

        reviews.append(ModelReview(reviewer_model=reviewer_model, scores=scores))

    return reviews


def rank_models(reviews: List[ModelReview]) -> List[str]:
    totals: Dict[str, float] = defaultdict(float)

    for review in reviews:
        for model_name, score in review.scores.items():
            totals[model_name] += score

    return sorted(totals.keys(), key=lambda m: totals[m], reverse=True)


def synthesize_final_answer(
    question: str, answers: List[ModelAnswer], ranking: List[str]
) -> str:
    answers_text = ""
    for ans in answers:
        answers_text += f"Model {ans.model_name}:\n{ans.answer}\n\n"

    ranking_text = " > ".join(ranking) if ranking else "(no ranking)"

    system_prompt = (
        "You are the chairman of an expert council of AI models. "
        "Read all answers and produce one final high-quality answer."
    )

    user_prompt = f"""
Question:
{question}

Answers from council models:
{answers_text}

Ranking from best to worst (based on peer reviews):
{ranking_text}

Your tasks:
1. Analyze the answers, especially higher-ranked ones.
2. Produce ONE final answer that is correct, clear, and concise.
3. Avoid obvious mistakes from weaker answers.

Now write the final answer.
"""

    final = call_model(CHAIRMAN_MODEL, system_prompt, user_prompt)
    return final.strip()


def run_council(question: str) -> CouncilResult:
    answers = get_answers(question)
    reviews = get_reviews(question, answers)
    ranking = rank_models(reviews)
    final_answer = synthesize_final_answer(question, answers, ranking)

    return CouncilResult(
        final_answer=final_answer,
        all_answers=answers,
        reviews=reviews,
        ranking=ranking,
    )


# -----------------------------
# FastAPI app
# -----------------------------
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow Render + your phone
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # ok for local dev
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ask", response_model=CouncilResult)
def ask_endpoint(q: Question):
    return run_council(q.question)
