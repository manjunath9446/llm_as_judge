import json
import os
import re
from pathlib import Path
from typing import Literal

from openai import OpenAI
from pydantic import BaseModel, Field

# Use a Judge model from a DIFFERENT family than the generator
JUDGE_MODEL = "openai/gpt-oss-20b"
# Robust absolute paths
BASE_DIR = Path(__file__).resolve().parent
LOG_FILE = BASE_DIR / "judge_audit.jsonl"
REPORT_FILE = BASE_DIR / "report.json"

class CriterionScore(BaseModel):
    score: int = Field(..., ge=1, le=5)
    rationale: str

class JudgeVerdict(BaseModel):
    winner: Literal["A", "B", "Tie"]
    response_a_criteria: dict[str, CriterionScore]
    response_b_criteria: dict[str, CriterionScore]
    overall_rationale: str

class Judge:
    def __init__(self, api_key: str):
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://api.groq.com/openai/v1"
        )
        self.system_prompt = """
You are an impartial, strict evaluation judge. You are comparing two AI outputs (Response A and Response B) against a Reference Answer.

[CRITICAL INSTRUCTIONS]
1. You must score BOTH Response A and Response B independently on every criterion (1-5).
2. Do NOT let the length of the response fool you. A shorter, correct answer is BETTER than a long, rambling, or incorrect answer.
3. RULE ENFORCEMENT: If an answer contradicts the Reference Answer, its "correctness" score MUST be 1.
4. RATIONALE CLARITY: When writing the rationale for "correctness", explicitly state what the Reference Answer says, and what the Response says. Do NOT confuse them. Example: "The Reference states $299, but the Response states $399. Therefore, it is factually incorrect."
5. CONCISENESS VS CORRECTNESS: A short but factually wrong answer should get 1/5 for Correctness, but it CAN get 5/5 for Conciseness because it didn't ramble. Evaluate them independently.
6. You MUST output your final verdict as a valid JSON object matching the schema below.

[OUTPUT SCHEMA]
{
  "winner": "A" | "B" | "Tie",
  "response_a_criteria": {
    "<criterion_name>": {"score": 1-5, "rationale": "..."}
  },
  "response_b_criteria": {
    "<criterion_name>": {"score": 1-5, "rationale": "..."}
  },
  "overall_rationale": "Summary of why the winner won."
}
"""

    def _parse_json(self, text: str) -> dict:
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            match = re.search(r'\{.*\}', text, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group(0))
                except json.JSONDecodeError:
                    pass
        raise ValueError(f"Could not parse JSON from response: {text[:100]}")

    def _call_judge(self, prompt: str, run_id: str) -> JudgeVerdict:
        resp = self.client.chat.completions.create(
            model=JUDGE_MODEL,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0,
            response_format={"type": "json_object"}
        )
        
        raw_output = resp.choices[0].message.content
        tokens = resp.usage.total_tokens
        
        with open(LOG_FILE, "a") as f:
            f.write(json.dumps({
                "run_id": run_id,
                "model": JUDGE_MODEL,
                "prompt": prompt,
                "raw_output": raw_output,
                "tokens": tokens
            }) + "\n")

        try:
            parsed = self._parse_json(raw_output)
            return JudgeVerdict(**parsed)
        except Exception as e:
            print(f"Parse error: {e}")
            empty_criteria = {"correctness": {"score": 1, "rationale": "Parse error"}}
            return JudgeVerdict(
                winner="Tie", 
                response_a_criteria=empty_criteria, 
                response_b_criteria=empty_criteria,
                overall_rationale="Parsing failed."
            )

    def evaluate_pair(self, tc: dict, rubric: dict) -> dict:
        rubric_str = "\n".join([f"- {k}: {v}" for k, v in rubric.items()])
        
        # 1. Order A vs B
        prompt_ab = f"""
[RUBRIC]
{rubric_str}

[QUESTION]
{tc['input']}

[REFERENCE]
{tc['expected_output']}

[RESPONSE A]
{tc['config_a_output']}

[RESPONSE B]
{tc['config_b_output']}
"""
        verdict_ab = self._call_judge(prompt_ab, f"{tc['id']}_AB")

        # 2. Order B vs A
        prompt_ba = f"""
[RUBRIC]
{rubric_str}

[QUESTION]
{tc['input']}

[REFERENCE]
{tc['expected_output']}

[RESPONSE A]
{tc['config_b_output']}

[RESPONSE B]
{tc['config_a_output']}
"""
        verdict_ba = self._call_judge(prompt_ba, f"{tc['id']}_BA")

        # Determine consistency
        consistent = (
            (verdict_ab.winner == "A" and verdict_ba.winner == "B") or
            (verdict_ab.winner == "B" and verdict_ba.winner == "A") or
            (verdict_ab.winner == "Tie" and verdict_ba.winner == "Tie")
        )
        
        # Determine actual winner mapping back to configs
        actual_winner = "Tie"
        if verdict_ab.winner == "A" and verdict_ba.winner == "B":
            actual_winner = "A"
        elif verdict_ab.winner == "B" and verdict_ba.winner == "A":
            actual_winner = "B"

        return {
            "id": tc["id"],
            "input": tc["input"],
            "actual_winner": actual_winner,
            "consistent": consistent,
            "verdict_ab": verdict_ab.model_dump(),
            "verdict_ba": verdict_ba.model_dump()
        }

def run_suite(suite_path: str, api_key: str) -> dict:
    LOG_FILE.parent.mkdir(exist_ok=True)
    LOG_FILE.touch()

    with open(suite_path) as f:
        suite = json.load(f)

    judge = Judge(api_key)
    results = []

    for tc in suite["test_cases"]:
        print(f"Evaluating {tc['id']}...")
        res = judge.evaluate_pair(tc, suite["rubric"])
        results.append(res)

    a_wins = sum(1 for r in results if r["actual_winner"] == "A")
    b_wins = sum(1 for r in results if r["actual_winner"] == "B")
    ties = sum(1 for r in results if r["actual_winner"] == "Tie")
    flips = sum(1 for r in results if not r["consistent"])

    report = {
        "suite": suite["suite_name"],
        "model_used": JUDGE_MODEL,
        "metrics": {
            "config_a_win_rate": f"{(a_wins / len(results)) * 100:.1f}%",
            "config_b_win_rate": f"{(b_wins / len(results)) * 100:.1f}%",
            "tie_rate": f"{(ties / len(results)) * 100:.1f}%",
            "position_bias_flip_rate": f"{(flips / len(results)) * 100:.1f}%"
        },
        "details": results
    }

    with open(REPORT_FILE, "w") as f:
        json.dump(report, f, indent=2)
    
    return report

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    api_key = os.getenv("RAG_GROK_API_KEY")
    if not api_key:
        raise ValueError("RAG_GROK_API_KEY not found in .env")
        
    report = run_suite("test_suite.json", api_key)
    
    print("\n=== EVALUATION REPORT ===")
    print(json.dumps(report["metrics"], indent=2))