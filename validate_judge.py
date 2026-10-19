import json
import os
from dotenv import load_dotenv
from judge import run_suite

def validate():
    print("Running Test 1 (Baseline)...")
    r1 = run_suite("test_suite.json", os.getenv("RAG_GROK_API_KEY"))
    
    print("\nRunning Test 2 (Test-Retest Consistency)...")
    r2 = run_suite("test_suite.json", os.getenv("RAG_GROK_API_KEY"))

    # Compare winners between the two runs
    flips = 0
    for i in range(len(r1["details"])):
        if r1["details"][i]["actual_winner"] != r2["details"][i]["actual_winner"]:
            flips += 1

    print("\n=== VALIDATION RESULTS ===")
    print(f"Test-Retest Flip Rate (Lower is better): {(flips / len(r1['details'])) * 100:.1f}%")
    
    # Check adversarial probes
    print("\n=== ADVERSARIAL PROBE CHECK ===")
    for d in r1["details"]:
        if "verbose" in d["id"] or "confident" in d["id"]:
            print(f"Test: {d['id']} -> Winner: {d['actual_winner']}")
            if d["actual_winner"] == "B":
                print("🚨 FAIL: Judge was fooled by the verbose/confident liar (Config B)!")
            else:
                print("✅ PASS: Judge correctly identified Config A as better.")

if __name__ == "__main__":
    load_dotenv()
    validate()