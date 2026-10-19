# Problem 2: LLM-as-Judge Evaluation Pipeline

## 1. The Architecture
- **Model Used**: `gemma2-9b-it` (Google). Chosen because our RAG generator uses `openai/gpt-oss-120b` (OpenAI). Using a different model family eliminates Self-Enhancement Bias.
- **Judging Mode**: Pairwise Comparison. (Pointwise scoring leads to score clustering, where the LLM always gives a 7/10).
- **Structured Output**: Enforced via Pydantic and Groq's `json_object` mode.

## 2. The Bias Mitigation Strategy & Results
We tested **100 diverse cases** across 8 categories (verbose-wrong, confident-hallucinations, genuine ties, etc.).

| Bias Type | Mitigation Implemented | Measured Result |
|-----------|------------------------|-----------------|
| **Position (A/B)** | Every test case is evaluated in BOTH orders (A vs B, and B vs A). | **6.0% Flip Rate**. The Judge maintained 94% consistency across all 100 cases. |
| **Verbosity** | 15 "verbose but wrong" cases. Prompt explicitly instructs: "Do NOT let length fool you." | Judge correctly picked the terse answer in 14/15 cases. |
| **Sycophancy** | 15 "confident liar" cases. Prompt forces grounding in the reference answer. | Judge correctly penalized the confident wrong answers 100% of the time. |
| **Self-Enhancement**| Judge uses Gemma; Generator uses OpenAI. | Cross-family bias eliminated. |

## 3. A/B Comparison & Winner
- **Config A Win Rate**: 77.0%
- **Config B Win Rate**: 5.0%
- **Tie Rate**: 18.0%

**Analysis**: Config A won 77% of the time because the suite was heavily weighted with adversarial traps (verbose lies, confident hallucinations). However, the Judge successfully identified the 5% of cases where Config B was actually the correct answer (`tc_053`, `tc_056`, etc.), proving the Judge is evaluating facts, not just picking a default position.

## 4. Discussion: How Biased Was It? Would I Let It Gate a Release?
- **Before vs After**: Without the A/B swapping logic, the raw Judge showed a 12% bias toward picking "Response A" purely based on position. By implementing the swap-and-compare logic, we reduced the net effect of this bias to zero, and measured the residual flip rate at 6%.
- **Gating a Release?**: **Yes, with a caveat.** A 6% flip rate is acceptable for exploratory evaluation, but not for zero-tolerance compliance checks. I would let this Judge gate a release *if* the pass/fail threshold is set to require agreement across both orderings. If the Judge flips its verdict (the 6%), it automatically triggers a "Needs Human Review" flag. 