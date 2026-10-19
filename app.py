import json
import streamlit as st
from pathlib import Path

st.set_page_config(page_title="LLM-as-Judge", layout="wide")
st.title("⚖️ LLM-as-Judge: Problem 2")

BASE_DIR = Path(__file__).resolve().parent
report_path = BASE_DIR / "report.json"

if not report_path.exists():
    st.warning("No report found. Run `python problem2_judge/judge.py` first.")
    st.stop()

report = json.loads(report_path.read_text())
metrics = report["metrics"]

st.header("Aggregate Metrics")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Config A Win Rate", metrics["config_a_win_rate"])
col2.metric("Config B Win Rate", metrics["config_b_win_rate"])
col3.metric("Tie Rate", metrics["tie_rate"])
col4.metric("Position Bias Flip Rate", metrics["position_bias_flip_rate"], 
            help="Percentage of cases where the judge changed its winner when Response A and B were swapped.")

st.divider()
st.header("Detailed Test Cases")

for detail in report["details"]:
    with st.expander(f"Test Case: {detail['id']} - Winner: Config {detail['actual_winner']}"):
        if not detail["consistent"]:
            st.error("🚨 **Position Bias Detected**: The judge changed its winner when the order was flipped!")
        else:
            st.success("✅ Consistent Verdict across both orders.")
            
        st.markdown(f"**Question:** {detail['input']}")
        st.markdown(f"**True Winner:** Config `{detail['actual_winner']}`")

        st.subheader("1. Original Order (Config A shown first, Config B shown second)")
        st.write(f"**Judge declared:** Response {detail['verdict_ab']['winner']} (which maps to Config {'A' if detail['verdict_ab']['winner']=='A' else 'B' if detail['verdict_ab']['winner']=='B' else 'Tie'})")
        
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("**Response A (Config A)**")
            for crit, data in detail["verdict_ab"]["response_a_criteria"].items():
                st.markdown(f"- **{crit.capitalize()}**: `{data['score']}/5` — *{data['rationale']}*")
        with col_b:
            st.markdown("**Response B (Config B)**")
            for crit, data in detail["verdict_ab"]["response_b_criteria"].items():
                st.markdown(f"- **{crit.capitalize()}**: `{data['score']}/5` — *{data['rationale']}*")

        st.divider()
        
        st.subheader("2. Swapped Order (Config B shown first, Config A shown second)")
        st.write(f"**Judge declared:** Response {detail['verdict_ba']['winner']} (which maps to Config {'B' if detail['verdict_ba']['winner']=='A' else 'A' if detail['verdict_ba']['winner']=='B' else 'Tie'})")

        col_a_swap, col_b_swap = st.columns(2)
        with col_a_swap:
            st.markdown("**Response A (Config B)**")
            for crit, data in detail["verdict_ba"]["response_a_criteria"].items():
                st.markdown(f"- **{crit.capitalize()}**: `{data['score']}/5` — *{data['rationale']}*")
        with col_b_swap:
            st.markdown("**Response B (Config A)**")
            for crit, data in detail["verdict_ba"]["response_b_criteria"].items():
                st.markdown(f"- **{crit.capitalize()}**: `{data['score']}/5` — *{data['rationale']}*")