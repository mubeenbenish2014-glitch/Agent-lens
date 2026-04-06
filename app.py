
import streamlit as st
import pandas as pd
from agent_core import (
    use_responses_api,
    use_chat_completions,
    use_ollama,
    parse_response
)

# ── Page Setup ─────────────────────────────────────────────────
st.set_page_config(page_title="AgentLens", page_icon="🔍", layout="wide")
st.title("🔍 AgentLens")
st.caption("AI-Powered LLM Discovery Assistant for Agentic Workflows")

# ── Sidebar ────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")

    mode = st.radio("🔌 Select API Mode:", [
        "🌐 Responses API (Web Search)",
        "💬 Chat Completions API",
        "🦙 Ollama (Local LLM)"
    ])

    ollama_model = "llama3.2"
    if "Ollama" in mode:
        ollama_model = st.text_input("Ollama Model Name:", value="llama3.2")
        st.info("Make sure Ollama is running at localhost:11434")

    st.markdown("---")
    st.markdown("### 📖 Mode Guide")
    st.markdown("""
- 🌐 **Responses API** — Live web search, latest info  
- 💬 **Chat Completions** — GPT-4o knowledge only  
- 🦙 **Ollama** — Free, local, private
    """)

    st.markdown("---")
    st.markdown("### 🕓 Search History")
    if "history" not in st.session_state:
        st.session_state.history = []
    for h in st.session_state.history[-5:]:
        st.markdown(f"- {h[:45]}...")

# ── Query Input ────────────────────────────────────────────────
st.markdown("### 📝 Describe Your Agentic Workflow")
query = st.text_area("", height=120, placeholder=(
    "e.g. I'm building a marketing automation agent that creates campaigns, "
    "targets audiences, and generates reports. Recommend the best LLMs."
))

col1, col2 = st.columns([1, 5])
search = col1.button("🔍 Search LLMs", use_container_width=True)
col2.button("🗑️ Clear", on_click=lambda: st.session_state.update({"history": st.session_state.history}))

# ── Search & Display ───────────────────────────────────────────
if search:
    if not query.strip():
        st.warning("⚠️ Please enter a workflow description.")
    else:
        st.session_state.history.append(query)

        with st.spinner("🔎 Searching for best LLMs..."):
            if "Responses" in mode:
                raw = use_responses_api(query)
            elif "Chat" in mode:
                raw = use_chat_completions(query)
            else:
                raw = use_ollama(query, ollama_model)

        models = parse_response(raw)

        # ── Mode Badge ─────────────────────────────────────────
        badge = {"🌐 Responses API (Web Search)": "🌐 Web Search",
                 "💬 Chat Completions API": "💬 Chat Completions",
                 "🦙 Ollama (Local LLM)": "🦙 Ollama Local"}
        st.success(f"✅ Results via **{badge.get(mode, mode)}** — {len(models)} LLMs found")

        if not models:
            st.warning("Could not parse structured output. Raw response:")
            st.markdown(raw)
        else:
            # ── LLM Cards ──────────────────────────────────────
            st.markdown("## 🤖 Recommended LLMs")
            cols = st.columns(2)
            for i, m in enumerate(models):
                with cols[i % 2]:
                    tool = m.get("Tool Calling", "")
                    icon = "🟢" if "Yes" in tool else "🔴"
                    score = m.get("Suitability Score", "N/A")

                    with st.expander(f"**#{i+1} — {m.get('Model Name', 'Unknown')}**  |  Score: {score}/10", expanded=True):
                        st.markdown(f"🏢 **Provider:** {m.get('Provider', 'N/A')}")
                        st.markdown(f"📏 **Parameters:** `{m.get('Parameters', 'N/A')}`")
                        st.markdown(f"📝 **Description:** {m.get('Description', 'N/A')}")
                        st.markdown(f"⚡ **Key Features:** {m.get('Key Features', 'N/A')}")
                        st.markdown(f"🔧 **Tool Calling:** {icon} {tool}")
                        st.markdown(f"💰 **Cost Tier:** {m.get('Cost Tier', 'N/A')}")

            # ── Comparison Table ───────────────────────────────
            st.markdown("## 📊 Comparison Table")
            df = pd.DataFrame(models)
            if not df.empty:
                st.dataframe(df, use_container_width=True, height=250)

            # ── Raw Response ───────────────────────────────────
            with st.expander("📄 View Raw API Response"):
                st.text(raw)