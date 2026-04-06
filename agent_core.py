from openai import OpenAI
from config import OPENAI_API_KEY, OPENAI_MODEL, OLLAMA_BASE_URL, OLLAMA_DEFAULT_MODEL

# ── Clients ────────────────────────────────────────────────────
openai_client = OpenAI(api_key=OPENAI_API_KEY)
ollama_client = OpenAI(base_url=OLLAMA_BASE_URL, api_key="ollama")

# ── System Prompt ──────────────────────────────────────────────
SYSTEM_PROMPT = """
You are AgentLens, an expert AI assistant for agentic workflow LLM discovery.
Recommend exactly 5 LLMs using this structured format:

---
**Model Name:** [name]
**Provider:** [company]
**Parameters:** [e.g. 7B, 70B, 405B]
**Description:** [1-2 sentences]
**Key Features:** [feature1, feature2, feature3]
**Tool Calling Support:** [Yes/No + detail]
**Cost Tier:** [Free / Low / Medium / High]
**Suitability Score:** [1-10]
---

Always return exactly this format. Be specific and accurate.
"""

# ── 1. Responses API + Web Search ─────────────────────────────
def use_responses_api(query: str) -> str:
    try:
        response = openai_client.responses.create(
            model=OPENAI_MODEL,
            tools=[{"type": "web_search_preview"}],
            instructions=SYSTEM_PROMPT,
            input=query
        )
        return response.output_text
    except Exception as e:
        return f"❌ Responses API Error: {str(e)}"

# ── 2. Chat Completions API ────────────────────────────────────
def use_chat_completions(query: str) -> str:
    try:
        response = openai_client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": query}
            ]
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"❌ Chat Completions Error: {str(e)}"

# ── 3. Ollama Local LLM ────────────────────────────────────────
def use_ollama(query: str, model: str = OLLAMA_DEFAULT_MODEL) -> str:
    try:
        response = ollama_client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": query}
            ]
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"❌ Ollama Error: {str(e)}"

# ── Parser: Raw Text → List of Dicts ──────────────────────────
def parse_response(raw: str) -> list:
    models = []
    blocks = raw.strip().split("---")
    for block in blocks:
        block = block.strip()
        if not block:
            continue
        m = {}
        for line in block.split("\n"):
            line = line.strip()
            for key, label in [
                ("Model Name",        "**Model Name:**"),
                ("Provider",          "**Provider:**"),
                ("Parameters",        "**Parameters:**"),
                ("Description",       "**Description:**"),
                ("Key Features",      "**Key Features:**"),
                ("Tool Calling",      "**Tool Calling Support:**"),
                ("Cost Tier",         "**Cost Tier:**"),
                ("Suitability Score", "**Suitability Score:**"),
            ]:
                if line.startswith(label):
                    m[key] = line.replace(label, "").strip()
        if m.get("Model Name"):
            models.append(m)
    return models