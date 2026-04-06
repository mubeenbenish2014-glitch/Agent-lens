import os
from dotenv import load_dotenv

load_dotenv()

# ── OpenAI Settings ────────────────────────────────────────────
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL   = "gpt-4o"

# ── Ollama Settings ────────────────────────────────────────────
OLLAMA_BASE_URL    = "http://localhost:11434/v1"
OLLAMA_DEFAULT_MODEL = "llama3.2"

# ── App Settings ───────────────────────────────────────────────
APP_TITLE   = "AgentLens"
APP_ICON    = "🔍"
MAX_RESULTS = 5