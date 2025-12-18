"""Configuration for MCP Tool Categorization."""

import os
from pathlib import Path

# Load .env file if exists
try:
    from dotenv import load_dotenv
    env_path = Path(__file__).parent.parent / ".env"
    if env_path.exists():
        load_dotenv(env_path)
except ImportError:
    pass

# Paths
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR
RESULTS_DIR = BASE_DIR / "results"
INPUT_FILE = DATA_DIR / "tool_schemas_deduped_by_desc.json"

# API Keys (from environment variables)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
UPSTAGE_API_KEY = os.getenv("UPSTAGE_API_KEY")

# LLM Configuration
LLM_CONFIG = {
    "openai": {
        "model": "gpt-4o",
        "temperature": 0,
        "max_tokens": 1000,
    },
    "anthropic": {
        "model": "claude-opus-4-5-20251101",
        "temperature": 0,
        "max_tokens": 1000,
    },
    "google": {
        "model": "gemini-3-pro-preview",
        "temperature": 0,
        "max_tokens": 1000,
    },
    "upstage": {
        "model": "solar-pro2",
        "temperature": 0,
        "max_tokens": 1000,
    },
}

# Batch size for LLM classification
BATCH_SIZE = 20

# Default provider
DEFAULT_PROVIDER = "openai"
