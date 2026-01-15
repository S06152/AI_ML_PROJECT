from config.env import EnvironmentConfig

class AppSettings:
    """Centralized application settings."""

    APP_TITLE = "Enterprise Q&A Chatbot with GROQ"

    DEFAULT_TEMPERATURE = 0.7
    DEFAULT_MAX_TOKENS = 150
    MIN_TOKENS = 50
    MAX_TOKENS = 300

    AVAILABLE_MODELS = [
        "qwen/qwen3-32b",
        "groq/compound-mini",
        "llama-3.1-8b-instant",
        "openai/gpt-oss-120b",
    ]

    GROQ_API_KEY = EnvironmentConfig.get("GROQ_API_KEY")
