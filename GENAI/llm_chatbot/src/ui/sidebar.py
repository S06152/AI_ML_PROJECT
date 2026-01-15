import streamlit as st
from src.config.settings import AppSettings

class Sidebar:
    """Sidebar UI component."""

    def render(self) -> dict:
        st.sidebar.header("⚙️ Configuration")

        api_key = st.sidebar.text_input("🔑 Groq API Key", type = "password", value = AppSettings.GROQ_API_KEY or "")
        model = st.sidebar.selectbox("🧠 LLM Model", AppSettings.AVAILABLE_MODELS)
        temperature = st.sidebar.slider("🔥 Temperature", 0.0, 1.0, AppSettings.DEFAULT_TEMPERATURE)
        max_tokens = st.sidebar.slider("📏 Max Tokens", AppSettings.MIN_TOKENS, AppSettings.MAX_TOKENS, AppSettings.DEFAULT_MAX_TOKENS)

        st.sidebar.markdown("---")
        st.sidebar.caption("🚀 Enterprise AI Chatbot")

        return {
            "api_key": api_key,
            "model": model,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
