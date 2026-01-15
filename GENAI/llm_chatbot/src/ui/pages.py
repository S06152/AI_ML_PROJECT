import streamlit as st
from config.settings import AppSettings
from services.chatbot_service import ChatbotService
from ui.sidebar import Sidebar
from utils.validators import InputValidator

class ChatbotApp:
    """Main Streamlit application class."""

    def __init__(self) -> None:
        self.sidebar = Sidebar()
        self.chatbot_service = ChatbotService()

    def run(self) -> None:
        st.set_page_config(page_title = AppSettings.APP_TITLE, page_icon = "🤖",layout = "wide")
        st.title("🤖 Enterprise AI Assistant")

        sidebar_config = self.sidebar.render()
        user_query = st.text_input("Ask your question")

        if not InputValidator.is_valid(user_query):
            st.info("Please enter a question.")
            return

        if not sidebar_config["api_key"]:
            st.warning("API key is required.")
            return

        with st.spinner("Generating response..."):
            response = self.chatbot_service.generate_response(
                query = user_query,
                api_key = sidebar_config["api_key"],
                model = sidebar_config["model"],
                temperature = sidebar_config["temperature"],
                max_tokens = sidebar_config["max_tokens"],
            )

        st.success("Response")
        st.write(response)
