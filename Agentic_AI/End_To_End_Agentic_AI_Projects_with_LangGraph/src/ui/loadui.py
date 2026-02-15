import streamlit as st
from .uiconfigfile import Config

class LoadStreamlitUI:
    """
    This class handles:
    - Loading UI configuration from Config class
    - Rendering Streamlit components
    - Capturing user selections
    """

    def __init__(self):
        """
        Constructor:
        - Initializes configuration reader
        - Creates dictionary to store user inputs
        """
        self.config = Config()
        self.user_controls = {}
    
    def load_streamlit_ui(self):
        """
        Builds and renders the Streamlit UI.

        Returns:
            dict: Dictionary containing user-selected values
        """

        # Set page configuration (title + layout)
        page_title = "🤖 " + self.config.get_page_title()
        st.set_page_config(page_title = page_title, layout = "wide")

        # Display main header
        st.header(page_title)

        # Sidebar section
        with st.sidebar:
            # Retrieve options from config file
            llm_options = self.config.get_llm_options()
            usecase_options = self.config.get_usecase_options()

            # LLM selection
            selected_llm = st.selectbox("Select LLM", llm_options)
            self.user_controls["selected_llm"] = selected_llm

            # GROQ-specific settings
            if selected_llm == "GROQ":

                # Model selection
                model_options = self.config.get_groq_model_options()
                selected_model = st.selectbox("Select Model", model_options)
                self.user_controls["selected_groq_model"] = selected_model

                # API Key input (stored in session state safely)
                groq_api_key = st.text_input("🔑 Groq API Key:", type = "password", key = "GROQ_API_KEY")
                self.user_controls["GROQ_API_KEY"] = groq_api_key
                
                # Validate API key
                if not groq_api_key:
                    st.warning("⚠️ Please enter your GROQ API key to proceed. Don't have? refer : https://console.groq.com/keys ")

            # USe case selection
            selected_usecase = st.selectbox("Select Use Case", usecase_options)
            self.user_controls["selected_usecase"] = selected_usecase

        return self.user_controls