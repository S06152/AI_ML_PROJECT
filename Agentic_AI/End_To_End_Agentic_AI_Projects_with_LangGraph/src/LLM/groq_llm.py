from langchain_groq import ChatGroq
import os
import streamlit as st

class GROQLLM:
    def __init__(self, user_contols_input):
        self.user_contols_input = user_contols_input
    
    def get_llm_model(self):
        try:
            groq_api_key = self.user_contols_input["GROQ_API_KEY"]
            selected_groq_model = self.user_contols_input["selected_groq_model"]
            if groq_api_key == "" and os.environ["GROQ_API_KEY"] == "":
                st.error("Please Enter the Groq API KEY")
            
            llm = ChatGroq(api_key = groq_api_key, model = selected_groq_model)

        except Exception as e:
            raise ValueError(f"Error Ocuured With Exception : {e}")    

        return llm


