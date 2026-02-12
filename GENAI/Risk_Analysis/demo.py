import os
import tempfile
from typing import List

import streamlit as st
from langchain_core.documents import Document
from langchain_community.document_loaders import UnstructuredExcelLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.language_models.base import BaseLanguageModel
from langchain_groq import ChatGroq


# ============================================================
# 1. LOAD EXCEL
# ============================================================

def load_excel_documents(uploaded_file) -> List[Document]:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".xlsx") as temp_file:
        temp_file.write(uploaded_file.read())
        temp_path = temp_file.name

    try:
        loader = UnstructuredExcelLoader(temp_path)
        documents = loader.load()
    finally:
        os.remove(temp_path)

    return documents


# ============================================================
# 2. SPLIT DOCUMENTS
# ============================================================

def split_documents(documents: List[Document]) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150
    )
    return splitter.split_documents(documents)


# ============================================================
# 3. EMBEDDINGS
# ============================================================

def create_embeddings():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


# ============================================================
# 4. VECTOR STORE
# ============================================================

def create_vectorstore(documents, embeddings):
    return FAISS.from_documents(documents, embeddings)


# ============================================================
# 5. RETRIEVER
# ============================================================

def create_retriever(vectorstore):
    return vectorstore.as_retriever(
        search_type="mmr",   # better than similarity
        search_kwargs={"k": 5}
    )


# ============================================================
# 6. LLM
# ============================================================

def initialize_llm(model: str, api_key: str):
    if not api_key:
        raise ValueError("GROQ API Key is missing")

    return ChatGroq(
        model=model,
        api_key=api_key,
        temperature=0.2,
        max_tokens=800
    )


# ============================================================
# 7. PROMPT
# ============================================================

def create_prompt(user_instruction: str):

    system_prompt = """
User prompt:
{user_prompt}

Context:
{context}
"""

    return ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}")
        ]
    )


# ============================================================
# 8. FORMAT DOCS
# ============================================================

def format_docs(docs: List[Document]) -> str:
    return "\n\n".join(doc.page_content for doc in docs)


# ============================================================
# 9. RAG CHAIN
# ============================================================

def create_rag_chain(retriever, prompt, llm: BaseLanguageModel):

    return (
        {
            "context": retriever | RunnableLambda(format_docs),
            "input": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )


# ============================================================
# 10. STREAMLIT APP
# ============================================================

def streamlit_app():

    st.set_page_config(
        page_title="Project Risk Intelligence",
        page_icon="🔍",
        layout="wide"
    )

    st.title("🔍 Project Risk Intelligence Assistant")

    # ---------------- Sidebar ----------------
    st.sidebar.header("⚙️ Configuration")

    api_key = st.sidebar.text_input("🔑 Groq API Key", type="password")
    model = st.sidebar.selectbox(
        "🧠 Select Model",
        [
            "llama-3.1-8b-instant",
            "qwen/qwen3-32b",
            "groq/compound-mini",
            "openai/gpt-oss-120b"
        ]
    )

    user_prompt = st.sidebar.text_area(
        "📝 System Prompt",
        value="Enter the Prompt based on your project or requirement"
    )

    uploaded_file = st.sidebar.file_uploader(
        "📂 Upload Risk Register (.xlsx only)",
        type=["xlsx"]
    )

    if not api_key or not uploaded_file or not user_prompt:
        st.info("⬅️ Please complete configuration in sidebar.")
        st.stop()

    # ---------------- Process Document Once ----------------
    if "vectorstore" not in st.session_state:

        with st.spinner("Processing document..."):

            documents = load_excel_documents(uploaded_file)
            chunks = split_documents(documents)
            embeddings = create_embeddings()
            vectorstore = create_vectorstore(chunks, embeddings)

            st.session_state.vectorstore = vectorstore

        st.success("✅ Document processed successfully!")

    vectorstore = st.session_state.vectorstore
    retriever = create_retriever(vectorstore)

    # ---------------- LLM & RAG ----------------
    llm = initialize_llm(model, api_key)
    prompt = create_prompt(user_prompt)
    rag_chain = create_rag_chain(retriever, prompt, llm)

    # ---------------- Query ----------------
    user_query = st.text_input(
        "💬 Ask your Project Risk Question",
        placeholder="Example: Provide top three risks"
    )

    if user_query:

        with st.spinner("🔎 Analyzing risks..."):

            try:
                result = rag_chain.invoke(user_query)

                st.markdown("## 📊 Risk Analysis Result")
                st.write(result)

                # Optional: Show retrieved documents
                with st.expander("🔍 View Retrieved Context"):
                    docs = retriever.invoke(user_query)
                    for i, doc in enumerate(docs):
                        st.markdown(f"**Chunk {i+1}:**")
                        st.write(doc.page_content)

            except Exception as e:
                st.error(f"❌ Error: {repr(e)}")


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    streamlit_app()
