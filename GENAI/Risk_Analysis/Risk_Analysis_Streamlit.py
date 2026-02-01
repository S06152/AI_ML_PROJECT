import os
import streamlit as st
from typing import List
from langchain_core.documents import Document
from langchain_community.document_loaders import UnstructuredExcelLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.language_models.base import BaseLanguageModel
from langchain_groq import ChatGroq

# -------------------------------------------------------------------
# 1. LOAD EXCEL AS UNSTRUCTURED DOCUMENTS
# -------------------------------------------------------------------

def load_excel_documents(file: str) -> List[Document]:

    """
    Load documents from an Excel file using UnstructuredExcelLoader.
    
    Parameters:
    -----------
    file : str
        Excel file to be loaded
    mode : str, optional
        Loading mode for the document loader (default is "elements")
    
    Returns:
    --------
    List[Document]
        List of LangChain Document objects
    """

    loader = UnstructuredExcelLoader(file)
    documents = loader.load()

    return documents

# -------------------------------------------------------------------
# 2. SPLIT DOCUMENTS
# -------------------------------------------------------------------

def split_documents_into_chunks(documents: List[Document], chunk_size: int = 400, chunk_overlap: int = 100) -> List[Document]:

    """
    Split documents into smaller chunks using RecursiveCharacterTextSplitter.
    
    Parameters:
    -----------
    documents : List[Document]
        List of LangChain Document objects to be split
    chunk_size : int, optional
        Maximum size of each chunk in characters (default is 400)
    chunk_overlap : int, optional
        Number of characters to overlap between chunks (default is 100)
    
    Returns:
    --------
    List[Document]
        List of chunked Document objects
    """
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = chunk_size, chunk_overlap = chunk_overlap)
    chunks = text_splitter.split_documents(documents)
    
    return chunks

# -------------------------------------------------------------------
# 3. EMBEDDINGS
# -------------------------------------------------------------------

def create_embeddings(model_name: str = "all-MiniLM-L6-v2") -> HuggingFaceEmbeddings:

    """
    Create a HuggingFace embeddings model.
    
    Parameters:
    -----------
    model_name : str, optional
        Name of the HuggingFace model to use for embeddings 
        (default is "all-MiniLM-L6-v2")
    
    Returns:
    --------
    HuggingFaceEmbeddings
        Initialized HuggingFace embeddings model
    """
    embeddings = HuggingFaceEmbeddings(model_name = model_name)
    
    return embeddings

# -------------------------------------------------------------------
# 4. VECTOR STORE
# -------------------------------------------------------------------

def create_faiss_vectorstore(documents: List[Document], embeddings: HuggingFaceEmbeddings) -> FAISS:

    """
    Create a FAISS vector database from documents and embeddings.
    
    Parameters:
    -----------
    documents : List[Document]
        List of Document objects to be stored in the vector database
    embeddings : HuggingFaceEmbeddings
        Embeddings model to use for vectorizing the documents
    
    Returns:
    --------
    FAISS
        FAISS vector store containing the embedded documents
    """
    vectorstore = FAISS.from_documents(documents, embeddings)
    
    return vectorstore

# -------------------------------------------------------------------
# 5. RETRIEVER
# -------------------------------------------------------------------

def create_retriever(vectorstore: FAISS, search_type: str = "similarity", k: int = 3) -> VectorStoreRetriever:

    """
    Create a retriever from a FAISS vector store.
    
    Parameters:
    -----------
    vectorstore : FAISS
        FAISS vector store to create retriever from
    search_type : str, optional
        Type of search to perform (default is "similarity")
        Options: "similarity", "mmr", "similarity_score_threshold"
    search_kwargs : dict, optional
        Additional search parameters (e.g., {'k': 4} for top 4 results)
    
    Returns:
    --------
    VectorStoreRetriever
        Retriever object for querying the vector store
    """
    
    retriever = vectorstore.as_retriever(search_type = search_type, search_kwargs = {"k" : k})
    
    return retriever

# -------------------------------------------------------------------
# 6. LLM INITIALIZATION
# -------------------------------------------------------------------

def initialize_llm(model_name: str, api_key: str , temperature: float = 0.3, max_tokens: int = 800) -> ChatGroq:

    """
    Initialize a ChatGroq LLM instance.
    
    Parameters:
    -----------
    model_name : str
        Name of the model to use 
    api_key : str
        Groq API key. 
    temperature : float,
        Sampling temperature (default is 0.0)
    max_tokens : int
        Maximum tokens in response (default is None)
    
    Returns:
    --------
    ChatGroq
        Initialized ChatGroq LLM instance
    """

    # Get API key from parameter or environment
    if not api_key:
        raise ValueError("GROQ_API_KEY is missing.")
                
    llm = ChatGroq(model_name = model_name, api_key = api_key, temperature = temperature, max_tokens = max_tokens)
    
    return llm

# -------------------------------------------------------------------
# 7. RAG PROMPT
# -------------------------------------------------------------------

'''
def create_rag_prompt( ) -> ChatPromptTemplate:

    system_prompt = """
    You are an expert Project Manager performing a management risk analysis.

    Your task:
        - Identify risks present in the provided project data
        - List the TOP THREE management risks only
        - Rank them from highest to lowest impact

    Rules:
        - Use ONLY the information in the provided context
        - Do NOT infer or assume anything
        - Do NOT use external knowledge
        - If risk-related information is not available, respond exactly with:
        "I cannot provide assistance on that item as the information is not available in the context."

    Context: 
    {context}
    """

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}")
        ]
    )
    
    return prompt
'''

def create_rag_prompt(user_prompt: str) -> ChatPromptTemplate:
    system_prompt = f"""
    {user_prompt}

    Context: 
    {{context}}
    """

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}")
        ]
    )

    return prompt

# -------------------------------------------------------------------
# 8. DOCUMENT FORMATTER
# -------------------------------------------------------------------

def format_docs(docs: List[Document]) -> str:

    """
    Format a list of documents into a single string.
    
    Parameters:
    -----------
    docs : List[Document]
        List of Document objects to format
    
    Returns:
    --------
    str
        Concatenated string of all document contents separated by double newlines
    """
    
    return "\n\n".join(doc.page_content for doc in docs)

# -------------------------------------------------------------------
# 9. RAG CHAIN
# -------------------------------------------------------------------

def create_rag_chain(retriever: VectorStoreRetriever, prompt: ChatPromptTemplate, llm: BaseLanguageModel):

    """
    Create a complete RAG (Retrieval-Augmented Generation) chain.
    
    Parameters:
    -----------
    retriever : VectorStoreRetriever
        Retriever for fetching relevant documents
    prompt : ChatPromptTemplate
        Prompt template for the LLM
    llm : BaseLanguageModel
        Language model for generating responses
    
    Returns:
    --------
    Runnable
        Complete RAG chain ready for invocation
    """
    rag_chain = (
        {
            "context": retriever | RunnableLambda(format_docs),
            "input": RunnablePassthrough()
        }
        | prompt
        | llm
    )
    
    return rag_chain

# -------------------------------------------------------------------
# 10. QUERY FUNCTION
# -------------------------------------------------------------------

def query_rag_chain( rag_chain, query: str) -> str:
    
    """
    Query the RAG chain and return the response content.
    
    Parameters:
    -----------
    rag_chain : Runnable
        The RAG chain to query
    query : str
        The question or prompt to send to the RAG chain
    
    Returns:
    --------
    str
        The content of the response from the RAG chain
    """
   
    response = rag_chain.invoke(query)
    
    return response.content

# -------------------------------------------------------------------
# 11. Streamlit UI
# -------------------------------------------------------------------
def streamlit_app():
    st.set_page_config(page_title = "Project Risk Analysis Assist", page_icon = "🔍", layout = "wide")
    st.title("🔍 Project Risk Analysis Assist")

    st.sidebar.header("⚙️ Configuration")
    api_key = st.sidebar.text_input("🔑 Groq API Key", type = "password")
    model = st.sidebar.selectbox("🧠 LLM Model", ["qwen/qwen3-32b", "groq/compound-mini", "llama-3.1-8b-instant", "openai/gpt-oss-120b"])
    temperature = st.sidebar.slider("🔥 Temperature", min_value = 0.0, max_value = 1.0, value = 0.7)
    max_tokens = st.sidebar.slider("📏 Max Tokens", min_value = 50, max_value = 300, value = 150)
    user_query = st.text_input("Ask your Risk Analysis query")

    user_prompt = st.text_input("📝 Provide your custom prompt message", help="Enter the instructions for the AI to follow during risk analysis.")

    if not user_prompt:
        st.warning("Custom prompt message is required")
        st.stop()

    if not api_key:
        st.warning("API Key is required")
        st.stop()
    
    with st.spinner("📄 Searching..."):
        # Load & prepare documents
        documents = load_excel_documents("Sample_Airbag_ECU_SW_Development_Plan.xlsx")
        chunks = split_documents_into_chunks(documents)

        # Vector store
        embeddings = create_embeddings()
        vectorstore = create_faiss_vectorstore(chunks, embeddings)
        retriever = create_retriever(vectorstore)

        # Initialize GROQ LLM Model
        llm = initialize_llm(model = model, api_key = api_key, temperature = temperature, max_tokens = max_tokens)

        # RAG
        prompt = create_rag_prompt(user_prompt)
        rag_chain = create_rag_chain(retriever, prompt, llm)

    # ---------------- Query ----------------   
    st.subheader("🧠 Risk Analysis")
    query = st.text_input("Enter your query")
    if st.button("🚀 Analyze"):
        with st.spinner("🔍 Analyzing risks..."):
            result = query_rag_chain(rag_chain, query)
        st.subheader("📊 Management Risk Analysis")
        st.markdown(result)

# -------------------------------------------------------------------
# 12. MAIN
# -------------------------------------------------------------------

if __name__ == "__main__":
    streamlit_app() 