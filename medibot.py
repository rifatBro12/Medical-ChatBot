import os
import streamlit as st
import time
from datetime import datetime

from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

DB_FAISS_PATH = "vectorstore/db_faiss"

# Custom CSS for better UI
st.markdown("""
<style>
    .main-header {
        text-align: center;
        color: #2E86AB;
        font-size: 2.5rem;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    
    .chat-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 20px;
        margin: 1rem 0;
    }
    
    .user-message {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem 1.5rem;
        border-radius: 20px 20px 5px 20px;
        margin: 0.5rem 0;
        margin-left: 20%;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        animation: slideInRight 0.3s ease-out;
    }
    
    .assistant-message {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        padding: 1rem 1.5rem;
        border-radius: 20px 20px 20px 5px;
        margin: 0.5rem 0;
        margin-right: 20%;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        animation: slideInLeft 0.3s ease-out;
    }
    
    .typing-indicator {
        display: flex;
        align-items: center;
        color: #666;
        font-style: italic;
        margin: 0.5rem 0;
    }
    
    .typing-dots {
        display: inline-block;
        margin-left: 0.5rem;
    }
    
    .typing-dots::after {
        content: '...';
        animation: typing 1.5s infinite;
    }
    
    @keyframes slideInRight {
        from { transform: translateX(100%); opacity: 0; }
        to { transform: translateX(0); opacity: 1; }
    }
    
    @keyframes slideInLeft {
        from { transform: translateX(-100%); opacity: 0; }
        to { transform: translateX(0); opacity: 1; }
    }
    
    @keyframes typing {
        0%, 20% { content: ''; }
        40% { content: '.'; }
        60% { content: '..'; }
        80%, 100% { content: '...'; }
    }
    
    .welcome-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 8px 16px rgba(0,0,0,0.1);
    }
    
    .stChatInput > div {
        border-radius: 25px;
        border: 2px solid #667eea;
    }
    
    .stChatInput > div:focus-within {
        border-color: #764ba2;
        box-shadow: 0 0 10px rgba(118, 75, 162, 0.3);
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def get_vectorstore():
    embedding_model = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
    return db


def set_custom_prompt(custom_prompt_template):
    prompt = PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])
    return prompt


def display_typing_indicator():
    """Display a typing indicator while processing"""
    typing_placeholder = st.empty()
    typing_placeholder.markdown("""
    <div class="typing-indicator">
        🤖 Medical AI is thinking<span class="typing-dots"></span>
    </div>
    """, unsafe_allow_html=True)
    return typing_placeholder

def display_message(role, content, is_typing=False):
    """Display a message with custom styling"""
    if role == 'user':
        st.markdown(f"""
        <div class="user-message">
            <strong>You:</strong><br>
            {content}
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="assistant-message">
            <strong>🤖 Medical AI:</strong><br>
            {content}
        </div>
        """, unsafe_allow_html=True)

def main():
    # Enhanced header with gradient
    st.markdown("""
    <div class="main-header">
        🏥 Medical AI Assistant
    </div>
    """, unsafe_allow_html=True)
    
    # Welcome card for first-time users
    if 'messages' not in st.session_state or len(st.session_state.messages) == 0:
        st.markdown("""
        <div class="welcome-card">
            <h3>👋 Welcome to Medical AI Assistant!</h3>
            <p>I'm here to help you with your medical questions.</p>
            <p>Ask me anything about health, symptoms, treatments, or medical conditions!</p>
        </div>
        """, unsafe_allow_html=True)

    # Initialize session state
    if 'messages' not in st.session_state:
        st.session_state.messages = []

    # Display chat history with custom styling
    for message in st.session_state.messages:
        display_message(message['role'], message['content'])

    # Enhanced chat input
    prompt = st.chat_input("💬 Ask me anything about health and medicine...", key="chat_input")

    if prompt:
        # Display user message immediately
        display_message('user', prompt)
        st.session_state.messages.append({'role': 'user', 'content': prompt})

        # Show typing indicator
        typing_placeholder = display_typing_indicator()
        
        # Simulate thinking time for better UX
        time.sleep(0.5)

        CUSTOM_PROMPT_TEMPLATE = """
        You are a helpful medical AI assistant. Use the provided context to answer the user's medical question accurately and professionally.
        
        Guidelines:
        - Provide clear, evidence-based information from the context
        - Be empathetic and supportive in your tone
        - If you don't know something, say so clearly
        - Always recommend consulting healthcare professionals for serious concerns
        - Keep responses concise but informative
        
        Context: {context}
        Question: {question}
        
        Provide a helpful medical response:
        """

        try:
            vectorstore = get_vectorstore()
            if vectorstore is None:
                st.error("Failed to load the medical knowledge base")

            qa_chain = RetrievalQA.from_chain_type(
                llm=ChatGroq(
                    groq_api_key=os.getenv("GROQ_API_KEY"),
                    model="llama-3.1-8b-instant",
                    temperature=0.1,  # Slightly higher for more natural responses
                ),
                chain_type="stuff",
                retriever=vectorstore.as_retriever(search_kwargs={'k': 4}),  # More context
                return_source_documents=True,
                chain_type_kwargs={'prompt': set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
            )

            # Get response
            response = qa_chain.invoke({'query': prompt})
            result = response["result"]
            
            # Clear typing indicator
            typing_placeholder.empty()
            
            # Add a small delay for natural feel
            time.sleep(0.3)
            
            # Display assistant response
            display_message('assistant', result)
            st.session_state.messages.append({'role': 'assistant', 'content': result})

        except Exception as e:
            typing_placeholder.empty()
            st.error(f"❌ Sorry, I encountered an error: {str(e)}")
            st.info("💡 Please try again or rephrase your question.")


if __name__ == "__main__":
    main()
