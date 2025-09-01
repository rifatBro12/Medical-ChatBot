# Medical-ChatBot

A Retrieval-Augmented Generation (RAG) chatbot that answers medical queries using PDF documents as context. The system processes PDFs, creates embeddings, stores them in a FAISS vector store, and integrates with a Grok-powered language model via LangChain for question answering. A Streamlit interface provides an interactive user experience.
Features

PDF Processing: Extracts text from PDF files in a specified directory.
Text Chunking: Splits documents into manageable chunks for efficient processing.
Vector Embeddings: Uses HuggingFace's sentence-transformers/all-MiniLM-L6-v2 for creating embeddings.
Vector Store: Stores embeddings in a FAISS database for fast similarity search.
LLM Integration: Leverages Grok's llama-3.1-8b-instant model via LangChain for accurate responses.
Streamlit UI: Provides a user-friendly chat interface for querying medical information.

Prerequisites

Python 3.8+
A Groq API key (set in a .env file as GROQ_API_KEY)
PDF files stored in a data/ directory
Required Python packages (listed in requirements.txt)

Setup Instructions

Clone the Repository:
git clone <repository-url>
cd <repository-directory>


Install Dependencies:Create a virtual environment and install the required packages:
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt


Set Up Environment Variables:Create a .env file in the project root and add your Groq API key:
GROQ_API_KEY=your_groq_api_key_here


Prepare PDF Data:Place your PDF files in the data/ directory. These files will be processed to create the knowledge base.

Run the Pipeline:

Step 1: Process PDFs and Create Vector Store:Run the script to process PDFs, chunk text, generate embeddings, and save them to a FAISS index:
python process_pdfs.py

Ensure the data/ directory exists and contains PDF files. The FAISS index will be saved to vectorstore/db_faiss.

Step 2: Run the Streamlit App:Launch the Streamlit interface to interact with the chatbot:
streamlit run app.py





Project Structure

data/: Directory for input PDF files.
vectorstore/db_faiss/: Directory for the FAISS vector store.
process_pdfs.py: Script to process PDFs, chunk text, and create the FAISS index.
app.py: Streamlit application for the chatbot interface.
.env: File for storing environment variables (e.g., GROQ_API_KEY).
requirements.txt: List of Python dependencies.

Usage

Run process_pdfs.py to generate the FAISS vector store from your PDFs.
Launch the Streamlit app using streamlit run app.py.
Open the provided URL in your browser (typically http://localhost:8501).
Enter a medical query in the chat input, and the chatbot will respond based on the PDF content.

Example Query
User Input: "What are the symptoms of diabetes?"Chatbot Response: (Depends on the PDF content, e.g., "Common symptoms of diabetes include increased thirst, frequent urination, and fatigue.")
Dependencies
Install the required packages using:
pip install langchain langchain-community langchain-groq langchain-huggingface faiss-cpu streamlit python-dotenv pypdf

Notes

Ensure the data/ directory contains valid PDF files before running process_pdfs.py.
The FAISS vector store is saved locally in vectorstore/db_faiss and loaded by the Streamlit app.
The chatbot uses the llama-3.1-8b-instant model from Groq, which is free but requires an API key.
For better performance, adjust chunk_size and chunk_overlap in process_pdfs.py based on your PDF content.

Troubleshooting

Error: GROQ_API_KEY not found: Ensure the .env file contains a valid GROQ_API_KEY.
No PDFs found: Verify that the data/ directory exists and contains PDF files.
Streamlit errors: Check that all dependencies are installed and compatible with your Python version.

Contributing
Contributions are welcome! Please submit a pull request or open an issue for suggestions or bug reports.
License
This project is licensed under the MIT License. See the LICENSE file for details.
