import streamlit as st
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os

from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_classic.chains.question_answering import load_qa_chain
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
# Load environment variables from .env file (used for API key)
load_dotenv()
# Configure Google Gemini API using API key
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

#Reading Pdf
def get_pdf_text(pdf_docs):
    text=""
    for pdf in pdf_docs:
        # Read each uploaded PDF file
        pdf_reader=PdfReader(pdf)
        for page in pdf_reader.pages:
            # Extract text from each page and append
            text+=page.extract_text() or ""
    return text

#Divide text into chunks
def get_text_chunks(text):
    # Split large text into smaller chunks for better processing
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks=text_splitter.split_text(text)
    return chunks

#Convert into VectorDB
def get_vector_store(text_chunks):
    # Convert text chunks into embeddings (numerical vectors)
    embeddings=GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    # Store embeddings in FAISS (vector database)
    vector_store=FAISS.from_texts(text_chunks, embedding=embeddings)
      # Save FAISS index locally for later use
    vector_store.save_local("faiss_index")

#QA CHAIN
def get_conversational_chain():
    # Custom prompt to guide the AI response
    prompt_template = """
Answer the question as detailed as possible from the provided context, make sure to provide all the details. If the answer is not in the provided context just say, "answer is not available in the context", don't provide the wrong answer.
Context:\n{context}\n
Question: \n{question}\n

Answer:
"""
    # Load Gemini chat model for answering questions
    model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.3)
    # Create prompt template with variables
    prompt = PromptTemplate(template=prompt_template, input_variables=["context","question"])
    # Load QA chain (combines docs + question + model)
    chain=load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

#USER INPUT
def user_input(user_question):
    # Check if FAISS index exists (PDF must be processed first)
    if not os.path.exists("faiss_index"):
        st.warning("Please upload and process the PDF files first.")
        return
    # Load embeddings model
    embeddings=GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    # Load saved FAISS vector database
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    # Find relevant chunks similar to user question
    docs = new_db.similarity_search(user_question)
    # Load QA chain
    chain = get_conversational_chain()
    # Pass relevant docs + question to model
    response = chain(
        {"input_documents": docs, "question": user_question},
        return_only_outputs=True
    )
    # Print response in terminal
    print(response)
    # Display response in Streamlit UI
    st.write("Reply: ", response["output_text"])
#MAIN APP
def main():
    # Set page title/config
    st.set_page_config("Chat With Multiple PDF")
    # App heading
    st.header("Chat with Multiple PDF using Gemini🧑‍💻")
    # Input box for user question
    user_question = st.text_input("Ask a Question from the PDF Files")
    # If user enters question → process it
    if user_question:
        user_input(user_question)
    # Sidebar for file upload
    with st.sidebar:
        st.title("Menu:")
        # Upload multiple PDF files
        pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)
        # Button to process PDFs
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                # Step 1: Extract text from PDFs
                raw_text = get_pdf_text(pdf_docs)
                # Step 2: Split text into chunks
                text_chunks = get_text_chunks(raw_text)
                # Step 3: Create vector database
                get_vector_store(text_chunks)
                st.success("Done")
# Entry point of the program
if __name__== "__main__":
    main()