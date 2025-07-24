import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from PyPDF2 import PdfReader # Or from pdfplumber import open as pdfplumber_open
from sentence_transformers import SentenceTransformer
import os
import time

st.title("PDF Reader App")

uploaded_file = st.file_uploader("Upload your PDF file", type=["pdf"])

text_content = ""

if uploaded_file is not None:
    try:
        # Using PyPDF2
        pdf_reader = PdfReader(uploaded_file)
        for page in pdf_reader.pages:
            text_content += page.extract_text() + "\n" # Add newline for readability between pages

    except Exception as e:
        st.error(f"Error reading PDF: {e}")

st.title("Text Splitter with RecursiveCharacterTextSplitter")
st.write("Upload a text file or paste text to split it into chunks.")

# User input for text
if text_content is not None:
    text_input = text_content
else:
    text_input = st.text_area("Please upload pdf documents")

# User input for chunk size and overlap
chunk_size = st.slider("Chunk Size:", min_value=100, max_value=2000, value=500, step=50)
chunk_overlap = st.slider("Chunk Overlap:", min_value=0, max_value=500, value=50, step=10)
chunks = ""
#button_container - st.container()

''' The above code, once called, will load each document, and
    finally we will use RecursiveCharacterTextSplitter to split the document so that indexing can be done later.'''

placeholder = st.empty()
if st.button("Clear Chunks"):
    placeholder = st.empty()
if st.button("Split Text"):
    if text_input:
        # Initialize RecursiveCharacterTextSplitter
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,  # Specifies how length is calculated (e.g., character count)
            separators=["\n\n", "\n", " ", ""], # Common separators for text
            add_start_index=True  # Adds the starting index of each chunk
        )

        # Split the text
        chunks = text_splitter.split_text(text_input)

        with placeholder.container():
            st.subheader("Split Chunks:")
            for i, chunk in enumerate(chunks[:20]):
                st.write(f"**Chunk {i+1}:**")
                st.code(chunk)
                st.write(f"Length: {len(chunk)} characters")
                #if text_splitter.add_start_index:
                #    st.write(f"Start Index: {chunk.metadata['start_index']}")
    else:
        st.warning("Please load a pdf file.")
        

st.title("Embeddings")
user_input = st.text_input("Enter your phrase:", "")
if st.button("Submit"):
    try:
        model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        sentences = chunks
        embeddings = model.encode(sentences)

        similarities = model.similarity(embeddings, embeddings)
        st.write(embeddings)
    except Exception as e:
        st.write(f"An error occurred: {e}")
