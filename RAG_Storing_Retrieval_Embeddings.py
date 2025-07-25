# Please run the following in your terminal
# pip install streamlit langchain langchain-google-genai pypdf faiss-cpu python-dotenv qdrant-client nest_asyncio rank_bm25 langchain-community

import asyncio
import streamlit as st
import os
from dotenv import load_dotenv
import nest_asyncio 

# Apply nest_asyncio to allow nested event loops.
# This is crucial for Streamlit environments where async operations
# might conflict with the main event loop or previous asyncio.run() calls.
nest_asyncio.apply()

# LangChain components
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.retrievers import BM25Retriever, EnsembleRetriever

# Google AI specific imports
from langchain.chat_models import init_chat_model
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI

# Qdrant specific imports
from langchain_community.vectorstores import Qdrant
from qdrant_client import QdrantClient, models # Import QdrantClient and models

# Load environment variables from .env file (for API keys)
load_dotenv()

# --- Configuration ---
# Set your Google API key here or in your .env file
# os.environ["GOOGLE_API_KEY"] = "YOUR_GOOGLE_API_KEY" # Uncomment and replace if not using .env

# --- Streamlit UI Setup ---
st.set_page_config(page_title="RAG with LangChain, Qdrant & Semantic Search(Google AI)", layout="wide")
st.title("📄 RAG Pipeline with LangChain, Qdrant, BM25 & Semantic Search (Google AI)")
st.markdown("""
    Upload a document (PDF or TXT), ask questions, and get answers based on its content.
    This app uses LangChain for orchestration, and Qdrant & Semantic Search for efficient similarity search,
    now powered by Google's Generative AI models.
""")

# Check if Google API key is set
if not os.getenv("GOOGLE_API_KEY"):
    st.warning("Please set your Google API key in the sidebar or in your .env file.")
    st.sidebar.text_input("Enter your Google API Key", type="password", key="google_api_key_input")
    if st.session_state.get("google_api_key_input"):
        os.environ["GOOGLE_API_KEY"] = st.session_state["google_api_key_input"]
        st.experimental_rerun() # Rerun to apply the API key

api_key = os.getenv("GOOGLE_API_KEY")

model = init_chat_model("gemini-2.0-flash", model_provider="google_genai")

#model = genai.GenerativeModel(model_name)

# --- Functions for RAG Pipeline ---

@st.cache_data
def load_and_process_document(uploaded_files, chunk_size, chunk_overlap):
    """
    Loads a document, splits it into chunks, creates embeddings,
    and builds a Qdrant vector store. Caches the result for performance.
    """
    
    documents = []
    for uploaded_file in uploaded_files:        
        if uploaded_file.type == "application/pdf":
            # Save the uploaded PDF to a temporary file
            with open("temp_doc.pdf", "wb") as f:
                f.write(uploaded_file.getbuffer())
            loader = PyPDFLoader("temp_doc.pdf")
            documents.extend(loader.load())
        elif uploaded_file.type == "text/plain":
            # Save the uploaded TXT to a temporary file
            with open("temp_doc.txt", "wb") as f:
                f.write(uploaded_file.getbuffer())
            loader = TextLoader("temp_doc.txt")
            documents.extend(loader.load())
        else:
            st.error("Unsupported file type. Please upload a PDF or TXT file.")
            return None

    #Load documents
    
    st.success(f"Loaded {len(documents)} pages/sections from the document.")
    

    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,  # Specifies how length is calculated (e.g., character count)
        separators=["\n\n", "\n", " ", ""], # Common separators for text
        add_start_index=True,  # Adds the starting index of each chunk
    )
    
    # Split documents
    chunks = text_splitter.split_documents(documents)
    st.success(f"Split document into {len(chunks)} chunks.")
    return chunks

# @st.cache_resource
def initialize_vectorEmbeddings(chunks):
    # Create embeddings
    st.info("Generating embeddings and building Qdrant index... This may take a moment.")
    try:
        # Define an async helper function to perform the embedding and Qdrant creation
        async def _create_embeddings_and_vector_store(chunks_to_embed):
            embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

            # Initialize an in-memory Qdrant client
            # For persistent storage or remote Qdrant, you would configure the client differently.
            client = QdrantClient(":memory:") # Use in-memory client

            # Create a collection for the documents
            # The vector size must match the embedding model's output dimension (768 for embedding-001)
            # You might need to adjust this if you use a different embedding model.
            client.recreate_collection(
                collection_name="my_documents",
                vectors_config=models.VectorParams(size=768, distance=models.Distance.COSINE),
            )

            # Use Qdrant.from_documents to upload chunks and embeddings to Qdrant
            vector_store = Qdrant(
                client=client,
                collection_name="my_documents",
                embeddings=embeddings,
            )
            vector_store.add_documents(chunks_to_embed) # Add documents to the collection
            return vector_store

        # Run the async helper function using asyncio.run()
        vector_store = asyncio.run(_create_embeddings_and_vector_store(chunks))
        
        # Create BM25 Retriever
        # BM25Retriever.from_documents builds the BM25 index from the text chunks
        bm25_retriever = BM25Retriever.from_documents(chunks)
        # You can set k to control the number of documents retrieved by BM25
        bm25_retriever.k = 5 # Retrieve top 5 documents by BM25 score

        st.success("Qdrant vector store and BM25 retriever created successfully!")
        return vector_store, bm25_retriever # Return both retrievers
    except Exception as e:
        st.error(f"Error creating embeddings, Qdrant index, or BM25 retriever: {e}")
        st.error("Please ensure your Google API key is valid and you have an active subscription.")
        return None, None # Return None for both in case of error


# @st.cache_resource
def get_rag_response(vector_store, bm25_retriever, query):
    """
    Performs retrieval and generation using a hybrid (Qdrant + BM25) retriever
    and a Google AI LLM.
    """
    if vector_store is None or bm25_retriever is None:
        st.error("Retrievers not available. Please upload and process a document first.")
        return "Error: Document not processed."

    # Use ChatGoogleGenerativeAI for the LLM
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.7)

    # Create an EnsembleRetriever to combine Qdrant (semantic) and BM25 (keyword) retrievers
    # Weights determine the importance of each retriever (sum should be 1)
    ensemble_retriever = EnsembleRetriever(
        retrievers=[vector_store.as_retriever(), bm25_retriever],
        weights=[0.5, 0.5] # 50% semantic, 50% keyword search
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff", # "stuff" combines all retrieved docs into one prompt
        retriever=vector_store.as_retriever(),
        return_source_documents=True # To show which parts of the document were used
    )

    try:
        response = qa_chain({"query": query})
        return response
    except Exception as e:
        st.error(f"Error during RAG query: {e}")
        return "Error: Could not get a response from the LLM."


# @st.cache_resource
async def perform_semantic_search(vector_store, query, k=5):
    """
    Performs semantic search on the vector store and returns top-k relevant chunks.
    """
    if vector_store is None:
        return []  

    # Qdrant's similarity_search_with_score returns a list of (Document, score) tuples
    # The score is typically a distance metric (lower is better for cosine distance)
    try:
        results = await vector_store.asimilarity_search_with_score(query, k=k)
        return results
    except Exception as e:
        st.error(f"Error during semantic search: {e}")
        return []
    

# --- Main Application Logic ---
st.write('Multiple PDF Upload')

uploaded_files = st.file_uploader("Upload PDF files", accept_multiple_files=True, type="pdf")

chunk_size = st.slider("Chunk Size:", min_value=100, max_value=2000, value=500, step=50)
chunk_overlap = st.slider("Chunk Overlap:", min_value=0, max_value=500, value=50, step=10)

vector_store = None
bm25_retriever = None

if uploaded_files:
    st.write(f"Total Number of files: {len(uploaded_files)}")
    chunks = load_and_process_document(uploaded_files, chunk_size, chunk_overlap)
    vector_store, bm25_retriever = initialize_vectorEmbeddings(chunks)
else:
    st.info("Upload a document to start asking questions.")

# Only proceed if both retrievers are successfully initialized
if vector_store and bm25_retriever:
    st.markdown("---")
    st.header("Ask a Question (Hybrid RAG)")
    st.info("This RAG uses a combination of semantic (Qdrant) and keyword (BM25) search.")
    user_rag_query = st.text_area("Your question for RAG:", placeholder="E.g., What is the main topic of this document?", height=100, key="rag_query")

    if st.button("Get Answer (Hybrid RAG)"):
        if user_rag_query:
            with st.spinner("Getting your answer..."):
                # Pass both retrievers to the RAG response function
                rag_response = get_rag_response(vector_store, bm25_retriever, user_rag_query)

                if rag_response and "result" in rag_response:
                    st.subheader("Answer:")
                    st.write(rag_response["result"])

                    if "source_documents" in rag_response and rag_response["source_documents"]:
                        st.subheader("Sources:")
                        for i, doc in enumerate(rag_response["source_documents"]):
                            st.write(f"**Source {i+1}:**")
                            st.markdown(f"```\n{doc.page_content[:500]}...\n```") # Show first 500 chars
                            if hasattr(doc, 'metadata') and 'page' in doc.metadata:
                                st.write(f"Page: {doc.metadata['page'] + 1}")
                            st.markdown("---")
                else:
                    st.error("Could not retrieve an answer. Please try re-uploading the document or a different query.")
        else:
            st.warning("Please enter a question for RAG.")

    st.markdown("---")
    st.header("Perform Semantic Search")
    user_semantic_query = st.text_area("Your query for semantic search:", placeholder="E.g., Key concepts discussed.", height=100, key="semantic_query")
    num_results = st.slider("Number of results to retrieve:", min_value=1, max_value=10, value=3)

    if st.button("Search Documents"):
        if user_semantic_query:
            with st.spinner(f"Searching for top {num_results} relevant chunks..."):
                # Run the async semantic search function
                search_results = asyncio.run(perform_semantic_search(vector_store, user_semantic_query, k=num_results))

                if search_results:
                    st.subheader(f"Top {len(search_results)} Relevant Chunks:")
                    for i, (doc, score) in enumerate(search_results):
                        st.write(f"**Result {i+1} (Score: {score:.4f}):**")
                        st.markdown(f"```\n{doc.page_content[:1000]}...\n```")
                        if hasattr(doc, 'metadata') and 'page' in doc.metadata:
                            st.write(f"Page: {doc.metadata['page'] + 1}")
                        st.markdown("---")
                else:
                    st.info("No relevant chunks found or an error occurred during search.")
        else:
            st.warning("Please enter a query for semantic search.")

st.markdown("---")
st.caption("Built with LangChain, Qdrant+BM25, Semantic Search, and Streamlit.")
