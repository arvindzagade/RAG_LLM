import os
import streamlit as st
import numpy as np
import faiss
import fitz  # PyMuPDF for PDF processing
import pickle  # To save embeddings
from langchain_groq import ChatGroq
from langchain.embeddings import HuggingFaceEmbeddings
from dotenv import load_dotenv

# Load API keys
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Initialize LLM
groq_llm = ChatGroq(api_key=GROQ_API_KEY)

# Paths
DATA_FOLDER = "data"
INDEX_PATH = "faiss_index.bin"  # Changed to binary for FAISS
EMBEDDINGS_PATH = "embeddings.pkl"

# Create folders if they don't exist
os.makedirs(DATA_FOLDER, exist_ok=True)

# Streamlit Page Config
st.set_page_config(page_title="📚 AI Research Assistant", page_icon="🔍", layout="wide")

# Sidebar - Upload PDFs
st.sidebar.markdown("## 📂 Upload Research Papers")
uploaded_files = st.sidebar.file_uploader("Upload PDFs", type=["pdf"], accept_multiple_files=True)

# **Function: Process PDFs & Save Embeddings**
def process_pdfs(files, chunk_size=500):
    """Save PDFs, extract text, and return chunked documents."""
    documents = []
    for file in files:
        file_path = os.path.join(DATA_FOLDER, file.name)
        if not os.path.exists(file_path):  # **Avoid reprocessing**
            with open(file_path, "wb") as f:
                f.write(file.read())

        with fitz.open(file_path) as doc:
            text = "\n".join([page.get_text("text") for page in doc])
            for i in range(0, len(text), chunk_size):
                documents.append(text[i:i+chunk_size])
    
    return documents

# **Load or Generate FAISS Index**
def load_or_generate_faiss():
    """Load existing FAISS index or generate a new one."""
    if os.path.exists(INDEX_PATH) and os.path.exists(EMBEDDINGS_PATH):
        st.session_state.faiss_index = faiss.read_index(INDEX_PATH)  # ✅ FAISS Binary Loading
        with open(EMBEDDINGS_PATH, "rb") as f:
            embeddings = pickle.load(f)
        st.session_state.documents = process_pdfs([])
        return True
    return False

# **Check if PDFs already exist and process only new ones**
if uploaded_files:
    new_docs = process_pdfs(uploaded_files)
    
    if new_docs:  # **Process only new files**
        with st.spinner("⏳ Processing PDFs..."):
            st.session_state.documents = new_docs
            embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
            embeddings = np.array([embedding_model.embed_query(doc) for doc in st.session_state.documents])

            # Store embeddings in FAISS index
            d = embeddings.shape[1]
            index = faiss.IndexFlatL2(d)
            index.add(embeddings)
            st.session_state.faiss_index = index

            # ✅ Save FAISS index & embeddings
            faiss.write_index(index, INDEX_PATH)
            with open(EMBEDDINGS_PATH, "wb") as f:
                pickle.dump(embeddings, f)

        st.sidebar.success(f"✅ {len(new_docs)} new file(s) uploaded & processed!")

# **Load existing index if available**
if "faiss_index" not in st.session_state:
    load_or_generate_faiss()

# **Chat Section**
st.markdown("## 🤖 AI Research Chat")
st.markdown("### Ask questions related to your uploaded research papers.")

user_query = st.text_input("🔍 Ask a research-related question:")

# **Function: Query RAG System**
def query_rag_system(user_query):
    """Retrieve relevant research excerpts and generate AI response."""
    if "documents" not in st.session_state or not st.session_state.documents:
        return "⚠️ No research papers found. Please upload PDFs first!", None
    
    # Get query embedding
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    query_embedding = embedding_model.embed_query(user_query)
    D, I = st.session_state.faiss_index.search(np.array([query_embedding]), k=3)

    # Retrieve similar research excerpts
    similar_docs = [st.session_state.documents[i] for i in I[0]]

    # Construct LLM prompt
    prompt = f"Based on these research excerpts, answer the query: {user_query}\n\n"
    for i, doc in enumerate(similar_docs):
        prompt += f"Excerpt {i+1}: {doc[:500]}...\n\n"
    prompt += "Provide a concise, structured answer."

    # Get response from LLM
    response = groq_llm.predict(prompt)

    return response, similar_docs

# **Handle Query Submission**
if st.button("🧠 Generate Answer"):
    if user_query:
        with st.spinner("⏳ Thinking..."):
            response, context = query_rag_system(user_query)

            # Display Chat Messages
            st.markdown(f"**User Query:** {user_query}")
            st.markdown(f"**AI Response:** {response}")

            if context:
                st.subheader("📖 Research Excerpts Used:")
                for i, excerpt in enumerate(context):
                    st.markdown(f"**Excerpt {i+1}:** `{excerpt[:400]}...`")
    else:
        st.warning("⚠️ Please enter a query.")

# Footer
st.markdown("---")
st.markdown('<p style="text-align:center;">Built with ❤️ using Streamlit & Groq API</p>', unsafe_allow_html=True)
