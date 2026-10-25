import os
import tempfile
import streamlit as st
from langchain_community.document_loaders import (
    PyPDFLoader, 
    Docx2txtLoader, 
    UnstructuredExcelLoader
)
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 1. Catchy Page Config & Custom Neon Theme injection
st.set_page_config(page_title="Multi-Doc Wizard RAG", page_icon="✨", layout="wide")

st.markdown("""
    <style>
    /* Gradient Title Banner */
    .gradient-text {
        background: linear-gradient(135deg, #FF4B4B, #4A00E0);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3rem;
        font-weight: 800;
        margin-bottom: 0px;
    }
    .subtitle-text {
        color: #64748B;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    /* Vibrant Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #F8FAFC;
        border-right: 2px solid #E2E8F0;
    }
    /* Action Prompt Buttons/Pills */
    div.stButton > button {
        background: linear-gradient(135deg, #6366F1, #4F46E5);
        color: white !important;
        border: none;
        border-radius: 20px;
        padding: 0.5rem 1rem;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 6px -1px rgba(99, 102, 241, 0.2);
    }
    div.stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 15px -3px rgba(99, 102, 241, 0.4);
        background: linear-gradient(135deg, #4F46E5, #4338CA);
    }
    </style>
""", unsafe_allow_html=True)

# 2. Multi-Format Robust Core Logic (PDF, Word, Excel support)
def build_vectorstore(uploaded_file, size: int, overlap: int) -> FAISS:
    """Dynamically loads PDF, Word (.docx), or Excel (.xlsx) files safely, chunks, and embeds them."""
    file_extension = os.path.splitext(uploaded_file.name)[1].lower()
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as tmp:
        tmp.write(uploaded_file.getvalue())
        tmp_path = tmp.name

    try:
        # Route file loading based on extension type
        if file_extension == ".pdf":
            docs = PyPDFLoader(tmp_path).load()
        elif file_extension in [".docx", ".doc"]:
            docs = Docx2txtLoader(tmp_path).load()
        elif file_extension in [".xlsx", ".xls"]:
            docs = UnstructuredExcelLoader(tmp_path, mode="elements").load()
        else:
            raise ValueError(f"Unsupported file extension: {file_extension}")

        if not docs:
            raise ValueError("The uploaded document seems to have no extractable text content.")
            
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=size,
            chunk_overlap=overlap,
            separators=["\n\n", "\n", ".", " ", ""]
        )
        chunks = splitter.split_documents(docs)
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        return FAISS.from_documents(chunks, embeddings)
        
    except Exception as e:
        st.error(f"❌ Document Parsing Error: {str(e)}")
        return None
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

def get_chain(vectorstore, model_name: str, top_k: int):
    """Creates a fail-safe RAG chain with system context filtering."""
    retriever = vectorstore.as_retriever(search_kwargs={"k": top_k})
    llm = ChatOpenAI(model=model_name, temperature=0, streaming=True)
    
    template = """You are an enthusiastic, expert multi-document intelligence assistant.
    Answer the user's question clearly using only the provided context. If the context does not 
    contain the answer, politely explain that it isn't covered in the document. Do not invent facts.
    
    Context:
    {context}
    
    Question: {question}
    Answer:"""
    
    prompt = ChatPromptTemplate.from_template(template)
    
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
        
    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain, retriever

# 3. Dynamic Colorful Sidebar
with st.sidebar:
    st.markdown("<h2 style='color: #4A00E0; margin-bottom:0;'>🎨 Wizard Panel</h2>", unsafe_allow_html=True)
    st.write("Configure your environment settings below.")
    
    api_key = st.text_input("🔑 OpenAI API Key", type="password",
                            value=os.getenv("OPENAI_API_KEY", ""))
    
    st.divider()
    st.markdown("### 🧠 Brain Settings")
    model_name = st.selectbox("Engine Model", ["gpt-4o-mini", "gpt-4o", "gpt-4"])
    
    with st.expander("🛠️ Advanced Extraction Knobs"):
        chunk_size = st.slider("Chunk size", 500, 2000, 1000, step=100)
        chunk_overlap = st.slider("Chunk overlap", 0, 400, 150, step=50)
        top_k = st.slider("Retrieved chunks (k)", 2, 10, 4)
    
    st.divider()
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# API Key Validation Blocker
if not api_key:
    st.markdown("<p class='gradient-text'>✨ Multi-Doc Wizard RAG</p>", unsafe_allow_html=True)
    st.warning("🔒 Safe Mode Active: Please input your OpenAI API Key in the left sidebar menu to ignite the engine.")
    st.stop()

os.environ["OPENAI_API_KEY"] = api_key

# 4. App Layout Split
col_file, col_chat = st.columns([1.1, 2], gap="large")

with col_file:
    st.markdown("<h3 style='color: #4A00E0;'>📥 Drop Zone</h3>", unsafe_allow_html=True)
    st.caption("Supports PDF, Word (.docx), and Excel (.xlsx) files")
    
    uploaded_file = st.file_uploader(
        "Upload a document", 
        type=["pdf", "docx", "doc", "xlsx", "xls"], 
        label_visibility="collapsed"
    )
    
    if uploaded_file is not None:
        file_sig = (uploaded_file.name, uploaded_file.size, chunk_size, chunk_overlap)
        
        if st.session_state.get("file_sig") != file_sig:
            with st.spinner(f"🔮 Processing and indexing {uploaded_file.name}..."):
                v_store = build_vectorstore(uploaded_file, chunk_size, chunk_overlap)
                if v_store:
                    st.session_state.vectorstore = v_store
                    st.session_state.file_sig = file_sig
                    st.session_state.messages = []
                    st.toast(f"Parsed {uploaded_file.name} successfully!", icon="🚀")
        
        if "vectorstore" in st.session_state:
            st.markdown(f"""
                <div style='background-color: #E0F2FE; border-left: 5px solid #0EA5E9; padding: 15px; border-radius: 8px;'>
                    <span style='color: #0369A1; font-weight: bold;'>🟢 Connected Document:</span><br>
                    <code style='color: #0F172A;'>{uploaded_file.name}</code>
                </div>
            """, unsafe_allow_html=True)
    else:
        if "vectorstore" in st.session_state:
            del st.session_state.vectorstore
            del st.session_state.file_sig
            st.session_state.messages = []

with col_chat:
    st.markdown("<p class='gradient-text'>⚡ Wizard Chat</p>", unsafe_allow_html=True)
    st.markdown("<p class='subtitle-text'>Instant, hyper-accurate answers streamed straight from your PDF, Word, or Excel data</p>", unsafe_allow_html=True)
    
    if "vectorstore" in st.session_state:
        if "messages" not in st.session_state:
            st.session_state.messages = []
            
        # History replay logic
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                if "sources" in msg and msg["sources"]:
                    with st.expander("🔍 Inspect Source Chunks"):
                        st.code(msg["sources"], language="markdown")

        prompt_to_submit = None

        # Catchy Interactive Prompt Cards
        if len(st.session_state.messages) == 0:
            st.markdown("<p style='color: #6366F1; font-weight: 600;'>🚀 Quick Launch Questions:</p>", unsafe_allow_html=True)
            p_col1, p_col2 = st.columns(2)
            with p_col1:
                if st.button("✨ Summarize Document Structure", use_container_width=True):
                    prompt_to_submit = "Give me an intuitive executive summary broken into core themes."
            with p_col2:
                if st.button("📊 Extract Crucial Metrics/Data", use_container_width=True):
                    prompt_to_submit = "List out all key data points, figures, numbers, or performance metrics found."

        chat_input = st.chat_input("Ask a question about your uploaded document...")
        question = chat_input or prompt_to_submit

        if question:
            st.session_state.messages.append({"role": "user", "content": question})
            with st.chat_message("user"):
                st.markdown(question)
                
            chain, retriever = get_chain(st.session_state.vectorstore, model_name, top_k)
            
            with st.chat_message("assistant"):
                try:
                    source_docs = retriever.invoke(question)
                    source_text = ""
                    for i, doc in enumerate(source_docs):
                        page = doc.metadata.get("page", doc.metadata.get("row", "N/A"))
                        source_text += f"[Chunk {i+1} | Source Location: {page}]\n{doc.page_content.strip()}\n\n"
                except Exception:
                    source_text = ""

                # Real-time token streaming container
                answer = st.write_stream(chain.stream(question))
                
                if source_text:
                    with st.expander("🔍 Inspect Source Chunks"):
                        st.code(source_text, language="markdown")
            
            st.session_state.messages.append({
                "role": "assistant", 
                "content": answer,
                "sources": source_text
            })
            
            if prompt_to_submit:
                st.rerun()
    else:
        st.info("💡 Getting Started: Drag and drop your PDF, Word (.docx), or Excel (.xlsx) file into the Drop Zone on the left panel to unlock the chat canvas!")

