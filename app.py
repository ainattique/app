import os
import tempfile
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 1. Page Configuration & Custom Styling
st.set_page_config(page_title="PDF RAG Agent", page_icon="📄", layout="wide")

# Custom CSS for a cleaner, modern chat interface look
st.markdown("""
    <style>
    .block-container { padding-top: 1.5rem; }
    .stChatInputContainer { padding-bottom: 1rem; }
    div[data-testid="stExpander"] { border: none; box-shadow: none; }
    </style>
""", unsafe_allow_html=True)

# 2. Helper Functions
def build_vectorstore(pdf_bytes: bytes, size: int, overlap: int) -> FAISS:
    """Load PDF, split into chunks, embed, and store in FAISS."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(pdf_bytes)
        tmp_path = tmp.name

    try:
        docs = PyPDFLoader(tmp_path).load()
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=size,
            chunk_overlap=overlap,
            separators=["\n\n", "\n", ".", " ", ""]
        )
        chunks = splitter.split_documents(docs)
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        return FAISS.from_documents(chunks, embeddings)
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

def get_chain(vectorstore, model_name: str, top_k: int):
    """Creates the RAG chain and retriever."""
    retriever = vectorstore.as_retriever(search_kwargs={"k": top_k})
    llm = ChatOpenAI(model=model_name, temperature=0, streaming=True) # Enabled streaming
    
    template = """You are an assistant for question-answering tasks. 
    Use the following pieces of retrieved context to answer the question. 
    If you don't know the answer, just say that you don't know, don't try to make up an answer.
    
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

# 3. Sidebar Layout
with st.sidebar:
    st.title("⚙️ Configuration")
    
    api_key = st.text_input("OpenAI API Key", type="password",
                            value=os.getenv("OPENAI_API_KEY", ""))
    
    st.divider()
    st.subheader("Model Settings")
    model_name = st.selectbox("Chat model", ["gpt-4o-mini", "gpt-4o", "gpt-4"])
    
    # IMPROVEMENT: Hidden advanced parameters under an expander for a cleaner UI
    with st.expander("🛠️ Advanced Developer Settings"):
        chunk_size = st.slider("Chunk size", 500, 2000, 1000, step=100)
        chunk_overlap = st.slider("Chunk overlap", 0, 400, 150, step=50)
        top_k = st.slider("Retrieved chunks (k)", 2, 10, 4)
    
    st.divider()
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.session_state.get("clicked_prompt", None)
        st.rerun()

# API Key Enforcement
if not api_key:
    st.title("📄 PDF RAG AGENT")
    st.info("🔐 Please enter your OpenAI API key in the sidebar to get started.")
    st.stop()

os.environ["OPENAI_API_KEY"] = api_key

# 4. Main App Layout split into 2 visual zones
col_file, col_chat = st.columns([1, 2], gap="large")

with col_file:
    st.title("📄 PDF Upload")
    st.write("Upload a document to train your local context helper.")
    uploaded_pdf = st.file_uploader("Choose a PDF file", type=["pdf"], label_visibility="collapsed")
    
    if uploaded_pdf is not None:
        file_sig = (uploaded_pdf.name, uploaded_pdf.size, chunk_size, chunk_overlap)
        
        if st.session_state.get("file_sig") != file_sig:
            with st.spinner("⏳ Extracting text and building vector space..."):
                st.session_state.vectorstore = build_vectorstore(
                    uploaded_pdf.getvalue(), chunk_size, chunk_overlap
                )
                st.session_state.file_sig = file_sig
                st.session_state.messages = []
                st.toast(f"Successfully processed {uploaded_pdf.name}!", icon="✅")
        
        st.success(f"📁 Active Document: **{uploaded_pdf.name}**")
    else:
        if "vectorstore" in st.session_state:
            del st.session_state.vectorstore
            del st.session_state.file_sig
            st.session_state.messages = []

with col_chat:
    st.title("💬 Chat Window")
    
    if "vectorstore" in st.session_state:
        if "messages" not in st.session_state:
            st.session_state.messages = []
            
        # Replay conversation history
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                if "sources" in msg and msg["sources"]:
                    with st.expander("📚 View Sources used for this answer"):
                        st.code(msg["sources"], language="markdown")

        # Setup prompt tracking state
        prompt_to_submit = None

        # IMPROVEMENT: Interactive Welcome Prompts / Suggestion Pills
        if len(st.session_state.messages) == 0:
            st.write("💡 *Not sure where to start? Try clicking one of these quick prompts:*")
            p_col1, p_col2, p_col3 = st.columns(3)
            with p_col1:
                if st.button("📋 Summarize the main topics", use_container_width=True):
                    prompt_to_submit = "Can you provide a comprehensive summary of the main topics discussed in this document?"
            with p_col2:
                if st.button("🔑 Extract key takeaways", use_container_width=True):
                    prompt_to_submit = "What are the top 5 key takeaways or findings from this PDF?"
            with p_col3:
                if st.button("❓ Find target questions", use_container_width=True):
                    prompt_to_submit = "List 3 interesting questions that can be answered explicitly using this document."

        # Grab user input from the standard chat bar
        chat_input = st.chat_input("Ask anything contained within the document...")
        
        # Determine final active question (either from a suggestion pill click or typed text)
        question = chat_input or prompt_to_submit

        if question:
            st.session_state.messages.append({"role": "user", "content": question})
            with st.chat_message("user"):
                st.markdown(question)
                
            chain, retriever = get_chain(st.session_state.vectorstore, model_name, top_k)
            
            with st.chat_message("assistant"):
                # Fetch sources first so we have them ready
                source_docs = retriever.invoke(question)
                source_text = ""
                for i, doc in enumerate(source_docs):
                    page = doc.metadata.get("page", "?")
                    page_num = page + 1 if isinstance(page, int) else page
                    source_text += f"--- [Chunk {i+1} | Page {page_num}] ---\n{doc.page_content.strip()}\n\n"

                # IMPROVEMENT: Live Token-by-Token Streaming response container
                # Passing the chain generator directly into st.write_stream
                answer = st.write_stream(chain.stream(question))
                
                if source_text:
                    with st.expander("📚 View Sources used for this answer"):
                        st.code(source_text, language="markdown")
            
            st.session_state.messages.append({
                "role": "assistant", 
                "content": answer,
                "sources": source_text
            })
            
            # Forces page update if a button pill was used to clear the prompt dashboard layout
            if prompt_to_submit:
                st.rerun()
    else:
        st.caption("Waiting for a document to be uploaded on the left side menu...")
