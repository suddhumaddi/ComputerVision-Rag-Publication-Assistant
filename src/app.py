import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA

# --- Load embeddings & FAISS vectorstore ---
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = FAISS.load_local("vectorstore", embeddings, allow_dangerous_deserialization=True)

# --- LLM + Retriever ---
llm = Ollama(model="llama3")
retriever = db.as_retriever(search_kwargs={"k": 4})

qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True
)

# --- Streamlit UI ---
st.set_page_config(page_title="RAG Publication Assistant", layout="centered")

st.title("Computer Vision Scholar - An Assistant for CV Publication Data")
st.sidebar.header("❓ Suggested Questions")
st.sidebar.markdown("""
- What is this publication about?    
- When was this publication released?  
- What is the main goal or objective of this work?  
- Which problem does this publication aim to solve?  
- What methodology or approach is used in the publication?   
""")

query = st.text_input("Enter your question:")

if query:
    result = qa.invoke({"query": query})
    answer = result["result"]

    if answer.strip():
        st.success(answer)
    else:
        st.warning("⚠️ Sorry, I couldn’t find that info in the publication.")

    # Debug view for retrieved chunks
    with st.expander("🔎 Debug Retrieved Chunks"):
        for i, doc in enumerate(result["source_documents"], start=1):
            st.markdown(f"**Chunk {i}:**")
            st.write(doc.page_content[:600] + "...")
