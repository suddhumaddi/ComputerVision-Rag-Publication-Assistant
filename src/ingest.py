import json
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document

# Paths
DATA_PATH = "data/project_1_publications.json"
VECTOR_DB_PATH = "vectorstore"

def load_publications():
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        publications = json.load(f)

    docs = []
    for pub in publications:
        enriched_text = f"""
Title: {pub.get('title', 'Unknown')}
Author: {pub.get('username', 'Unknown')}
License: {pub.get('license', 'Unknown')}
Description: {pub.get('publication_description', '')}
"""
        docs.append(Document(page_content=enriched_text, metadata={"id": pub.get("id", "")}))
    return docs

def chunk_and_embed(docs):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100
    )
    chunks = []
    for doc in docs:
        chunks.extend(text_splitter.split_documents([doc]))

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(chunks, embeddings)

    vectorstore.save_local(VECTOR_DB_PATH)
    print(f"✅ Vectorstore saved at {VECTOR_DB_PATH}")

if __name__ == "__main__":
    docs = load_publications()
    chunk_and_embed(docs)
