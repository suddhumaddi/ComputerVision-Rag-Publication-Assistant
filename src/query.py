from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_ollama import OllamaLLM

# Paths
VECTOR_DB_PATH = "vectorstore"

# Load embeddings + vectorstore
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.load_local(VECTOR_DB_PATH, embeddings, allow_dangerous_deserialization=True)

# Use Ollama (local LLaMA 3)
llm = OllamaLLM(model="llama3")

# Custom prompt to restrict answers only to publication data
template = """
You are a helpful assistant that answers questions strictly based on the given publication data.
If the answer is not in the data, reply: "I don't know. This is outside the provided publication information."

Context:
{context}

Question:
{question}

Answer:
"""

prompt = PromptTemplate(template=template, input_variables=["context", "question"])

qa = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
    chain_type_kwargs={"prompt": prompt}
)

def main():
    print("📚 Ask me anything about the publication! Type 'exit' to quit.")
    while True:
        query = input("You: ")
        if query.lower() in ["exit", "quit"]:
            break
        answer = qa.run(query)
        print("Assistant:", answer)

if __name__ == "__main__":
    main()
