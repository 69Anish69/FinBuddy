# modules/chatbot.py
import re
from transformers import pipeline
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFacePipeline
from langchain.docstore.document import Document
import streamlit as st

# Load Hugging Face text generation pipeline locally
generator = pipeline("text2text-generation", model="google/flan-t5-base", max_length=256)
llm = HuggingFacePipeline(pipeline=generator)

qa_system = None

def display_chatbot_ui():
    st.header("💬 Ask AI about your Financial Docs")
    question = st.text_input("Ask a question:")
    if question:
        answer = answer_question(question)
        st.success(answer)

def load_docs_and_create_qa_system(doc_texts):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    documents = []
    for text in doc_texts:
        docs = splitter.create_documents([text])
        documents.extend(docs)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = FAISS.from_documents(documents, embeddings)

    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=vector_store.as_retriever())
    return qa_chain

def set_documents(doc_texts):
    global qa_system
    qa_system = load_docs_and_create_qa_system(doc_texts)

def answer_question(question):
    global qa_system
    if not qa_system:
        return "⚠️ No documents loaded. Please upload financial data first."

    # Handle basic finance questions via simple logic
    q = question.lower()
    if "total balance" in q or "closing balance" in q:
        for doc in qa_system.retriever.vectorstore.docstore._dict.values():
            match = re.search(r"Closing Balance:\s*\$([0-9.,]+)", doc.page_content)
            if match:
                return f"🏦 Closing balance is ${match.group(1)}"
        return "❌ Couldn't find closing balance in the document."

    elif "total income" in q:
        total = 0
        for doc in qa_system.retriever.vectorstore.docstore._dict.values():
            matches = re.findall(r"\|.*\|\s*\+([0-9.]+)", doc.page_content)
            total += sum(float(m) for m in matches)
        return f"💰 Total income is ${total:.2f}"

    elif "total expense" in q:
        total = 0
        for doc in qa_system.retriever.vectorstore.docstore._dict.values():
            matches = re.findall(r"\|.*\|\s*-([0-9.]+)", doc.page_content)
            total += sum(float(m) for m in matches)
        return f"💸 Total expenses are ${total:.2f}"

    try:
        return qa_system.run(question)
    except Exception as e:
        return f"❌ Error: {str(e)}"