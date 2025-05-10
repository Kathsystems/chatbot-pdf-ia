import streamlit as st
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.document_loaders import TextLoader
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from dotenv import load_dotenv
from langchain.text_splitter import CharacterTextSplitter

load_dotenv()

st.title("📚 Chatbot com PDFs")

query = st.text_input("Faça uma pergunta sobre o conteúdo:")

if query:
    loader = TextLoader("inputs/exemplo.txt")
    documents = loader.load()
    splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    texts = splitter.split_documents(documents)

    embeddings = OpenAIEmbeddings()
    db = FAISS.from_documents(texts, embeddings)

    docs = db.similarity_search(query)
    chain = load_qa_chain(OpenAI(temperature=0), chain_type="stuff")
    resposta = chain.run(input_documents=docs, question=query)

    st.markdown("**Resposta:**")
    st.write(resposta)
