# 🤖 Chatbot com PDFs usando IA (LangChain + OpenAI + FAISS)

Projeto criado para responder perguntas com base em documentos PDF ou textos usando IA Generativa, embeddings e busca vetorial.

## 🚀 Funcionalidades

- Carregamento de arquivos de texto
- Vetorização com FAISS
- Embeddings com OpenAI
- Respostas com base em conteúdo específico

## 🧪 Exemplo

### Entrada:

> O que o texto diz sobre inteligência artificial?

### Resposta:

> A inteligência artificial é uma área da computação que simula comportamentos humanos, como raciocínio, aprendizado e adaptação.

## 🛠️ Como Rodar

```bash
git clone https://github.com/seuusuario/chatbot-pdf-ia.git
cd chatbot-pdf-ia
pip install -r requirements.txt
streamlit run app/main.py
