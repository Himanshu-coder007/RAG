# RAG — Retrieval-Augmented Generation

A Python-based Retrieval-Augmented Generation (RAG) application that allows users to ask questions about documents and receive answers based on the retrieved document content.

## 🚀 Features

* Load PDF documents
* Split documents into smaller chunks
* Generate embeddings using Hugging Face Sentence Transformers
* Store embeddings in ChromaDB
* Perform similarity-based document retrieval
* Generate answers using Google Gemini
* Environment-based API key configuration using `.env`
* Interactive command-line chat interface

## 🏗️ Architecture

```text
Document (PDF)
      ↓
PyPDFLoader
      ↓
Text Splitter
      ↓
Document Chunks
      ↓
Hugging Face Embeddings
      ↓
ChromaDB
      ↓
User Query
      ↓
Similarity Search
      ↓
Relevant Context
      ↓
Gemini
      ↓
Generated Answer
```

## 🛠️ Technologies Used

* Python
* LangChain
* Hugging Face Sentence Transformers
* ChromaDB
* Google Gemini
* PyPDF
* python-dotenv

## 📁 Project Structure

```text
RAG/
│
├── sql.pdf
├── generate_embeddings.py
├── rag.py
├── .env
├── .gitignore
├── README.md
│
└── chroma_db_nccn/
```

> `.env` and `chroma_db_nccn/` should not be committed to Git.

## ⚙️ Installation

Clone the repository:

```bash
git clone <your-repository-url>
cd RAG
```

Create/install the required dependencies:

```bash
python -m pip install -U langchain-huggingface langchain-chroma langchain-text-splitters chromadb pypdf sentence-transformers python-dotenv google-genai
```

## 🔑 Environment Variables

Create a `.env` file in the project root:

```env
GEMINI_API_KEY=your_gemini_api_key
```

Never commit your `.env` file or expose your API key publicly.

## 📌 Generate Embeddings

Place your PDF document in the project directory and run:

```bash
python generate_embeddings.py
```

This processes the document, creates chunks, generates embeddings, and stores them in ChromaDB.

## 💬 Run the RAG Application

Start the application with:

```bash
python rag.py
```

You can then enter questions such as:

```text
What is the annual fixed compensation?
```

```text
What is the maximum annual earning potential?
```

```text
What is the variable bonus percentage?
```

The application retrieves relevant document chunks and sends that context to Gemini to generate the final answer.

## 🔄 How RAG Works

The application follows these steps:

1. A document is loaded.
2. The document is split into smaller chunks.
3. Each chunk is converted into a vector embedding.
4. The embeddings are stored in ChromaDB.
5. The user enters a question.
6. The question is compared with stored embeddings.
7. The most relevant chunks are retrieved.
8. The retrieved context is added to a prompt.
9. Gemini generates an answer using that context.

## 🔒 Security

Do not commit secrets to Git.

Recommended `.gitignore`:

```gitignore
.env
__pycache__/
*.pyc
chroma_db_nccn/
.venv/
venv/
```

If an API key has ever been committed or exposed, revoke/rotate it immediately and remove it from Git history before pushing the repository.

## 🚧 Future Improvements

* React frontend
* FastAPI backend
* PDF upload through the web interface
* Chat interface
* Multiple document support
* Document-specific Chroma collections
* Source citations in answers
* Conversation history
* Streaming responses
* Excel/CSV document support
* Improved retrieval with reranking

## 📄 License

This project is for educational and development purposes.
