# RAG (Retrieval-Augmented Generation) Pipeline

A complete implementation of a RAG system using LangChain, ChromaDB, and Groq LLM for intelligent document querying with conversation history support.

## 🎯 Overview

This project demonstrates a production-ready RAG pipeline that combines semantic search with large language models to provide accurate, context-aware answers from your documents. It features three levels of sophistication:

- **Simple RAG**: Basic retrieval and generation
- **Enhanced RAG**: Confidence scoring and source tracking
- **Advanced RAG**: Multi-turn conversations with history awareness

## ✨ Features

- 📄 **Multi-format Document Loading**: Text files and PDFs
- 🔍 **Semantic Search**: Using SentenceTransformer embeddings
- 💾 **Persistent Vector Store**: ChromaDB with cosine similarity
- 🤖 **LLM Integration**: ChatGroq (Gemma2-9b-it model)
- 💬 **Conversation History**: Context-aware multi-turn dialogues
- 📊 **Confidence Metrics**: Similarity scores and source citations
- 🎨 **Professional Output**: Formatted results with emojis and structure
- 🔧 **Extensible Architecture**: Easy to customize and extend

## 🏗️ Architecture

```
Document Sources (PDF/TXT)
         ↓
    Text Splitter (Chunking)
         ↓
  Embedding Generation (SentenceTransformer)
         ↓
   Vector Store (ChromaDB)
         ↓
    RAG Retriever ←→ User Query
         ↓
   LLM (ChatGroq) + Context
         ↓
    Generated Answer + Sources
```

## 📋 Requirements

- Python 3.8+
- Required packages (see `requirements.txt`):
  - langchain
  - langchain-community
  - langchain-groq
  - sentence-transformers
  - chromadb
  - numpy
  - python-dotenv
  - pymupdf (for PDF processing)

## 🚀 Installation

1. **Clone the repository**:
```bash
git clone https://github.com/PhoneixDeadeye/RAG.git
cd RAG
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Set up environment variables**:
Create a `.env` file in the root directory:
```env
GROQ_API_KEY=your_groq_api_key_here
```

Get your Groq API key from: https://console.groq.com/

## 📁 Project Structure

```
RAG/
├── notebook/
│   └── document.ipynb          # Main Jupyter notebook with full implementation
├── data/
│   ├── text_files/             # Input text files
│   ├── pdf_files/              # Input PDF files
│   └── vector_store/           # Persistent ChromaDB storage
├── src/
│   └── __init__.py
├── README.md
├── requirements.txt
├── pyproject.toml
└── uv.lock
```

## 💻 Usage

### Quick Start

1. **Open the Jupyter notebook**:
```bash
jupyter notebook notebook/document.ipynb
```

2. **Run all cells** to initialize the components

3. **Query your documents**:

#### Simple RAG
```python
answer = rag_simple("What are the key features?", rag_retriever, llm, top_k=3)
print(answer)
```

#### Enhanced RAG (with confidence scores)
```python
result = rag_enhanced(
    "What are the key features?", 
    rag_retriever, 
    llm, 
    top_k=3, 
    min_score=0.1, 
    return_contexts=True
)
print(f"Answer: {result['answer']}")
print(f"Confidence: {result['confidence']:.2%}")
```

#### Advanced RAG (with conversation history)
```python
# Initialize pipeline
advanced_rag = AdvancedRAGPipeline(rag_retriever, llm)

# First query
result1 = advanced_rag.query("What is machine learning?", top_k=5)

# Follow-up query (uses conversation history)
result2 = advanced_rag.query("How is it different from deep learning?", top_k=5)

# View conversation history
history = advanced_rag.get_history()

# Clear history when needed
advanced_rag.clear_history()
```

## 🔧 Components

### 1. EmbeddingManager
Handles document embedding generation using SentenceTransformer:
- Model: `all-MiniLM-L6-v2` (384-dimensional embeddings)
- Batch processing with progress tracking

### 2. VectorStore
Manages document embeddings in ChromaDB:
- Persistent storage
- Cosine similarity metric
- Automatic collection management

### 3. RAGRetriever
Query-based document retrieval:
- Top-k similarity search
- Score threshold filtering
- Metadata preservation

### 4. AdvancedRAGPipeline
Multi-turn conversation support:
- Tracks last 3 conversation turns
- Context-aware prompt construction
- History management methods

## 📊 Example Output

```
🔍 RAG ENHANCED QUERY DEMO
====================================================================================================

📋 Query: 'Which technique is used for caching?'
⚙️  Parameters: top_k=3, min_score=0.1, return_contexts=True
----------------------------------------------------------------------------------------------------

💡 GENERATED ANSWER:
====================================================================================================
The document mentions using Redis for caching to improve performance...
====================================================================================================

📚 SOURCES & EVIDENCE:
----------------------------------------------------------------------------------------------------

┌─ SOURCE #1
│  📄 File: rohan_cv_aug.pdf
│  📖 Page: 2
│  🎯 Similarity Score: 0.8523 (85.23%)
│
│  📝 Preview:
│  Implemented Redis caching layer to reduce database load by 40%...
└──────────────────────────────────────────────────────────────────────────────────────────────────

📊 RETRIEVAL METRICS:
----------------------------------------------------------------------------------------------------
✅ Overall Confidence: 0.8523 (85.23%)
📦 Documents Retrieved: 3
```

## 🎓 Key Concepts

### Chunking Strategy
- **Chunk Size**: 500 characters
- **Chunk Overlap**: 50 characters
- **Separators**: Paragraphs → Lines → Spaces

### Retrieval Parameters
- **top_k**: Number of documents to retrieve (default: 3-5)
- **min_score**: Minimum similarity threshold (default: 0.2)
- **score_threshold**: Filter results by confidence

### Conversation History
- Maintains last 3 conversation turns
- Automatically included in prompt context
- Enables follow-up questions and clarifications

## 🔍 Advanced Features

### Custom Document Processing
```python
# Add your own documents
from langchain_community.document_loaders import DirectoryLoader

loader = DirectoryLoader("your_docs/", glob="**/*.pdf")
documents = loader.load()
chunks = split_documents(documents, chunk_size=500, chunk_overlap=50)
```

### Adjust Embedding Model
```python
embedding_manager = EmbeddingManager(model_name="all-mpnet-base-v2")
```

### Fine-tune LLM Parameters
```python
llm = ChatGroq(
    api_key=groq_api_key, 
    model="gemma2-9b-it", 
    temperature=0.1,  # Lower = more deterministic
    max_tokens=1024    # Maximum response length
)
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is open source and available under the MIT License.

## 👤 Author

**Rohan**
- GitHub: [@PhoneixDeadeye](https://github.com/PhoneixDeadeye)

## 🙏 Acknowledgments

- [LangChain](https://github.com/langchain-ai/langchain) - Framework for LLM applications
- [ChromaDB](https://www.trychroma.com/) - Vector database
- [SentenceTransformers](https://www.sbert.net/) - Embedding models
- [Groq](https://groq.com/) - Fast LLM inference

## 📚 Resources

- [LangChain Documentation](https://python.langchain.com/)
- [ChromaDB Documentation](https://docs.trychroma.com/)
- [RAG Paper](https://arxiv.org/abs/2005.11401)
- [Sentence-BERT Paper](https://arxiv.org/abs/1908.10084)

## 🐛 Known Issues

- Large PDF files (>100 pages) may take longer to process
- Groq API has rate limits on free tier

## 🗺️ Roadmap

- [ ] Add support for more document formats (DOCX, HTML)
- [ ] Implement streaming responses
- [ ] Add web interface with Streamlit/Gradio
- [ ] Multi-language support
- [ ] Query optimization and caching
- [ ] Batch processing for large document sets
- [ ] Advanced citation and source linking

## 📞 Support

If you encounter any issues or have questions, please file an issue on GitHub.

---

⭐ If you find this project helpful, please give it a star!
