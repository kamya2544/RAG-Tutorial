# PDF Q&A Chatbot

A simple RAG (Retrieval-Augmented Generation) chatbot that allows users to upload PDF documents and ask questions about their content.

## Features

- Upload PDF documents
- Ask PDF related questions
- Intelligent context retrieval using ChromaDB
- Powered by Claude AI for accurate answers
- Clean, professional interface

## Tech Stack

- **Frontend**: Streamlit
- **Vector Database**: ChromaDB
- **Embeddings**: HuggingFace (sentence-transformers)
- **LLM**: Claude (Anthropic)
- **Framework**: LangChain


## How It Works

1. **Upload PDF**: User uploads a PDF document through the sidebar
2. **Text Extraction**: System extracts text from all pages
3. **Chunking**: Text is split into manageable chunks (1000 chars with 200 overlap)
4. **Embedding**: Each chunk is converted to vector embeddings
5. **Storage**: Embeddings stored in ChromaDB vector database
6. **Question**: User asks a question
7. **Search**: System finds most relevant chunks using similarity search
8. **Answer**: Claude generates an answer based on relevant context

## File Structure

```
pdf-qa-chatbot/
├── app.py              # Main Streamlit application
├── requirements.txt    # Python dependencies
└── README.md          # Documentation
```

## Key Components

### PDFChatbot Class
- `extract_text_from_pdf()`: Extracts text from PDF files
- `process_document()`: Chunks text and stores in vector database
- `find_relevant_context()`: Retrieves relevant document sections
- `generate_answer()`: Uses Claude to generate answers

## Configuration

- **Chunk Size**: 1000 characters
- **Chunk Overlap**: 200 characters
- **Top K Results**: 3 most relevant chunks
- **Embedding Model**: all-MiniLM-L6-v2
- **LLM Model**: Claude Sonnet 4


## Future Enhancements

- Support for multiple PDFs
- Export chat history
- Custom chunk size configuration
- Support for other document formats

