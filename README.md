# PDF Q&A Chatbot

A simple RAG (Retrieval-Augmented Generation) chatbot that allows users to upload PDF documents and ask questions about their content.

## Features

- 📄 Upload PDF documents
- 💬 Ask natural language questions
- 🔍 Intelligent context retrieval using ChromaDB
- 🤖 Powered by Claude AI for accurate answers
- 🎨 Clean, professional interface

## Tech Stack

- **Frontend**: Streamlit
- **Vector Database**: ChromaDB
- **Embeddings**: HuggingFace (sentence-transformers)
- **LLM**: Claude (Anthropic)
- **Framework**: LangChain

## Local Setup

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd pdf-qa-chatbot
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variable**
   ```bash
   export ANTHROPIC_API_KEY="your-api-key-here"
   ```

4. **Run the application**
   ```bash
   streamlit run app.py
   ```

## Deployment to Streamlit Cloud

1. **Push your code to GitHub**
   - Create a new repository on GitHub
   - Push the code to your repository

2. **Deploy on Streamlit Cloud**
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Click "New app"
   - Select your repository, branch, and `app.py`
   - Add your API key in "Advanced settings" → "Secrets"
   
3. **Configure Secrets**
   In the Streamlit Cloud dashboard, add:
   ```toml
   ANTHROPIC_API_KEY = "your-api-key-here"
   ```

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

## Troubleshooting

**Issue**: "ANTHROPIC_API_KEY not found"
- **Solution**: Ensure you've set the environment variable or added it to Streamlit secrets

**Issue**: PDF processing fails
- **Solution**: Ensure PDF is text-based (not scanned images)

**Issue**: Slow response time
- **Solution**: Consider reducing chunk size or top_k parameter

## Future Enhancements

- Support for multiple PDFs
- Citation of source chunks
- Export chat history
- Custom chunk size configuration
- Support for other document formats

## License

MIT License

## Support

For issues or questions, please open an issue on GitHub.
