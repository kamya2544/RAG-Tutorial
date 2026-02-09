"""PDF Q&A Chatbot - RAG-based PDF document Q&A assistant"""

import os
# Disable Hugging Face telemetry to avoid telemetry-related runtime errors
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

import streamlit as st
import chromadb
from chromadb.config import Settings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
import PyPDF2
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="PDF Q&A Assistant", layout="wide")
if "messages" not in st.session_state:
    st.session_state.messages = []
if "pdf_processed" not in st.session_state:
    st.session_state.pdf_processed = False


class PDFChatbot:
    """PDF document processor and Q&A system."""
    
    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        self.client = chromadb.Client(Settings(anonymized_telemetry=False, is_persistent=False))
        try:
            self.tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
            self.model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model.to(self.device)
        except Exception as e:
            self.tokenizer = None
            self.model = None
            st.error(f"Failed to load generation model: {e}")
        self.collection = None
    
    def extract_text(self, pdf_file) -> str:
        """Extract text from PDF."""
        # Try PyPDF2 first (fast, pure-Python)
        try:
            reader = PyPDF2.PdfReader(pdf_file)
            texts = []
            empty_pages = []
            for i, page in enumerate(reader.pages):
                try:
                    t = page.extract_text() or ""
                except Exception:
                    t = ""
                texts.append(t)
                if not t.strip():
                    empty_pages.append(i)

            full = "\n".join([t for t in texts if t and t.strip()])
            if full.strip():
                return full
        except Exception:
            pass

        # Fallback to PyMuPDF (better extraction for complex PDFs)
        try:
            import fitz  # PyMuPDF

            # Streamlit uploaded files are file-like; read bytes then reopen in fitz
            if hasattr(pdf_file, "read"):
                pdf_bytes = pdf_file.read()
                try:
                    pdf_file.seek(0)
                except Exception:
                    pass
                doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            else:
                doc = fitz.open(pdf_file)

            texts = []
            empty_pages = []
            for i, page in enumerate(doc):
                try:
                    t = page.get_text("text") or ""
                except Exception:
                    t = ""
                texts.append(t)
                if not t.strip():
                    empty_pages.append(i)

            full = "\n".join([t for t in texts if t and t.strip()])
            if full.strip():
                return full
        except Exception:
            pass

        # Last resort: OCR pages that appear empty (requires pytesseract + pdf2image)
        try:
            from pdf2image import convert_from_bytes, convert_from_path
            import pytesseract

            if hasattr(pdf_file, "read"):
                pdf_bytes = pdf_file.read()
                try:
                    pdf_file.seek(0)
                except Exception:
                    pass
                images = convert_from_bytes(pdf_bytes, dpi=300)
            else:
                images = convert_from_path(pdf_file, dpi=300)

            ocr_texts = []
            for img in images:
                try:
                    ocr_texts.append(pytesseract.image_to_string(img))
                except Exception:
                    ocr_texts.append("")

            full = "\n".join(ocr_texts)
            return full
        except Exception:
            return ""
    
    def process_document(self, pdf_file) -> bool:
        """Process and index PDF document."""
        try:
            text = self.extract_text(pdf_file)
            if not text.strip():
                st.error("PDF appears empty.")
                return False
            
            try:
                self.client.delete_collection("pdf_documents")
            except:
                pass
            
            self.collection = self.client.create_collection("pdf_documents")
            chunks = self.text_splitter.split_text(text)
            
            for idx, chunk in enumerate(chunks):
                try:
                    self.collection.add(
                        embeddings=[self.embeddings.embed_query(chunk)],
                        documents=[chunk],
                        ids=[f"chunk_{idx}"]
                    )
                except:
                    continue
            return True
        except Exception as e:
            st.error(f"Processing failed: {str(e)}")
            return False
    
    def answer_question(self, question: str) -> str:
        """Find context and generate answer."""
        if not self.collection:
            return "No document loaded."
        
        try:
            results = self.collection.query(
                query_embeddings=[self.embeddings.embed_query(question)],
                n_results=3
            )
            context = "\n\n".join(results.get("documents", [[]])[0])
            
            if not self.model or not self.tokenizer:
                return "Generation model unavailable."

            prompt = f"Answer based on: {context}\n\nQuestion: {question}\nAnswer:"
            inputs = self.tokenizer(prompt, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            outputs = self.model.generate(**inputs, max_length=512)
            generated = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
            return generated
        except Exception as e:
            return f"Error: {str(e)}"


@st.cache_resource
def get_chatbot():
    return PDFChatbot()

chatbot = get_chatbot()

st.title("PDF Q&A Assistant")
st.markdown("Upload a PDF and ask questions about its content.")

with st.sidebar:
    st.header("Upload Document")
    file = st.file_uploader("Choose PDF", type="pdf")
    
    if file and not st.session_state.pdf_processed:
        with st.spinner("Processing..."):
            if chatbot.process_document(file):
                st.session_state.pdf_processed = True
                st.session_state.messages = []
                st.success("Ready!")
    
    if st.session_state.pdf_processed and st.button("Clear History"):
        st.session_state.messages = []
        st.rerun()

if st.session_state.pdf_processed:
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
    
    if question := st.chat_input("Ask about the document"):
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.write(question)
        
        with st.chat_message("assistant"):
            response_placeholder = st.empty()
            response_placeholder.write("Generating response...")
            answer = chatbot.answer_question(question)
            response_placeholder.write(answer)
        
        st.session_state.messages.append({"role": "assistant", "content": answer})
else:
    st.info("Upload a PDF to start asking questions.")

st.markdown("---")
st.markdown("<p style='text-align:center; color:gray;'>Built by Kamya Mehra</p>", unsafe_allow_html=True)
