import os
import logging
from typing import List, Dict, Any, Optional, Union
from pathlib import Path
from dotenv import load_dotenv
from PyPDF2 import PdfReader

from vectordb import VectorDB
from prompt_builder import PromptBuilder
from prompt_instructions import PromptInstructions

try:
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.messages import SystemMessage, HumanMessage
except Exception:
    # If langchain_core is not installed, user will get an informative error later.
    ChatPromptTemplate = None
    SystemMessage = None
    HumanMessage = None

# Optional langchain chat model wrappers
try:
    from langchain_openai import ChatOpenAI
except Exception:
    ChatOpenAI = None
try:
    from langchain_groq import ChatGroq
except Exception:
    ChatGroq = None
try:
    from langchain_google_genai import ChatGoogleGenerativeAI
except Exception:
    ChatGoogleGenerativeAI = None


logging.basicConfig(
    level=logging.INFO, 
    format="%(levelname)s: %(message)s")

logger = logging.getLogger(__name__)

load_dotenv()
DEFAULT_DATA_PATH = Path(os.getenv("DATA_PATH", "./data"))





def extract_text_from_pdf(file_path: Union[str, Path]) -> str:
    """Extract text from PDF file.
    args:
        file_path: path to PDF file
    returns:
        Extracted text as a single string.
    """

    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    reader = PdfReader(str(file_path))
    text_pages = []
    for p in reader.pages:
        page_text = p.extract_text()
        if page_text:
            text_pages.append(page_text)
    return "\n".join(text_pages).strip()







def load_documents(data_path: Union[str, Path]) -> List[Dict[str, Any]]:
    """Load .pdf, .txt, and .md files from a directory or single file.
    args:
        data_path: directory or single file path
    returns:
        List of dicts with "content" and "metadata" keys.
    """

    p = Path(data_path)
    if not p.exists():
        raise FileNotFoundError(f"Data path not found: {p}")

    files = [p] if p.is_file() else list(p.glob("*"))
    supported_exts = {".pdf", ".txt", ".md"}
    results: List[Dict[str, Any]] = []

    for f in files:
        if f.suffix.lower() not in supported_exts:
            continue
        if f.suffix.lower() == ".pdf":
            content = extract_text_from_pdf(f)
            metadata = {"source": str(f), "type": "pdf"}
        else:
            content = f.read_text(encoding="utf-8", errors="ignore")
            metadata = {"source": str(f), "type": "text"}
        if content and content.strip():
            results.append({"content": content.strip(), "metadata": metadata})

    logger.info(f"Loaded {len(results)} document(s)")
    return results




