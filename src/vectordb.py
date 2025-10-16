import os
import uuid
import logging
from typing import List, Dict, Any, Optional
import numpy as np


try:
    import chromadb
except Exception:
    chromadb = None

# sentence-transformers is optional (heavy). If not present, we'll raise informative errors.
try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None

from langchain.text_splitter import RecursiveCharacterTextSplitter
import tiktoken

# --- New imports for memory ---
from langchain.memory import ConversationSummaryMemory
from langchain_groq import ChatGroq


logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s]: %(message)s"
)
