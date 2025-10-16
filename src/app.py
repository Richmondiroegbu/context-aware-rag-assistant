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

