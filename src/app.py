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





class RAGAssistant:
    """Retrieval-Augmented Generation assistant that can query multiple LLM providers.

    - Uses VectorDB for retrieval (expects vectordb.VectorDB compatibility).
    - Builds prompts using langchain_core.ChatPromptTemplate (required by your evaluator).
    - Supports any combination of: OpenAI (langchain_openai.ChatOpenAI), Groq (langchain_groq.ChatGroq),
      and Google Gemini (langchain_google_genai.ChatGoogleGenerativeAI) if their packages are installed
      and API keys are provided in the environment.
    """

    def __init__(self, prompt: Dict[str, Any], vector_db: Optional[VectorDB] = None):
        if ChatPromptTemplate is None:
            raise RuntimeError(
                "langchain_core package not found. Install langchain-core (and langchain-openai/langchain-groq/langchain-google-genai as needed)."
            )

        self.vector_db = vector_db or VectorDB()
        self.prompt = prompt
        self.models: Dict[str, Any] = {}
        self._initialize_llms()
        logger.info("RAGAssistant initialized with models: %s", list(self.models.keys()))

    def _initialize_llms(self) -> None:
        """Initialize available LangChain chat model wrappers if API keys + packages are present.

        The method will attempt to instantiate each wrapper. It prefers environment variables but
        will not fail if a provider is missing — the missing provider is simply not available.
        """
        # OpenAI
        openai_key = os.getenv("OPENAI_API_KEY")
        openai_model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        if openai_key:
            if ChatOpenAI is not None:
                try:
                    # Some langchain-openai variants accept `api_key`, others `openai_api_key` or rely on env vars.
                    try:
                        llm = ChatOpenAI(model=openai_model, api_key=openai_key, temperature=0.0)
                    except TypeError:
                        llm = ChatOpenAI(model=openai_model, openai_api_key=openai_key, temperature=0.0)
                    self.models["openai"] = llm
                except Exception as e:
                    logger.warning("Failed to initialize ChatOpenAI: %s", e)
            else:
                logger.warning("OPENAI_API_KEY present but langchain_openai not installed; skipping OpenAI wrapper.")

        # Groq
        groq_key = os.getenv("GROQ_API_KEY")
        groq_model = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
        if groq_key:
            if ChatGroq is not None:
                try:
                    try:
                        llm = ChatGroq(model=groq_model, api_key=groq_key, temperature=0.0)
                    except TypeError:
                        llm = ChatGroq(model=groq_model, groq_api_key=groq_key, temperature=0.0)
                    self.models["groq"] = llm
                except Exception as e:
                    logger.warning("Failed to initialize ChatGroq: %s", e)
            else:
                logger.warning("GROQ_API_KEY present but langchain_groq not installed; skipping Groq wrapper.")

        # Google Gemini
        google_key = os.getenv("GOOGLE_API_KEY")
        google_model = os.getenv("GOOGLE_MODEL", "gemini-pro")
        if google_key:
            if ChatGoogleGenerativeAI is not None:
                try:
                    try:
                        llm = ChatGoogleGenerativeAI(google_api_key=google_key, model=google_model, temperature=0.0)
                    except TypeError:
                        # fallback to generic constructor
                        llm = ChatGoogleGenerativeAI(model=google_model, temperature=0.0)
                    self.models["google"] = llm
                except Exception as e:
                    logger.warning("Failed to initialize ChatGoogleGenerativeAI: %s", e)
            else:
                logger.warning("GOOGLE_API_KEY present but langchain_google_genai not installed; skipping Google wrapper.")






    def _build_chat_prompt(self, user_message: str) -> Any:
        """Build a LangChain ChatPromptTemplate and format messages for the LLM.
        args:
            user_message: The user's question including context.
        returns:
            Returns a list of BaseMessage instances (SystemMessage, HumanMessage, etc.).
        """
        chat_template = ChatPromptTemplate.from_messages([
            ("system", "{system_prompt}"),
            ("human", "{user_message}"),
        ])

        formatted_messages = chat_template.format_messages(
            system_prompt=self.prompt.get("system_prompt", "You are a helpful assistant."),
            user_message=user_message,
        )
        return formatted_messages
    

    
    def add_documents(self, documents: List[Dict[str, Any]]) -> None:
        """Add documents to the vector DB.
        args:
            documents: List of dicts with "content" and optional "metadata" keys.
        """

        if not documents:
            logger.warning("No documents provided to add to vector DB.")
            return
        self.vector_db.add_documents(documents)
        logger.info(f"Added {len(documents)} document(s) to vector DB.")
        








    def query(
        self,
        question: str,
        n_results: int = 5,
        providers: Optional[Union[str, List[str]]] = "all",
    ) -> Dict[str, Any]:
        """Retrieve context from the vector DB and query one or more LLM providers.

        - `providers`: "all" (default) to query all available providers, a single provider string
          ("openai", "groq", "google"), or a list of those strings.
        args:
            question: The user's question.
            n_results: Number of context chunks to retrieve from the vector DB.
            providers: Which LLM providers to query ("all", single string, or list of strings).
        returns:
            Returns a dict with per-provider responses and the retrieval context used.
        """
        if not question or not question.strip():
            raise ValueError("Question must be non-empty.")

        docs = self.vector_db.search(question, n_results)
        context_text = "\n\n".join(docs.get("documents", []))

        user_message = f"{self.prompt.get('full_prompt','')}\n\nContext:\n{context_text}\n\nQuestion: {question}"
        formatted_messages = self._build_chat_prompt(user_message)

        # Determine providers to call
        if providers == "all" or providers is None:
            target_providers = list(self.models.keys())
        elif isinstance(providers, str):
            target_providers = [providers]
        else:
            target_providers = list(providers)

        if not target_providers:
            raise RuntimeError("No LLM providers are available. Set API keys and install provider packages.")

        results: Dict[str, Any] = {"question": question, "context_used": context_text, "responses": {}}

        for p in target_providers:
            llm = self.models.get(p)
            if llm is None:
                results["responses"][p] = {"error": "provider not available"}
                continue

            try:
                # Most langchain chat model wrappers accept a list of BaseMessage objects directly
                resp = llm.invoke(formatted_messages)

                # Best-effort content extraction
                content = None
                if hasattr(resp, "content"):
                    content = resp.content
                elif hasattr(resp, "text"):
                    try:
                        content = resp.text()
                    except TypeError:
                        content = resp.text
                elif isinstance(resp, dict):
                    content = resp.get("text") or resp.get("content") or str(resp)
                else:
                    content = str(resp)

                results["responses"][p] = {"answer": content}

            except Exception as e:
                logger.exception("LLM call failed for provider %s: %s", p, e)
                results["responses"][p] = {"error": str(e)}

        return results

