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





class VectorDB:
    """
    Wrapper around ChromaDB using SentenceTransformer embeddings,
    now with optional conversational memory (ConversationSummaryMemory).
    """

    def __init__(
        self,
        collection_name: Optional[str] = None,
        embedding_model: Optional[str] = None,
        enable_memory: bool = True,
    ):
        self.collection_name = collection_name or os.getenv("CHROMA_COLLECTION_NAME", "rag_documents")
        self.embedding_model_name = embedding_model or os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
        db_path = os.getenv("CHROMA_DB_PATH", "./chroma_db")


        if chromadb is None:
            raise RuntimeError("chromadb package not installed. `pip install chromadb`")

        try:
            if hasattr(chromadb, "PersistentClient"):
                self.client = chromadb.PersistentClient(path=db_path)
            else:
                self.client = chromadb.Client()
        except Exception as e:
            raise RuntimeError(f"Failed to initialize chromadb client: {e}")

    
        if SentenceTransformer is None:
            raise RuntimeError("sentence-transformers is not installed. `pip install sentence-transformers` (also install torch).")

        logger.info("Loading sentence-transformers model (this may take a while on first run)...")
        self.embedding_model = SentenceTransformer(self.embedding_model_name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"description": "RAG document collection"},
        )

    
        self.memory = None
        if enable_memory:
            try:
                llm = ChatGroq(temperature=0)
                self.memory = ConversationSummaryMemory(llm=llm, memory_key="chat_history", return_messages=True)
                logger.info("Conversation memory initialized with ConversationSummaryMemory.")
            except Exception as e:
                logger.warning(f"Failed to initialize memory: {e}")

        logger.info(f"VectorDB initialized with collection '{self.collection_name}'")


    @staticmethod
    def _get_token_enconder():
        """Get tiktoken encoder for token counting."""
        try:
            return tiktoken.get_encoding("cl100k_base")
        except Exception:
            try:
                return tiktoken.encoding_for_model("gpt-3.5-turbo")
            except Exception:
                return None
            

    
    @classmethod
    def token_len(cls, text: str) -> int:
        """Return the number of tokens in a text string.
        args:
            text (str): input text
        returns:
            token count (int)
        """
        enc = cls._get_token_enconder()
        if not enc:
            logger.warning("tiktoken not available, returning character length instead of token length.")
            return len(text.split())
        try:
            return len(enc.encode(text or "", disallowed_special=()))
        except Exception:
            logger.warning("tiktoken encoding failed, returning character length instead of token length.")
            return len(text.split())




    def chunk_text(
        self,
        text: str,
        chunk_size: int = 700,
        chunk_overlap: int = 150,
    ) -> List[str]:
        """Chunk text into smaller pieces using RecursiveCharacterTextSplitter.
        args:
            text (str): Input text to chunk.
            chunk_size (int): max token per chunk .
            chunk_overlap (int): Overlap between chunks.
        returns:
            List of text chunks.
        """

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=self.token_len,
            separators=["\n\n", "\n", " ", ""]
        )
        return splitter.split_text(text or "")
    

    def add_documents(self, documents: List[Dict[str, Any]], batch_size: int = 100) -> None:
        """Add documents to the vector DB.
        args:
            documents (list[Dict[str, Any]]): List of dicts with "content" and optional "metadata" keys.
            batch_size (int): number of chunks to add in each mini-batch to vector DB
        """

        if not documents:
            logger.warning("No documents to add.")
            return
        
        total_added = 0
        for doc in documents:
            content = doc.get("content", "").strip()
            metadata = doc.get("metadata", {})
            if not content:
                continue
            
            chunks = self.chunk_text(content)
            if not chunks:
                continue

            try:
                embeddings = self.embedding_model.encode(chunks, batch_size=8, show_progress_bar=False, convert_to_numpy=True)
                if isinstance(embeddings, np.ndarray):
                    embeddings = embeddings.tolist()
            except TypeError:
                embeddings = self.embedding_model.encode(chunks, batch_size=8, show_progress_bar=False)
                if isinstance(embeddings, np.ndarray):
                    embeddings = embeddings.tolist()


            chunk_ids = [str(uuid.uuid4()) for _ in chunks]

            for i in range(0, len(chunks), batch_size):
                batch_texts = chunks[i:i + batch_size]
                batch_embeddings = embeddings[i:i + batch_size]
                batch_meta = [metadata] * len(batch_texts)
                batch_ids = chunk_ids[i:i + batch_size]
                self.collection.add(
                    documents=batch_texts,
                    embeddings=batch_embeddings,
                    metadatas=batch_meta,
                    ids=batch_ids
                )

            total_added += len(batch_texts)

        logger.info(f"Added {total_added} document chunks to collection '{self.collection_name}'")


    def search(
        self,
        query: str,
        n_results: int = 5,
    ) -> List[Dict[str, Any]]:
        """Search the vector DB for similar documents.
        args:
            query (str): The search query.
            n_results (int): Number of top results to return.
        returns:
            Dict[str, Any]: search results with documents, metadatas, distances.
            """
        if not query or not query.strip():
            return {"documents": [], "metadatas": [], "distances": []}
        
        try:
            query_embedding = self.embedding_model.encode([query], convert_to_numpy=True)
            if isinstance(query_embedding, np.ndarray):
                query_embedding = query_embedding.tolist()
        except TypeError:
            query_embedding = self.embedding_model.encode([query])
            if isinstance(query_embedding, np.ndarray):
                query_embedding = query_embedding.tolist()

        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=n_results,
            include=["documents", "metadatas", "distances"]
        )


        if self.memory:
            context = "\n".join(results.get("documents", [[]])[0])

            try:
                self.memory.save_context(
                    {"input": query},
                    {"output": context[:1000]}
                )
            except Exception as e:
                logger.warning(f"Memory update Failed: {e}")


            return{
                "documents": results.get("documents", [[]])[0],
                "metadatas": results.get("metadatas", [[]])[0],
                "distances": results.get("distances", [[]])[0],           
            }
        


        
    def get_conversation_summary(self) -> Optional[str]:
        """Get the current conversation summary from memory, if enabled.
        returns:
            Conversation summary string or None if memory not enabled.
        """
        if self.memory:
            return None
        try:
            summary = self.memory.load_memory_variables({}).get("chat_history", "")
            return summary
        except Exception as e:
            logger.warning(f"Failed to get conversation summary: {e}")
            return None


        
        
        




