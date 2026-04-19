import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from pathlib import Path
import shutil
from ..config import settings

DB_URL = settings.chroma_url


def _is_recoverable_chroma_error(exc: BaseException) -> bool:
    error_text = str(exc).lower()
    return (
        "range start index" in error_text
        or "pyo3_runtime.panicexception" in error_text
        or "could not connect to tenant" in error_text
        or str(type(exc).__name__).lower() == "panicexception"
    )


def _init_chroma_client(path: str):
    """Initialize Chroma client and recover automatically from local index corruption."""
    try:
        return chromadb.PersistentClient(path=path)
    except BaseException as exc:
        db_path = Path(path)
        if db_path.exists():
            shutil.rmtree(db_path, ignore_errors=True)
        try:
            return chromadb.PersistentClient(path=path)
        except BaseException as retry_exc:
            if db_path.exists():
                shutil.rmtree(db_path, ignore_errors=True)
            return chromadb.PersistentClient(path=path)

client = _init_chroma_client(DB_URL)
embedding_function = SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")

# Get or create the unified collection
collection = client.get_or_create_collection(
    name="notes_collection",
    embedding_function=embedding_function
)

def add_to_vector_db(user_id: str, id: str, title: str, body: str, tags: str = ""):
    """Embeds and adds a new note to the vector database."""
    document = f"Title: {title}\nTags: {tags}\n\n{body}"
    collection.add(
        ids=[str(id)],
        documents=[document],
        metadatas=[{"user_id": user_id, "title": title, "tags": tags}]
    )

def update_in_vector_db(user_id: str, id: str, title: str, body: str, tags: str = ""):
    """Updates an existing note's embedded representation."""
    document = f"Title: {title}\nTags: {tags}\n\n{body}"
    collection.update(
        ids=[str(id)],
        documents=[document],
        metadatas=[{"user_id": user_id, "title": title, "tags": tags}]
    )

def delete_from_vector_db(id: str):
    """Removes a note's embedding from the vector database by ID."""
    try:
        collection.delete(ids=[str(id)])
    except Exception:
        # Already deleted or never embedded, safe to swallow
        pass

def semantic_search_vector_db(user_id: str, query: str, limit: int = 5) -> list[str]:
    """
    Performs semantic similarity search filtered to the given user's notes.
    Returns a list of matching note IDs (as strings).
    """
    try:
        results = collection.query(
            query_texts=[query],
            n_results=limit,
            where={"user_id": user_id},
        )
    except Exception:
        # Collection empty or other transient error
        return []

    if not results or not results["ids"]:
        return []

    return results["ids"][0]