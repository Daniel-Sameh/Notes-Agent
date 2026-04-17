import chromadb
from chromadb.config import Settings
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from .models import Note
from ..config import settings

DB_URL=settings.chroma_url

# Initialize ChromaDB persistent client pointing to our local folder
client = chromadb.PersistentClient(path=DB_URL)

# Use the lightweight and fast all-MiniLM-L6-v2 embedding model (local)
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

def delete_from_vector_db(user_id: str, id: str):
    """Removes a note's embedding from the vector database by ID."""
    collection.delete(
        ids=[str(id)],
        where={"user_id": user_id}
    )

def semantic_search_vector_db(user_id: str, query: str, limit: int = 5) -> list[str]:
    """
    Performs a semantic similarity search on the queries.
    Returns a list of matching note IDs (as strings).
    """
    results = collection.query(
        query_texts=[query],
        n_results=limit,
        where={"user_id": user_id}
    )
    
    if not results or not results["ids"]:
        return []
    
    # results["ids"] is a list of lists: [['id1', 'id2', ...]]
    return results["ids"][0]

