import sys
from unittest.mock import MagicMock

def patch_chromadb():
    mock_chromadb = MagicMock()
    mock_chromadb.PersistentClient.return_value = MagicMock()
    sys.modules['chromadb'] = mock_chromadb
    sys.modules['chromadb.config'] = MagicMock()
    sys.modules['chromadb.utils'] = MagicMock()
    sys.modules['chromadb.utils.embedding_functions'] = MagicMock()
    sys.modules['chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction'] = MagicMock()

patch_chromadb()
