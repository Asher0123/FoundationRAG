class LoaderError(Exception):
    """Raised when document loading fails."""


class SplitterError(Exception):
    """Raised when splitting fails."""


class EmbeddingError(Exception):
    """Raised when embedding generation fails."""


class VectorStoreError(Exception):
    """Raised when vector retrieval fails."""


class GenerationError(Exception):
    """Raised when LLM generation fails."""
    