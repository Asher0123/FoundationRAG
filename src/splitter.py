from abc import ABC, abstractmethod
from typing import List
from loaders import Document
import hashlib
from exceptions import SplitterError
import re


class Splitter(ABC):

    @abstractmethod
    def split(self, document: List[Document], **kwargs) -> List[Document]:
        pass


class FixedCharacterSplitter(Splitter):
    """
    Split documents using fixed-size character chunking.

    Args:
        document: List of Document objects.
        chunk_size: Maximum chunk size, default is 1000.
        chunk_overlap: Overlap between chunks,default is 200.

    Returns:
        List of chunked Document objects with metadata.
    """
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200) -> None:
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split(self, document: List[Document], **kwargs) -> List[Document]:

        if self.chunk_overlap >= self.chunk_size:
            raise SplitterError("Chunk overlap cannot be greater than or equal to chunk size")

        if not document:
            raise SplitterError("Empty document")

        try:
            chunks = []
            step = self.chunk_size - self.chunk_overlap
            for doc in document:
                for idx, start in enumerate(range(0, len(doc.content), step)):
                    end = min(start + self.chunk_size, len(doc.content))
                    chunk_text = doc.content[start:end].strip()

                    if not chunk_text:
                        continue

                    metadata = doc.metadata.copy()
                    metadata["chunk_id"] = idx
                    metadata["chunk_checksum"] = hashlib.md5(chunk_text.encode()).hexdigest()

                    chunks.append(Document(chunk_text, metadata))

            return chunks

        except SplitterError:
            raise

        except Exception as e:
            raise SplitterError(f"Error in splitting: {e}") from e


class SentenceSplitter(Splitter):
    """
    Split documents using sentence-aware chunking.

    Default Args:
        chunk_size = 1000 
        chunk_overlap = 200

    """
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200) -> None:

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split(self, document: List[Document], **kwargs) -> List[Document]:

        if self.chunk_overlap >= self.chunk_size:
            raise SplitterError("Chunk overlap cannot be greater than or equal to chunk size")

        if not document:
            raise SplitterError("Empty document")

        chunks = []
        try:
            idx = 0
            for doc in document:
                sentences = re.split(r'(?<=[.!?])\s+', doc.content)

                current_chunk = []
                current_length = 0

                for sentence in sentences:

                    sentence = sentence.strip()

                    if not sentence:
                        continue

                    sentence_length = len(sentence)

                    # Handle oversized sentence
                    if sentence_length > self.chunk_size:

                        # Flush existing chunk first
                        if current_chunk:

                            chunk_text = " ".join(current_chunk).strip()

                            if chunk_text:

                                metadata = doc.metadata.copy()
                                metadata["chunk_id"] = idx
                                metadata["chunk_checksum"] = hashlib.md5(chunk_text.encode()).hexdigest()

                                chunks.append(Document(chunk_text, metadata))
                            idx+=1

                            current_chunk = []
                            current_length = 0

                        # Split oversized sentence manually
                        step = self.chunk_size - self.chunk_overlap

                        for start in range(0, sentence_length, step):
                            end = min(start + self.chunk_size, sentence_length)
                            chunk_text = sentence[start:end].strip()

                            if not chunk_text:
                                continue

                            metadata = doc.metadata.copy()
                            metadata["chunk_id"] = idx
                            metadata["chunk_checksum"] = hashlib.md5(chunk_text.encode() ).hexdigest()

                            chunks.append(Document(chunk_text, metadata))
                            idx += 1

                        continue

                    # Account for joining space
                    additional_length = (sentence_length if not current_chunk else sentence_length + 1)

                    # Add sentence if within limit
                    if current_length + additional_length <= self.chunk_size:

                        current_chunk.append(sentence)
                        current_length += additional_length

                    else:

                        # Save current chunk
                        chunk_text = " ".join(current_chunk).strip()

                        if chunk_text:

                            metadata = doc.metadata.copy()
                            metadata["chunk_id"] = idx
                            metadata["chunk_checksum"] = hashlib.md5(chunk_text.encode()).hexdigest()
                            chunks.append(Document(chunk_text, metadata))
                            idx += 1

                        # Create overlap
                        overlap_chunk = []
                        overlap_length = 0

                        for s in reversed(current_chunk):
                            s_length = len(s)
                            additional_overlap = (s_length if not overlap_chunk else s_length + 1)

                            if (
                                overlap_length + additional_overlap
                                <= self.chunk_overlap
                            ):

                                overlap_chunk.insert(0, s)
                                overlap_length += additional_overlap

                            else:
                                break

                        # Start new chunk
                        current_chunk = overlap_chunk + [sentence]

                        current_length = sum(len(s) for s in current_chunk) + max(len(current_chunk) - 1, 0)

                # Add remaining chunk
                if current_chunk:

                    chunk_text = " ".join(current_chunk).strip()

                    if chunk_text:
                        metadata = doc.metadata.copy()
                        metadata["chunk_id"] = idx
                        metadata["chunk_checksum"] = hashlib.md5(chunk_text.encode()).hexdigest()

                        chunks.append(Document(chunk_text, metadata))

                        idx+=1

            return chunks
        except SplitterError:
            raise
        except Exception as e:
            raise SplitterError(f"Error while splitting document: {e}") from e