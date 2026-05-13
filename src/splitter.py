from abc import ABC, abstractmethod
from typing import Optional,List
from loaders import Document
import hashlib
from exceptions import SplitterError

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
    def __init__(self,chunk_size: int = 1000,chunk_overlap: int = 200) -> None:
        self.chunk_size= chunk_size
        self.chunk_overlap= chunk_overlap

    def split(self, document: List[Document], **kwargs)->List[Document]:
        """Provide chunk size = x and chunk overlap = y"""
        chunk_size = kwargs.get("chunk_size", 500)
        chunk_overlap = kwargs.get("chunk_overlap", 100)
        if self.chunk_overlap>=self.chunk_size:
            raise SplitterError("Chunk overlap cannot be greater than or equal to chunk size")
        
        if not document:
            raise SplitterError("Empty document")
        chunks=[]
        try:
            for doc in document:
                for idx,start in enumerate(range(0,len(doc.content),self.chunk_size - self.chunk_overlap)):
                    end=min(start+self.chunk_size, len(doc.content))
                    chunk_text=doc.content[start:end]
                    if not chunk_text:
                        continue
                    metadata = doc.metadata.copy()
                    metadata["chunk_id"]=idx
                    metadata['chunk_checksum']=hashlib.md5(chunk_text.encode()).hexdigest()
                    # metadata["chunk_size"]=len(chunk_text)
                    chunk=Document(chunk_text,metadata)
                    chunks.append(chunk)
            return chunks
        except SplitterError:
            raise
        except Exception as e:
            raise SplitterError(f"Error in splitting") from e            


class SentenceSplitter(Splitter):
    """
    Split documents using sentence-aware chunking.

    Default Args:
        chunk_size = 1000 
        chunk_overlap = 200

    """
    def __init__(self,chunk_size: int = 1000,chunk_overlap: int = 200) -> None:
        self.chunk_size= chunk_size
        self.chunk_overlap= chunk_overlap

    def split(self, document: List[Document], **kwargs)->List[Document]:
        """    
        Args:
            document: List of Document objects.

        Returns:
            List of chunked Document objects with metadata."""
        #Create chunks(Sentence based chunking)

        chunk_size = kwargs.get("chunk_size", 500)
        chunk_overlap = kwargs.get("chunk_overlap", 100)

        if self.chunk_overlap>=self.chunk_size:
            raise SplitterError("Chunk overlap cannot be greater than chunk size")

        if not document:
            raise SplitterError("Empty document")

        chunks = []
        try:
            for doc in document: 
                sentences = doc.content.split('.')

                current_chunk = []
                current_length = 0
                idx=0

                for sentence in sentences:
                    metadata = doc.metadata.copy()  
                    sentence = sentence.strip()
                    if not sentence:
                        continue
                    
                    sentence_length = len(sentence)

                    if current_length + sentence_length <= self.chunk_size:
                        current_chunk.append(sentence)
                        current_length += sentence_length
                    else:
                        chunk_text='. '.join(current_chunk) + '.'
                        if not chunk_text.strip():
                            continue
                        metadata["chunk_id"]=idx
                        metadata['chunk_checksum']=hashlib.md5(chunk_text.encode()).hexdigest()
                        # Save chunk
                        chunks.append(Document(chunk_text, metadata))

                        idx += 1

                        # Start new chunk with overlap
                        overlap_chunk = []
                        overlap_length = 0

                        # Add last sentences until overlap satisfied
                        for s in reversed(current_chunk):
                            if overlap_length + len(s) <= self.chunk_overlap:
                                overlap_chunk.insert(0, s)
                                overlap_length += len(s)
                            else:
                                break

                        current_chunk = overlap_chunk + [sentence]
                        current_length = sum(len(s) for s in current_chunk)

                # Add last chunk
                if current_chunk:
                    metadata=doc.metadata.copy()
                    chunk_text='. '.join(current_chunk) + '.'
                    if not chunk_text.strip():
                        continue
                    metadata["chunk_id"]=idx+1
                    metadata['chunk_checksum']=hashlib.md5(chunk_text.encode()).hexdigest()
                    
                    chunks.append(Document(chunk_text, metadata))

            return chunks
        except SplitterError:
            raise
        except Exception as e:
            raise SplitterError(f"Error in splitting") from e