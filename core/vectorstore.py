import faiss # type: ignore
from faiss import IndexFlatL2 # type: ignore
import numpy as np
from typing import List, Tuple
from embedder import Embedder
from loaders import Document
from exceptions import VectorStoreError
import json
from dataclasses import asdict



class FAISSVectorStore:
    """
    Vector store implementation using FAISS.

    Supports indexing document embeddings and retrieving
    semantically similar documents using vector similarity search.
    """
    def __init__(self, path: str):
        self.path=path


    def save_index(self, index: IndexFlatL2):
        faiss.write_index(index, f"{self.path}/index.faiss")

    def load_index(self):
        return faiss.read_index(f"{self.path}/index.faiss")

    #Step 4: Store the embeddings
    def add_document(self, chunks: List[Document], embedder: Embedder)->IndexFlatL2:
        """
        Generate embeddings for documents and add them to a FAISS index.

        Args:
            chunks:
                List of Document objects to index.

            embedder:
                Embedder instance used to generate vector embeddings.

        Returns:
            FAISS IndexFlatL2 index containing document embeddings.
        """
        try:

            with open(f"{self.path}/chunks.json", "w") as f:
                json.dump([asdict(chunk) for chunk in chunks], f)


            texts = [doc.content for doc in chunks]

            if not any(text.strip() for text in texts):
                raise VectorStoreError(f"No text in the chunks")

            embeddings=embedder.embed_document(texts)

            if embeddings.size==0:
                raise VectorStoreError(f"Generated embeddings are empty")
            
            index=faiss.IndexFlatL2(len(embeddings[0]))

            index.add(embeddings)

            self.save_index(index)

            return index
        
        except VectorStoreError:
            raise
        
        except Exception as e:
            raise VectorStoreError(f"Error in indexing chunks") from e
        

    #Step 5: Retrieve documents
    def retrieve_docs(self, query: str, embedder: Embedder, k = 4)->List[Tuple[Document, float]]:
        """
        Retrieve the top-k most semantically similar documents
        for a given query using FAISS vector similarity search.

        Args:
            query:
                Input query string used for retrieval.

            k:
                Number of similar documents to retrieve.

            embedder:
                Embedder instance used to generate query embeddings.

        Returns:
            List of retrieved Document objects ranked by similarity.

        Raises:
            VectorStoreError:
                Raised when retrieval fails due to invalid inputs,
                empty indexes, embedding failures, or FAISS errors.
        """
        try:
            if k<=0:
                raise VectorStoreError(f"k value cannot be less than 0. ")
            if not query.strip() or len(query)<=3:
                raise VectorStoreError(f"Query cannot be empty or less than 2 characters")
            
            with open(f'{self.path}/chunks.json','r') as f:
                chunks=json.load(f)
                chunks = [Document(**chunk) for chunk in chunks]
            
            query_embeddings=embedder.embed_document([query])

            retrieved_chunks=[]

            index=self.load_index()

            if index.ntotal == 0:
                raise VectorStoreError(f"Nothing indexed.")
            
            D,I=index.search(query_embeddings,k=k)

            for idx, dist in zip(I[0], D[0]):
                if idx==-1:
                    continue
                retrieved_chunks.append((chunks[idx],dist))

            return retrieved_chunks 
        
        except VectorStoreError:
            raise
        except Exception as e:
            raise VectorStoreError(f"Error in retrieving chunks") from e
        

