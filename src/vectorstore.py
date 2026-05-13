import faiss # type: ignore
from faiss import IndexFlatL2, Index # type: ignore
import numpy as np
from typing import List, Tuple
from embedder import Embedder
from loaders import Document
from exceptions import VectorStoreError
import json
from dataclasses import asdict
import os


class FAISSVectorStore:
    """
    Vector store implementation using FAISS.

    Supports indexing document embeddings and retrieving
    semantically similar documents using vector similarity search.
    """
    def __init__(self, path: str):
        self.path=path
        os.makedirs(self.path, exist_ok=True)


    def save_index(self, index: faiss.Index):
        index_path = os.path.join(self.path, "index.faiss")
        faiss.write_index(index, index_path)


    def load_index(self):      
        index_path = os.path.join(self.path, "index.faiss")

        if not os.path.exists(index_path):
            raise FileNotFoundError(f"Index file not found at '{index_path}'")  
        return faiss.read_index(index_path)

    #Step 4: Store the embeddings
    def add_document(self, chunks: List[Document], embedder: Embedder)->faiss.Index:
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
            # Fix for chunks.json getting overwritten with every ingestion. Now new chunks will be added not overwritten.
            fname=os.path.join(self.path,'chunks.json')

            if not os.path.isfile(fname) or os.path.getsize(fname) == 0:
                with open(fname, "w") as f:
                    f.write(json.dumps([asdict(chunk) for chunk in chunks], indent=3))
            else:
                with open(fname,'r') as f:
                    new_chunks=json.load(f)

                new_chunks.extend([asdict(chunk) for chunk in chunks])
            
                with open(fname, "w") as f:
                    f.write(json.dumps(new_chunks, indent=3))


            texts = [doc.content for doc in chunks]

            if not any(text.strip() for text in texts):
                raise VectorStoreError(f"No text in the chunks")

            embeddings=embedder.embed_document(texts)

            if embeddings.size==0:
                raise VectorStoreError(f"Generated embeddings are empty")
            
            index=faiss.IndexFlatIP(len(embeddings[0]))

            index.add(embeddings)

            self.save_index(index)

            return index
        
        except VectorStoreError:
            raise
        
        except Exception as e:
            raise VectorStoreError(f"Error in indexing chunks: {e}") from e
        

    #Step 5: Retrieve documents
    def retrieve_docs(self, query: str, embedder: Embedder, k = 4, **kwargs)->List[Tuple[Document, float]]:
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
            
            chunk_path = os.path.join(self.path, "chunks.json")

            if not os.path.exists(chunk_path):
                raise FileNotFoundError(f"The path '{chunk_path}' was not found.")
            
            source=kwargs.get('source')

            with open(chunk_path, 'r') as f:
                chunks=json.load(f)
                # print(chunks)
                # chunks = [Document(chunk['content'], chunk['metadata']) for chunk in chunks]
            query_embeddings=embedder.embed_document([query])

            retrieved_chunks=[]

            index=self.load_index()



            if query_embeddings.shape[1] != index.d:
                raise VectorStoreError(
                    "Query embedding dimension does not match index dimension."
                )

            if index.ntotal == 0:
                raise VectorStoreError(f"Nothing indexed.")
            
            D,I=index.search(query_embeddings,k=k)

            for idx, dist in zip(I[0], D[0]):
                if idx==-1:
                    continue
                retrieved_chunks.append((Document(chunks[idx]['content'], chunks[idx]['metadata']),dist))

            print(retrieved_chunks)
            return retrieved_chunks 
        
        except VectorStoreError:
            raise
        except Exception as e:
            raise VectorStoreError(f"Error in retrieving chunks {e}") from e