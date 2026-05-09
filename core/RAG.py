from loaders import PDFLoader, DOCXLoader
from splitter import FixedCharacterSplitter, SentenceSplitter
from embedder import Embedder
from vectorstore import FAISSVectorStore
from generate import Generator
from typing import List
import faiss
import json
import os
import logging
# import langfuse


class RAG:
    def __init__(
        self,
        loader=None,
        splitter=None,
        embedder=None,
        vectorstore=None,
        generator=None,
    ):
        self.loader = loader or PDFLoader()
        self.splitter = splitter or SentenceSplitter()
        self.embedder = embedder or Embedder()
        self.vectorstore = vectorstore or FAISSVectorStore('.')
        self.generator = generator or Generator()

    def save(self):
        try:
            """Saves the metadata of the chunks and embeddings"""
            metadata = {
                "splitter_type": self.splitter.__class__.__name__,
                "embedder_model": self.embedder.model.__class__.__name__,
                "vectorstore_type": "FAISS",
                "config": {"chunk_size": self.splitter.chunk_size, "chunk_overlap":self.splitter.chunk_overlap}
            }

            if not os.path.exists(self.vectorstore.path):
                raise FileNotFoundError(f"The path '{self.vectorstore.path}' was not found.")

            try:
                with open(f"{self.vectorstore.path}/metadata.json", "w") as f:
                    json.dump(metadata, f)
            except Exception as e:
                logging.error(f"Error in creating metadata.json file: {e}")
        
        except Exception as e:
            logging.error(f"Couldn't save the file due to error: {e}")
            raise e


    def ingest(self, doc_path: str):

        if not os.path.exists(doc_path):
            raise FileNotFoundError(f"The path '{doc_path}' was not found.")
        
        document=self.loader.load_document(doc_path)

        chunks=self.splitter.split(document)

        index=self.vectorstore.add_document(chunks, self.embedder)
        self.save()


    def query(self, query: str):

        retrieved_chunks=self.vectorstore.retrieve_docs(query=query, embedder= self.embedder)

        response=self.generator.generate_answer(query,retrieved_chunks)

        return response
    


if __name__=="__main__":

    rag=RAG(DOCXLoader(), SentenceSplitter())

    # rag.ingest("C:\\Users\\aksha\\Downloads\\CoverLetter_Bending_Spoons.pdf")
    rag.ingest("C:\\Path.docx")
    print("Doc ingested")

    query = "What is Cloud Computing?"

    response=rag.query(query)
    print(response)