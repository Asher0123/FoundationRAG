from loaders import Loader
from splitter import FixedCharacterSplitter, SentenceSplitter
from embedder import Embedder
from vectorstore import FAISSVectorStore
from generate import Generator
from typing import List
import faiss
import json
import os
import logging
from langfuse import Langfuse
from dotenv.main import set_key, load_dotenv, find_dotenv


load_dotenv(dotenv_path=".\\core\\.env")

langfuse=Langfuse(public_key=os.getenv("LANGFUSE_PUBLIC_KEY"), 
                  secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
                  base_url=os.getenv("LANGFUSE_BASE_URL"))


class RAG:
    def __init__(
        self,
        loader=None,
        splitter=None,
        embedder=None,
        vectorstore=None,
        generator=None,
    ):
        self.loader = loader or Loader()
        self.splitter = splitter or SentenceSplitter()
        self.embedder = embedder or Embedder()
        self.vectorstore = vectorstore or FAISSVectorStore('..\\vectorstore')
        self.generator = generator or Generator()

    def get_metric_type(self, metric: int) -> str: # Tells which type of search is used for FAISS
        if metric==0:
            return "IP"
        elif metric == 1:
            return "L2"
        elif metric == 2:
            return "L1"
        elif metric == 3:
            return "Linf"
        elif metric == 4:
            return "Lp"

    def save(self, index):
        try:
            """Saves the metadata of the chunks and embeddings"""
            metadata = {
                "splitter_type": self.splitter.__class__.__name__,
                "embedder_model": self.embedder.model.__class__.__name__,
                "vectorstore_type": "FAISS",
                "similarity_type":self.get_metric_type(index.metric_type),
                "embedding dimension": index.d,
                "config": {"chunk_size": self.splitter.chunk_size, "chunk_overlap":self.splitter.chunk_overlap}
            }

            if not os.path.exists(self.vectorstore.path):
                raise FileNotFoundError(f"The path '{self.vectorstore.path}' was not found.")

            try:
                with open(f"{self.vectorstore.path}/metadata.json", "w") as f:
                    json.dump(metadata, f, indent=3)
            except Exception as e:
                logging.error(f"Error in creating metadata.json file: {e}")
        
        except Exception as e:
            logging.error(f"Couldn't save the file due to error: {e}")
            raise e


    def ingest(self, doc_path: str):

        if not os.path.exists(doc_path):
            raise FileNotFoundError(f"The path '{doc_path}' was not found.")
        
        document=self.loader.load(doc_path)

        chunks=self.splitter.split(document)

        index=self.vectorstore.add_document(chunks, self.embedder)
        self.save(index)


    def query(self, query: str, k: int = 4,observe=True):
        try:
            if not query.strip():
                raise Exception 

            if observe:
                with langfuse.start_as_current_observation(name="QnA",input={"query":query}) as trace:
                
                    retrieved_chunks=self.vectorstore.retrieve_docs(query=query, embedder=self.embedder, k=k)

                    with langfuse.start_as_current_observation(name="Retrieval",input={"query":query}) as retrieval_span:

                        formatted_chunks = [{
                                                "content": chunk.content,
                                                "metadata": chunk.metadata,
                                                "score": score
                                            }
                                            for chunk, score in retrieved_chunks
        ]

                        retrieval_span.update(output={"retrieved_chunks": formatted_chunks})
                        
                    response=self.generator.generate_answer(query,retrieved_chunks, langfuse)

                    usage=getattr(response,"usage_metadata",{})

                    trace.update(output={"response": str(response.content)},metadata={
                                        "input": usage.get("input_tokens"),
                                        "output": usage.get("output_tokens"),
                                        "total": usage.get("total_tokens")})
                    
                langfuse.flush()
                return response.content
            
            retrieved_chunks=self.vectorstore.retrieve_docs(query=query, embedder= self.embedder, k=k)

            response=self.generator.generate_answer(query,retrieved_chunks)

            return response.content
        
        except Exception as e:
            logging.error(f"Error {e}")
            raise
            


if __name__=="__main__":

    rag=RAG(splitter=SentenceSplitter())

    # rag.ingest("C:\\Path\\file.pdf")
    # rag.ingest("C:\\ Path")
    print("Doc ingested")
    
    query = "Why does the candidate want to work in Bending Spoons?"
    
    response=rag.query(query,k=1)

    print(response)