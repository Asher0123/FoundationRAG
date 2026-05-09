from langchain_aws import ChatBedrockConverse
import os
from typing import List,Optional, Tuple
from langfuse import LangfuseSpan
from dotenv.main import set_key, load_dotenv, find_dotenv
from loaders import Document
from exceptions import GenerationError

load_dotenv()

class Generator:
    """
    Handles answer generation using a language model.

    Responsible for constructing prompts from retrieved
    document context and generating responses using
    an LLM backend.
    """
    def __init__(self, model=None):
        self.model= model or ChatBedrockConverse(model_id = "us.anthropic.claude-opus-4-1-20250805-v1:0", region_name='us-east-1',
                                aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
                                aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"))

    def generate_answer(self,query: str, retrieved_chunks: List[Tuple[Document, float]], trace:Optional[LangfuseSpan]=None):
        """
        Generate an answer for a query using retrieved documents.

        Args:
            query:
                User query string.

            retrieved_chunks:
                List of retrieved Document objects used as context.

            trace:
                Optional Langfuse tracing span for observability.

        Returns:
            Generated response content as a string.

        Raises:
            GenerationError:
                Raised when generation fails due to invalid inputs,
                model issues, or LLM invocation failures.
        """
        try:
            if not query.strip():
                raise GenerationError("Query cannot be empty")

            if not retrieved_chunks:
                raise GenerationError("No chunks retrieved.")
            
            context = "\n".join([doc.content for doc, _ in retrieved_chunks])

            if not context.strip():
                raise GenerationError("Empty chunks")

            prompt=f"""Answer the query {query} from the provided set of texts: {context}"""

            if not hasattr(self.model, "invoke"):
                raise GenerationError("Missing invoke method")
            
            if trace:
                span = trace.span(
                name="generation",
                input={"prompt": prompt}
            )
                print("Generating answer")

                response=self.model.invoke(prompt)
                span.end(output={"response": str(response.content)})
                return response.content
            
            response=self.model.invoke(prompt)
            return response.content
        except GenerationError:
            raise
        except Exception as e:
            raise GenerationError(f"Error in generating answer {e}") from e