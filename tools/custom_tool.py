from crewai.tools import BaseTool
from typing import Type
from pydantic import BaseModel, Field
import json
from chromadb.config import Settings
import chromadb
from sentence_transformers import SentenceTransformer

model= SentenceTransformer("all-MiniLM-L6-v2")  # Load embedding model
class RAGToolInput(BaseModel):
    """Input schema for MyCustomTool."""
    query: str = Field(..., description="query to search in the database")
    k: int = Field(3, description="Number of top results to retrieve")

class RAGTool(BaseTool):
    name: str = "RAGTool"
    description: str = (
        "This tool retrieves the top-k relevant policy clauses from the insurance document "
    "based on a natural language query. It can be used multiple times to explore different aspects "
    "of the policy — such as exclusions, treatment coverage, deductible rules, and hospitalization terms."
    )
    args_schema: Type[BaseModel] = RAGToolInput
    def _run(self, query: str,k:int) -> list[str]:
    
        # Load the preprocessed chunks from disk
        collection_name="chunks"
        # all_chunks_path="all_chunks.json"
        # with open(all_chunks_path, "r", encoding="utf-8") as f:
        #     all_chunks = json.load(f)

        # Initialize ChromaDB client
        PERSIST_DIR = "./chroma_db"
        client = chromadb.PersistentClient(
            path=PERSIST_DIR
        )
        collection = client.get_or_create_collection(name=collection_name)

        # Encode the query
        query_embedding = model.encode(query).tolist()

        # Query ChromaDB
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=k,
            include=["documents","metadatas", "distances"]
        )
        documents = results.get("documents", [[]])[0]  # Safely access the first list of docs

        return documents  # This will be a list[str]



   
