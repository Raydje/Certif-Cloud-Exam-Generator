from langchain_google_genai import GoogleGenerativeAIEmbeddings
from pydantic import SecretStr
from configs.setting import get_settings



def get_google_embeddings() -> GoogleGenerativeAIEmbeddings:
    """
    Instantiates the current Google Gemini embedding model for the ingestion plane.
    Uses Matryoshka Representation Learning (MRL) to truncate dimensions 
    and optimize Pinecone Serverless costs.
    """
    settings = get_settings()
    return GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=SecretStr(settings.gemini_api_key),
        task_type="RETRIEVAL_DOCUMENT",
        # Truncate to 768 to optimize Pinecone Serverless costs
        # while retaining high semantic performance.
        dimensions=768 
    )