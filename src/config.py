"""Configuration loaded from environment variables."""
import os
from dotenv import load_dotenv

load_dotenv()


class Settings:
    # Azure OpenAI
    AOAI_ENDPOINT = os.environ["AZURE_OPENAI_ENDPOINT"]
    AOAI_KEY = os.environ["AZURE_OPENAI_KEY"]
    AOAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-10-21")
    EMBED_DEPLOYMENT = os.getenv("EMBED_DEPLOYMENT", "text-embedding-3-small")
    CHAT_DEPLOYMENT = os.getenv("CHAT_DEPLOYMENT", "gpt-4.1-mini")
    ENRICH_DEPLOYMENT = os.getenv("ENRICH_DEPLOYMENT", "gpt-4.1-mini")
    EMBED_DIMENSIONS = int(os.getenv("EMBED_DIMENSIONS", "1536"))

    # Azure AI Search
    SEARCH_ENDPOINT = os.environ["AZURE_SEARCH_ENDPOINT"]
    SEARCH_KEY = os.environ["AZURE_SEARCH_KEY"]
    INDEX_NAME = os.getenv("AZURE_SEARCH_INDEX", "rag-index")

    # Chunking
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1500"))
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))


settings = Settings()
