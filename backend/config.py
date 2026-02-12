"""Configuration management for the application."""
from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    """Application settings."""

    # OpenAI
    openai_api_key: str
    openai_model: str = "gpt-4o"

    # ChromaDB
    chroma_persist_dir: str = "./chroma_db"
    chroma_collection_name: str = "ti_datasheets"

    # Application
    backend_host: str = "0.0.0.0"
    backend_port: int = 8000
    frontend_url: str = "http://localhost:3000"

    # Embedding
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    # LangChain
    langchain_tracing_v2: bool = False
    langchain_api_key: Optional[str] = None

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()
