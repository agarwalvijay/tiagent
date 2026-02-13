"""Configuration management for the application."""
from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    """Application settings."""

    # LLM Provider (openai, deepseek, or groq)
    llm_provider: str = "openai"

    # OpenAI
    openai_api_key: Optional[str] = None
    openai_model: str = "gpt-4o"

    # DeepSeek
    deepseek_api_key: Optional[str] = None
    deepseek_model: str = "deepseek-chat"
    deepseek_base_url: str = "https://api.deepseek.com"

    # Groq
    groq_api_key: Optional[str] = None
    groq_model: str = "openai/gpt-oss-20b"  # or llama-3.1-70b-versatile
    groq_base_url: str = "https://api.groq.com/openai/v1"

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
