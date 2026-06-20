"""Central application configuration.

All settings are sourced from environment variables (loaded from `.env`).
See `.env.example` for the full list of supported variables.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field, computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict

VectorBackend = Literal["pgvector", "pinecone", "qdrant", "weaviate", "milvus"]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
        populate_by_name=True,
    )

    # ----- General -----
    app_name: str = "rag-url-ingestion"
    log_level: str = "INFO"
    environment: str = Field(default="development")

    # ----- OpenAI or GROQ / Embeddings -----
    # openai_api_key: str = Field(default="", alias="OPENAI_API_KEY")
    openai_api_key: str = Field(default="", alias="GROQ_API_KEY")
    embedding_model: str = "text-embedding-3-large"
    # text-embedding-3-large supports the `dimensions` param up to 3072.
    embedding_dimension: int = 2048
    embedding_batch_size: int = 128
    embedding_max_concurrency: int = 8
    embedding_max_retries: int = 6
    embedding_timeout_seconds: float = 60.0

    # ----- Crawler -----
    crawl_max_depth: int = 1          # 0 = only the seed URL
    crawl_max_pages: int = 50
    crawl_same_domain_only: bool = True
    crawl_concurrency: int = 5
    crawl_timeout_seconds: int = 30
    crawl_min_words: int = 20         # drop near-empty pages
    crawl_user_agent: str = "rag-url-system/1.0 (+https://example.com/bot)"

    # ----- Chunking -----
    chunk_target_tokens: int = 1000
    chunk_min_tokens: int = 800
    chunk_max_tokens: int = 1200
    chunk_overlap_tokens: int = 175
    tokenizer_encoding: str = "cl100k_base"

    # ----- Vector store -----
    vector_backend: VectorBackend = "pgvector"
    vector_namespace: str = "default"
    vector_collection: str = "rag_chunks"
    upsert_batch_size: int = 100

    # Pinecone
    pinecone_api_key: str = ""
    pinecone_index: str = "rag-chunks"
    pinecone_cloud: str = "aws"
    pinecone_region: str = "us-east-1"

    # Qdrant
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str = ""

    # Weaviate
    weaviate_url: str = "http://localhost:8080"
    weaviate_api_key: str = ""

    # Milvus
    milvus_uri: str = "http://localhost:19530"
    milvus_token: str = ""

    # ----- Postgres (metadata + pgvector) -----
    db_name_local: str = Field(default="rag", alias="DB_NAME_LOCAL")
    db_username_local: str = Field(default="postgres", alias="DB_USERNAME_LOCAL")
    db_password_local: str = Field(default="postgres", alias="DB_PASSWORD_LOCAL")
    db_host_local: str = Field(default="localhost", alias="DB_HOST_LOCAL")
    db_port_local: int = Field(default=5432, alias="DB_PORT_LOCAL")

    @computed_field  # type: ignore[prop-decorator]
    @property
    def async_database_url(self) -> str:
        return (
            f"postgresql+asyncpg://{self.db_username_local}:{self.db_password_local}"
            f"@{self.db_host_local}:{self.db_port_local}/{self.db_name_local}"
        )

    @computed_field  # type: ignore[prop-decorator]
    @property
    def sync_database_url(self) -> str:
        return (
            f"postgresql+psycopg://{self.db_username_local}:{self.db_password_local}"
            f"@{self.db_host_local}:{self.db_port_local}/{self.db_name_local}"
        )


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
