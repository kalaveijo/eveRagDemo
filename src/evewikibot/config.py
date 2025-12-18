"""Configuration for the Eve Online RAG Bot."""

import os
from dataclasses import dataclass


@dataclass
class OllamaConfig:
    """Configuration for Ollama model."""
    
    base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    model: str = os.getenv("OLLAMA_MODEL", "gemma3:1b")
    temperature: float = float(os.getenv("OLLAMA_TEMPERATURE", "0.7"))
    embedding_model: str = os.getenv("OLLAMA_EMBEDDING_MODEL", "embeddinggemma:300m")


@dataclass
class PgVectorConfig:
    """Configuration for PgVector database."""
    
    host: str = os.getenv("PGVECTOR_HOST", "localhost")
    port: int = int(os.getenv("PGVECTOR_PORT", "5432"))
    database: str = os.getenv("PGVECTOR_DATABASE", "evewiki")
    user: str = os.getenv("PGVECTOR_USER", "")
    password: str = os.getenv("PGVECTOR_PASSWORD", "")
    collection_name: str = os.getenv("PGVECTOR_COLLECTION", "eve_wiki")
    embedding_dim: int = int(os.getenv("PGVECTOR_EMBEDDING_DIM", "384"))
    top_k: int = int(os.getenv("PGVECTOR_TOP_K", "5"))


@dataclass
class BotConfig:
    """Overall bot configuration."""
    
    ollama: OllamaConfig
    pgvector: PgVectorConfig
    debug: bool = os.getenv("DEBUG", "false").lower() == "true"
    
    @classmethod
    def from_env(cls) -> "BotConfig":
        """Create config from environment variables."""
        return cls(
            ollama=OllamaConfig(),
            pgvector=PgVectorConfig(),
        )
