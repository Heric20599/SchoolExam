from functools import lru_cache
from typing import Any

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

VIDYANETRA_CORS_ORIGINS = (
    "https://www.vidyanetra.in",
    "https://vidyanetra.in",
)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    cors_origins: list[str] = Field(
        default=list(VIDYANETRA_CORS_ORIGINS),
        description="Comma-separated browser origins allowed for CORS (VidhyaNetra only by default).",
    )

    @field_validator("cors_origins", mode="before")
    @classmethod
    def _parse_cors_origins(cls, value: Any) -> list[str]:
        if isinstance(value, str):
            return [origin.strip() for origin in value.split(",") if origin.strip()]
        return value

    openai_api_key: str
    openai_embed_model: str = "text-embedding-3-small"
    openai_chat_model: str = "gpt-4o-mini"

    pinecone_api_key: str
    pinecone_index: str = "schoolknowledgebase"
    pinecone_cloud: str = "aws"
    pinecone_region: str = "us-east-1"
    embed_dim: int = 1536

    chunk_size: int = 800
    chunk_overlap: int = 120
    max_pdf_mb: int = 50


@lru_cache
def get_settings() -> Settings:
    return Settings()
