from contextlib import suppress

from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec

from app.config import Settings


def build_openai_client(settings: Settings) -> OpenAI:
    return OpenAI(api_key=settings.openai_api_key)


def build_pinecone_client(settings: Settings) -> Pinecone:
    return Pinecone(api_key=settings.pinecone_api_key)


def ensure_pinecone_index(settings: Settings, client: Pinecone) -> None:
    existing = [idx.name for idx in client.list_indexes()]
    if settings.pinecone_index not in existing:
        client.create_index(
            name=settings.pinecone_index,
            dimension=settings.embed_dim,
            metric="cosine",
            spec=ServerlessSpec(cloud=settings.pinecone_cloud, region=settings.pinecone_region),
        )
    with suppress(Exception):
        client.describe_index(settings.pinecone_index)
