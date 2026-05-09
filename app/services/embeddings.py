from __future__ import annotations

import time

from openai import OpenAI

from app.errors import UpstreamError


def embed_texts(client: OpenAI, model: str, texts: list[str], batch_size: int = 100) -> list[list[float]]:
    vectors: list[list[float]] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        for attempt in range(2):
            try:
                resp = client.embeddings.create(model=model, input=batch)
                vectors.extend([item.embedding for item in resp.data])
                break
            except Exception as exc:  # pragma: no cover - network path
                if attempt == 1:
                    raise UpstreamError("Embedding generation failed", {"reason": str(exc)}) from exc
                time.sleep(1.0)
    return vectors
