from __future__ import annotations

import os

import httpx
from pydantic import ValidationError

from .models import CalaQueryResponse


class CalaError(RuntimeError):
    pass


class CalaClient:
    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.cala.ai/v1",
        timeout: float = 180,
        transport: httpx.BaseTransport | None = None,
    ):
        if not api_key:
            raise ValueError("CALA_API_KEY is required")
        self._client = httpx.Client(
            base_url=base_url.rstrip("/"),
            headers={"X-API-KEY": api_key, "Accept": "application/json"},
            timeout=timeout,
            transport=transport,
        )

    def __enter__(self) -> CalaClient:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def close(self) -> None:
        self._client.close()

    @classmethod
    def from_environment(cls) -> CalaClient:
        return cls(
            api_key=os.environ.get("CALA_API_KEY", ""),
            base_url=os.environ.get("CALA_BASE_URL", "https://api.cala.ai/v1"),
        )

    def knowledge_query(self, query: str, *, return_entities: bool = True) -> CalaQueryResponse:
        try:
            response = self._client.post(
                "/knowledge/query",
                json={"input": query, "return_entities": return_entities},
            )
            response.raise_for_status()
            return CalaQueryResponse.model_validate(response.json())
        except httpx.HTTPStatusError as exc:
            raise CalaError(
                f"Cala knowledge_query returned {exc.response.status_code}: {exc.response.text}"
            ) from exc
        except httpx.HTTPError as exc:
            raise CalaError(f"Cala knowledge_query failed: {exc}") from exc
        except (ValueError, ValidationError) as exc:
            raise CalaError(f"Cala knowledge_query returned invalid JSON: {exc}") from exc
