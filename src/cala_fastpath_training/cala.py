from __future__ import annotations

import os
from urllib.parse import urlparse

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
        
        # Validate base_url to prevent credential leakage to untrusted endpoints
        validated_url = self._validate_base_url(base_url)
        
        self._client = httpx.Client(
            base_url=validated_url,
            headers={"X-API-KEY": api_key, "Accept": "application/json"},
            timeout=timeout,
            transport=transport,
        )
    
    @staticmethod
    def _validate_base_url(base_url: str) -> str:
        """
        Validate the base URL to prevent credential leakage.
        
        Ensures the URL uses HTTPS and has a valid hostname to prevent
        API keys from being sent to untrusted or plaintext endpoints.
        
        Args:
            base_url: The base URL to validate
            
        Returns:
            The validated and normalized base URL (with trailing slash removed)
            
        Raises:
            ValueError: If the URL scheme is not HTTPS or hostname is invalid
        """
        if not base_url:
            raise ValueError("base_url cannot be empty")
        
        parsed = urlparse(base_url)
        
        # Enforce HTTPS to prevent credential exposure over plaintext
        if parsed.scheme != "https":
            raise ValueError(
                f"base_url must use HTTPS scheme to protect API credentials, got {parsed.scheme!r}"
            )
        
        # Ensure a valid hostname exists
        if not parsed.hostname:
            raise ValueError("base_url must have a valid hostname")
        
        # Return normalized URL without trailing slash
        return base_url.rstrip("/")

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
