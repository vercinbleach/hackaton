from __future__ import annotations

import os
import time
from pathlib import Path
from urllib.parse import parse_qs, quote, urlencode, urlparse, urlsplit

import httpx
from pydantic import ValidationError

from .models import EvaluationStart, JsonObject, UploadedDataset, UploadReservation


class PioneerError(RuntimeError):
    pass


DEFAULT_DATASET_CONTENT_TYPE = "application/x-ndjson"


class PioneerClient:
    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.pioneer.ai",
        timeout: float = 60,
        transport: httpx.BaseTransport | None = None,
    ):
        if not api_key:
            raise ValueError("PIONEER_API_KEY is required")
        validated_url = self._validate_base_url(base_url)

        self._client = httpx.Client(
            base_url=validated_url,
            headers={"X-API-Key": api_key, "Accept": "application/json"},
            timeout=timeout,
            transport=transport,
        )
        self._upload_client = httpx.Client(
            timeout=timeout,
            transport=transport,
            follow_redirects=False,
        )

    @staticmethod
    def _validate_base_url(base_url: str) -> str:
        """Validate a credential-bearing Pioneer API endpoint."""
        if not base_url:
            raise ValueError("base_url cannot be empty")
        parsed = urlparse(base_url)
        if parsed.scheme != "https":
            raise ValueError("base_url must use HTTPS")
        if not parsed.hostname:
            raise ValueError("base_url must have a valid hostname")
        if parsed.username is not None or parsed.password is not None:
            raise ValueError("base_url must not contain user information")
        if parsed.query or parsed.fragment:
            raise ValueError("base_url must not contain a query or fragment")
        return base_url.rstrip("/")

    def __enter__(self) -> PioneerClient:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def close(self) -> None:
        self._upload_client.close()
        self._client.close()

    @classmethod
    def from_environment(cls) -> PioneerClient:
        return cls(
            api_key=os.environ.get("PIONEER_API_KEY", ""),
            base_url=os.environ.get("PIONEER_BASE_URL", "https://api.pioneer.ai"),
        )

    def _request(
        self,
        method: str,
        path: str,
        *,
        payload: JsonObject | None = None,
    ) -> JsonObject:
        try:
            response = self._client.request(method, path, json=payload)
            response.raise_for_status()
            if not response.content:
                return {}
            value = response.json()
        except httpx.HTTPStatusError as exc:
            detail = exc.response.text
            raise PioneerError(
                f"Pioneer {method} {path} returned {exc.response.status_code}: {detail}"
            ) from exc
        except httpx.HTTPError as exc:
            raise PioneerError(f"Pioneer {method} {path} failed: {exc}") from exc
        except ValueError as exc:
            raise PioneerError(f"Pioneer returned non-JSON data for {method} {path}") from exc
        if not isinstance(value, dict):
            raise PioneerError(f"Pioneer returned a non-object for {method} {path}")
        return value

    def _request_text(self, method: str, path: str) -> str:
        try:
            response = self._client.request(method, path)
            response.raise_for_status()
            return response.text
        except httpx.HTTPStatusError as exc:
            detail = exc.response.text
            raise PioneerError(
                f"Pioneer {method} {path} returned {exc.response.status_code}: {detail}"
            ) from exc
        except httpx.HTTPError as exc:
            raise PioneerError(f"Pioneer {method} {path} failed: {exc}") from exc

    def list_trainable_models(self) -> JsonObject:
        return self._request("GET", "/base-models?task_type=encoder&supports_training=true")

    def upload_dataset(
        self,
        path: Path,
        *,
        dataset_name: str,
        purpose: str,
        dataset_type: str = "custom",
        content: bytes | None = None,
    ) -> UploadedDataset:
        if dataset_type not in {"classification", "ner", "custom", "decoder"}:
            raise ValueError(
                "dataset_type must be classification, ner, custom, or decoder"
            )
        try:
            reservation = UploadReservation.model_validate(
                self._request(
                    "POST",
                    "/felix/datasets/upload/url",
                    payload={
                        "dataset_name": dataset_name,
                        "dataset_type": dataset_type,
                        "format": "jsonl",
                        "filename": path.name,
                        "type": purpose,
                        "visibility": "private",
                        "generation_type": "upload",
                    },
                )
            )
        except ValidationError as exc:
            raise PioneerError(f"invalid dataset upload reservation: {exc}") from exc
        if reservation.dataset_name != dataset_name:
            raise PioneerError("dataset upload reservation returned an unexpected name")

        upload_url = urlsplit(reservation.presigned_url)
        if upload_url.scheme != "https":
            raise PioneerError("presigned URL must use HTTPS")
        if not upload_url.hostname:
            raise PioneerError("presigned URL must have a valid hostname")
        signed_content_types = parse_qs(
            upload_url.query,
            keep_blank_values=True,
            max_num_fields=100,
        ).get("content-type", [])
        if len(signed_content_types) > 1:
            raise PioneerError("dataset upload URL contains multiple content-type values")
        content_type = (
            signed_content_types[0] if signed_content_types else DEFAULT_DATASET_CONTENT_TYPE
        )
        try:
            response = self._upload_client.put(
                reservation.presigned_url,
                content=path.read_bytes() if content is None else content,
                headers={"Content-Type": content_type},
            )
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise PioneerError(
                f"dataset upload returned {exc.response.status_code}: {exc.response.text}"
            ) from exc
        except httpx.HTTPError as exc:
            raise PioneerError(f"dataset upload failed: {exc}") from exc

        self._request(
            "POST",
            "/felix/datasets/upload/process",
            payload={"dataset_id": reservation.dataset_id},
        )
        return UploadedDataset(
            dataset_id=reservation.dataset_id,
            dataset_name=reservation.dataset_name,
            version_number=reservation.version_number,
        )

    def get_dataset(self, name: str, version_number: str | None = None) -> JsonObject:
        if not name.strip():
            raise ValueError("dataset name cannot be empty")
        path = f"/felix/datasets/{quote(name, safe='')}"
        if version_number is not None:
            path += f"/{quote(version_number, safe='')}"
        return self._request("GET", path)

    def preview_dataset(self, name: str, version_number: str) -> JsonObject:
        """Return Pioneer's server-side preview of one dataset version."""
        if not name.strip():
            raise ValueError("dataset name cannot be empty")
        if not version_number.strip():
            raise ValueError("dataset version cannot be empty")
        path = (
            f"/felix/datasets/{quote(name, safe='')}/"
            f"{quote(version_number, safe='')}/preview"
        )
        return self._request("GET", path)

    def download_dataset(self, name: str, version_number: str) -> str:
        """Download one dataset version as JSONL text."""
        if not name.strip():
            raise ValueError("dataset name cannot be empty")
        if not version_number.strip():
            raise ValueError("dataset version cannot be empty")
        path = (
            f"/felix/datasets/{quote(name, safe='')}/"
            f"{quote(version_number, safe='')}/download"
            "?format=jsonl&standard_columns=false"
        )
        return self._request_text("GET", path)

    def wait_for_dataset(
        self, dataset: UploadedDataset, *, interval: int = 5, timeout: int = 600
    ) -> JsonObject:
        deadline = time.monotonic() + timeout
        while True:
            result = self.get_dataset(dataset.dataset_name, dataset.version_number)
            if result.get("id") != dataset.dataset_id:
                raise PioneerError(f"dataset {dataset.dataset_name!r} returned an unexpected ID")
            if result.get("version_number") != dataset.version_number:
                raise PioneerError(
                    f"dataset {dataset.dataset_name!r} returned an unexpected version"
                )
            status = result.get("status")
            if status == "ready":
                return result
            if status == "failed":
                raise PioneerError(
                    f"dataset {dataset.dataset_name!r} failed: {result.get('processing_error')}"
                )
            if time.monotonic() >= deadline:
                raise PioneerError(
                    f"timed out waiting for dataset {dataset.dataset_name!r}; "
                    f"last status: {status!r}"
                )
            time.sleep(interval)

    def start_training(
        self,
        *,
        model_name: str,
        dataset_name: str | None = None,
        dataset_names: list[str] | None = None,
        base_model: str,
        epochs: int,
        learning_rate: float,
    ) -> JsonObject:
        names = list(dataset_names or [])
        if dataset_name is not None:
            names.append(dataset_name)
        names = list(dict.fromkeys(names))
        if not names or any(not name.strip() for name in names):
            raise ValueError("at least one non-empty dataset name is required")
        return self._request(
            "POST",
            "/felix/training-jobs",
            payload={
                "model_name": model_name,
                "base_model": base_model,
                "datasets": [{"name": name} for name in names],
                "training_type": "lora",
                "nr_epochs": epochs,
                "learning_rate": learning_rate,
            },
        )

    def start_generation(
        self,
        *,
        task_type: str,
        dataset_name: str,
        labels: list[str],
        num_examples: int,
        domain_description: str | None = None,
        classified_examples: list[JsonObject] | None = None,
        is_seed: bool = False,
        synthesis_session_id: str | None = None,
    ) -> JsonObject:
        if task_type not in {"ner", "classification"}:
            raise ValueError("task_type must be 'ner' or 'classification'")
        if not dataset_name.strip():
            raise ValueError("dataset_name cannot be empty")
        unique_labels = list(dict.fromkeys(labels))
        if not unique_labels or any(not label.strip() for label in unique_labels):
            raise ValueError("at least one non-empty label is required")
        if num_examples < 1:
            raise ValueError("num_examples must be positive")
        if synthesis_session_id is not None and not synthesis_session_id.strip():
            raise ValueError("synthesis_session_id cannot be empty")

        payload: JsonObject = {
            "task_type": task_type,
            "dataset_name": dataset_name,
            "labels": unique_labels,
            "num_examples": num_examples,
        }
        if domain_description:
            payload["domain_description"] = domain_description
        if classified_examples:
            payload["classified_examples"] = classified_examples
        query: dict[str, str] = {"is_seed": str(is_seed).lower()}
        if synthesis_session_id is not None:
            query["synthesis_session_id"] = synthesis_session_id
        result = self._request(
            "POST",
            f"/generate?{urlencode(query)}",
            payload=payload,
        )
        if not isinstance(result.get("job_id") or result.get("id"), str):
            raise PioneerError("generation response did not include a job ID")
        return result

    def get_generation(self, job_id: str) -> JsonObject:
        if not job_id.strip():
            raise ValueError("job_id cannot be empty")
        return self._request("GET", f"/generate/jobs/{quote(job_id, safe='')}")

    def wait_for_generation(
        self, job_id: str, *, interval: int = 5, timeout: int = 1800
    ) -> JsonObject:
        deadline = time.monotonic() + timeout
        while True:
            result = self.get_generation(job_id)
            status = result.get("status")
            if status in {"ready", "complete"}:
                return result
            if status in {"failed", "errored"}:
                detail = result.get("error") or result.get("error_message")
                raise PioneerError(f"generation {job_id} failed: {detail}")
            if time.monotonic() >= deadline:
                raise PioneerError(
                    f"timed out waiting for generation {job_id}; last status: {status!r}"
                )
            time.sleep(interval)

    def get_training(self, job_id: str) -> JsonObject:
        if not job_id.strip():
            raise ValueError("job_id cannot be empty")
        return self._request("GET", f"/felix/training-jobs/{quote(job_id, safe='')}")

    def get_training_logs(self, job_id: str) -> JsonObject:
        """Return server-side logs and validation diagnostics for a training job."""
        if not job_id.strip():
            raise ValueError("job_id cannot be empty")
        return self._request("GET", f"/felix/training-jobs/{quote(job_id, safe='')}/logs")

    def wait_for_training(
        self, job_id: str, *, interval: int = 15, timeout: int = 14400
    ) -> JsonObject:
        deadline = time.monotonic() + timeout
        terminal = {
            "complete",
            "deployed",
            "failed",
            "errored",
            "stopped",
            "terminated",
            "cancelled",
        }
        while True:
            result = self.get_training(job_id)
            status = result.get("status")
            if status in terminal:
                if status not in {"complete", "deployed"}:
                    raise PioneerError(f"training {job_id} ended with status {status!r}")
                return result
            if time.monotonic() >= deadline:
                raise PioneerError(
                    f"timed out waiting for training {job_id}; last status: {status!r}"
                )
            time.sleep(interval)

    def start_evaluation(self, *, model_id: str, dataset_name: str) -> EvaluationStart:
        try:
            return EvaluationStart.model_validate(
                self._request(
                    "POST",
                    "/felix/evaluations",
                    payload={"base_model": model_id, "dataset_name": dataset_name},
                )
            )
        except ValidationError as exc:
            raise PioneerError(f"invalid evaluation response: {exc}") from exc

    def get_evaluation(self, evaluation_id: str) -> JsonObject:
        return self._request("GET", f"/felix/evaluations/{quote(evaluation_id, safe='')}")

    def wait_for_evaluation(
        self, evaluation_id: str, *, interval: int = 10, timeout: int = 3600
    ) -> JsonObject:
        deadline = time.monotonic() + timeout
        while True:
            result = self.get_evaluation(evaluation_id)
            status = result.get("status")
            if status == "complete":
                return result
            if status == "failed":
                raise PioneerError(f"evaluation {evaluation_id} failed")
            if time.monotonic() >= deadline:
                raise PioneerError(
                    f"timed out waiting for evaluation {evaluation_id}; last status: {status!r}"
                )
            time.sleep(interval)
