from __future__ import annotations

import mimetypes
import os
import time
from pathlib import Path
from urllib.parse import quote

import httpx
from pydantic import ValidationError

from .models import EvaluationStart, JsonObject, UploadedDataset, UploadReservation


class PioneerError(RuntimeError):
    pass


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
        self._client = httpx.Client(
            base_url=base_url.rstrip("/"),
            headers={"X-API-Key": api_key, "Accept": "application/json"},
            timeout=timeout,
            transport=transport,
        )

    def __enter__(self) -> PioneerClient:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def close(self) -> None:
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

    def list_trainable_models(self) -> JsonObject:
        return self._request("GET", "/base-models?task_type=encoder&supports_training=true")

    def upload_dataset(
        self,
        path: Path,
        *,
        dataset_name: str,
        purpose: str,
    ) -> UploadedDataset:
        try:
            reservation = UploadReservation.model_validate(
                self._request(
                    "POST",
                    "/felix/datasets/upload/url",
                    payload={
                        "dataset_name": dataset_name,
                        "dataset_type": "custom",
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

        content_type = mimetypes.guess_type(path.name)[0] or "application/x-ndjson"
        try:
            response = self._client.put(
                reservation.presigned_url,
                content=path.read_bytes(),
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

    def get_dataset(self, name: str) -> JsonObject:
        return self._request("GET", f"/felix/datasets/{quote(name, safe='')}")

    def wait_for_dataset(self, name: str, *, interval: int = 5, timeout: int = 600) -> JsonObject:
        deadline = time.monotonic() + timeout
        while True:
            result = self.get_dataset(name)
            versions = result.get("versions", [])
            latest = versions[0] if versions else result
            if not isinstance(latest, dict):
                raise PioneerError(f"dataset {name!r} returned an invalid version")
            status = latest.get("status")
            if status == "ready":
                return latest
            if status == "failed":
                raise PioneerError(f"dataset {name!r} failed: {latest.get('processing_error')}")
            if time.monotonic() >= deadline:
                raise PioneerError(
                    f"timed out waiting for dataset {name!r}; last status: {status!r}"
                )
            time.sleep(interval)

    def start_training(
        self,
        *,
        model_name: str,
        dataset_name: str,
        base_model: str,
        epochs: int,
        learning_rate: float,
    ) -> JsonObject:
        return self._request(
            "POST",
            "/felix/training-jobs",
            payload={
                "model_name": model_name,
                "base_model": base_model,
                "datasets": [{"name": dataset_name}],
                "training_type": "lora",
                "nr_epochs": epochs,
                "learning_rate": learning_rate,
            },
        )

    def get_training(self, job_id: str) -> JsonObject:
        return self._request("GET", f"/felix/training-jobs/{quote(job_id, safe='')}")

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
