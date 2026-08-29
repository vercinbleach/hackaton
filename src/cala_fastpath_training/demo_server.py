from __future__ import annotations

import argparse
import json
import os
import re
import time
import unicodedata
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from .catalog import Catalog, load_catalog
from .compiler import compile_plan
from .models import BenchmarkCase, GenerationRecord, Plan

ROOT = Path(__file__).resolve().parents[2]
CATALOG_PATH = ROOT / "training" / "config" / "catalog.json"
HARNESS_DIR = ROOT / "demo_harness"
EXTENSION_DIR = ROOT / "extension"
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8765
MAX_REQUEST_BYTES = 16_384
MAX_QUERY_LENGTH = 2_000
MOCK_MODEL = "mock/cala-fastpath-v0"
_CHROME_EXTENSION_ORIGIN = re.compile(r"^chrome-extension://[a-p]{32}$")


def _fold(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value.casefold())
    return "".join(character for character in normalized if not unicodedata.combining(character))


def _knowledge_plan(
    *,
    root: str,
    filters: list[dict[str, Any]],
    returns: list[str],
    order_by: str | None = None,
    limit: int | None = None,
    limit_mention: str | None = None,
) -> Plan:
    return Plan.model_validate(
        {
            "operation": "knowledge_query",
            "root": root,
            "filters": filters,
            "return": returns,
            "entity": None,
            "order_by": order_by,
            "limit": limit,
            "limit_mention": limit_mention,
            "reason": None,
        }
    )


def _unsupported_plan(reason: str) -> Plan:
    return Plan.model_validate(
        {
            "operation": "unsupported",
            "root": None,
            "filters": [],
            "return": [],
            "entity": None,
            "order_by": None,
            "limit": None,
            "limit_mention": None,
            "reason": reason,
        }
    )


def _mock_plan(query: str) -> Plan:
    folded = " ".join(_fold(query).split()).strip(" .,!?¿¡")

    if folded in {
        "companies founded by former google employees",
        "empresas fundadas por antiguos empleados de google",
    }:
        return _knowledge_plan(
            root="companies",
            filters=[
                {
                    "kind": "previous_job_eq",
                    "mention": "Google",
                    "value": "Google",
                }
            ],
            returns=[],
        )

    if folded in {
        "spanish startups with funding between 10m and 50m",
        "startups espanolas con financiacion entre 10m y 50m",
    }:
        location_mention = "Spanish" if folded.startswith("spanish ") else "españolas"
        return _knowledge_plan(
            root="startups",
            filters=[
                {
                    "kind": "location_eq",
                    "mention": location_mention,
                    "value": "Spain",
                },
                {
                    "kind": "funding_gt",
                    "mention": "10M",
                    "value": "10M",
                },
                {
                    "kind": "funding_lt",
                    "mention": "50M",
                    "value": "50M",
                },
            ],
            returns=["funding"],
        )

    if folded in {
        "top 5 spanish startups by funding",
        "las 5 startups espanolas con mas financiacion",
    }:
        location_mention = "Spanish" if " spanish " in f" {folded} " else "españolas"
        return _knowledge_plan(
            root="startups",
            filters=[
                {
                    "kind": "location_eq",
                    "mention": location_mention,
                    "value": "Spain",
                }
            ],
            returns=["funding"],
            order_by="funding:desc",
            limit=5,
            limit_mention="5",
        )

    if any(term in folded for term in ("explain", "why", "explica", "por que")) and any(
        term in folded for term in ("company", "companies", "empresa", "openai")
    ):
        return _unsupported_plan("open_ended_explanation")

    return _unsupported_plan("mock_query_not_configured")


def mock_generation_record(
    case: BenchmarkCase,
    catalog: Catalog,
    *,
    latency_ms: float,
) -> GenerationRecord:
    plan = _mock_plan(case.query)
    if plan.operation == "unsupported":
        return GenerationRecord(
            case_id=case.id,
            query=case.query,
            system="mock",
            model=MOCK_MODEL,
            plan=plan,
            cala_query=None,
            latency_ms=latency_ms,
            decision="abstained",
            abstention_reason=plan.reason,
        )
    cala_query = compile_plan(plan, catalog)
    return GenerationRecord(
        case_id=case.id,
        query=case.query,
        system="mock",
        model=MOCK_MODEL,
        plan=plan,
        cala_query=cala_query,
        latency_ms=latency_ms,
        decision="accepted",
    )


class DemoServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(
        self,
        server_address: tuple[str, int],
        catalog: Catalog,
        *,
        mock_delay_ms: int,
    ) -> None:
        super().__init__(server_address, DemoRequestHandler)
        self.catalog = catalog
        self.mock_delay_ms = mock_delay_ms


class DemoRequestHandler(BaseHTTPRequestHandler):
    server: DemoServer

    _STATIC_FILES = {
        "/demo/": (HARNESS_DIR / "index.html", "text/html; charset=utf-8"),
        "/demo/harness.css": (HARNESS_DIR / "harness.css", "text/css; charset=utf-8"),
        "/demo/harness.js": (
            HARNESS_DIR / "harness.js",
            "text/javascript; charset=utf-8",
        ),
        "/extension/content.js": (
            EXTENSION_DIR / "content.js",
            "text/javascript; charset=utf-8",
        ),
        "/extension/styles.css": (
            EXTENSION_DIR / "styles.css",
            "text/css; charset=utf-8",
        ),
    }

    def log_message(self, format: str, *args: object) -> None:
        print(f"[demo] {self.address_string()} {format % args}")

    def _allowed_origin(self) -> str | None:
        origin = self.headers.get("Origin")
        if origin in {
            f"http://{DEFAULT_HOST}:{self.server.server_port}",
            f"http://localhost:{self.server.server_port}",
        }:
            return origin
        if origin and _CHROME_EXTENSION_ORIGIN.fullmatch(origin):
            return origin
        return None

    def _common_headers(self, *, content_type: str, content_length: int) -> None:
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(content_length))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("Content-Security-Policy", "default-src 'self'; frame-ancestors 'none'")
        if origin := self._allowed_origin():
            self.send_header("Access-Control-Allow-Origin", origin)
            self.send_header("Vary", "Origin")

    def _send_bytes(
        self,
        status: HTTPStatus,
        body: bytes,
        *,
        content_type: str,
    ) -> None:
        self.send_response(status)
        self._common_headers(content_type=content_type, content_length=len(body))
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def _send_json(self, status: HTTPStatus, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
        self._send_bytes(status, body, content_type="application/json; charset=utf-8")

    def _read_json(self) -> dict[str, Any]:
        content_type = self.headers.get("Content-Type", "").partition(";")[0].strip()
        if content_type != "application/json":
            raise ValueError("Content-Type must be application/json")
        raw_length = self.headers.get("Content-Length")
        if raw_length is None:
            raise ValueError("Content-Length is required")
        length = int(raw_length)
        if length < 1 or length > MAX_REQUEST_BYTES:
            raise ValueError("request body is empty or too large")
        value = json.loads(self.rfile.read(length))
        if not isinstance(value, dict):
            raise ValueError("request body must be a JSON object")
        return value

    def do_OPTIONS(self) -> None:
        if self.path != "/plan":
            self._send_json(HTTPStatus.NOT_FOUND, {"error": "not_found"})
            return
        self.send_response(HTTPStatus.NO_CONTENT)
        self.send_header("Content-Length", "0")
        self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Max-Age", "600")
        if origin := self._allowed_origin():
            self.send_header("Access-Control-Allow-Origin", origin)
            self.send_header("Vary", "Origin")
        self.end_headers()

    def do_HEAD(self) -> None:
        self.do_GET()

    def do_GET(self) -> None:
        request_path = self.path.partition("?")[0]
        if request_path == "/":
            self.send_response(HTTPStatus.TEMPORARY_REDIRECT)
            self.send_header("Location", "/demo/")
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        if request_path == "/health":
            self._send_json(
                HTTPStatus.OK,
                {"status": "ok", "planner": "mock", "model": MOCK_MODEL},
            )
            return
        static_file = self._STATIC_FILES.get(request_path)
        if static_file is None and re.fullmatch(
            r"/playground/knowledge-query/[^/]+/?", request_path
        ):
            static_file = self._STATIC_FILES["/demo/"]
        if static_file is None:
            self._send_json(HTTPStatus.NOT_FOUND, {"error": "not_found"})
            return
        path, content_type = static_file
        try:
            body = path.read_bytes()
        except FileNotFoundError:
            self._send_json(HTTPStatus.NOT_FOUND, {"error": "asset_not_found"})
            return
        self._send_bytes(HTTPStatus.OK, body, content_type=content_type)

    def do_POST(self) -> None:
        if self.path != "/plan":
            self._send_json(HTTPStatus.NOT_FOUND, {"error": "not_found"})
            return
        started = time.perf_counter()
        try:
            payload = self._read_json()
            case = BenchmarkCase.model_validate(payload)
            if not case.query.strip():
                raise ValueError("query must contain non-whitespace characters")
            if len(case.query) > MAX_QUERY_LENGTH:
                raise ValueError(f"query exceeds {MAX_QUERY_LENGTH} characters")
        except (json.JSONDecodeError, UnicodeDecodeError, ValidationError, ValueError) as exc:
            self._send_json(
                HTTPStatus.BAD_REQUEST,
                {"error": "invalid_request", "detail": str(exc)},
            )
            return
        if self.server.mock_delay_ms:
            time.sleep(self.server.mock_delay_ms / 1_000)
        record = mock_generation_record(case, self.server.catalog, latency_ms=0)
        latency_ms = (time.perf_counter() - started) * 1_000
        record = record.model_copy(update={"latency_ms": latency_ms})
        self._send_json(
            HTTPStatus.OK,
            record.model_dump(by_alias=True, exclude_none=False),
        )


def create_server(
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
    *,
    mock_delay_ms: int | None = None,
) -> DemoServer:
    if host not in {"127.0.0.1", "localhost"}:
        raise ValueError("demo server must bind to localhost")
    if mock_delay_ms is None:
        mock_delay_ms = int(os.environ.get("CALA_FASTPATH_MOCK_DELAY_MS", "650"))
    if not 0 <= mock_delay_ms <= 10_000:
        raise ValueError("mock delay must be between 0 and 10000 milliseconds")
    return DemoServer((host, port), load_catalog(CATALOG_PATH), mock_delay_ms=mock_delay_ms)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Cala FastPath extension demo")
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--mock-delay-ms", type=int)
    args = parser.parse_args()
    server = create_server(args.host, args.port, mock_delay_ms=args.mock_delay_ms)
    print(f"Cala FastPath demo: http://{server.server_address[0]}:{server.server_port}/demo/")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
