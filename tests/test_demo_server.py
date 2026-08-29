from __future__ import annotations

from threading import Thread

import httpx

from cala_fastpath_training.catalog import load_catalog
from cala_fastpath_training.demo_server import CATALOG_PATH, create_server, mock_generation_record
from cala_fastpath_training.models import BenchmarkCase


def test_mock_accepts_google_founder_query() -> None:
    record = mock_generation_record(
        BenchmarkCase(
            id="google-founders",
            query="Companies founded by former Google employees",
        ),
        load_catalog(CATALOG_PATH),
        latency_ms=12.5,
    )

    assert record.decision == "accepted"
    assert record.plan is not None
    assert record.plan.root == "companies"
    assert record.cala_query == "companies.founder.previous_job=Google.return(name)"


def test_mock_accepts_spanish_funding_range() -> None:
    record = mock_generation_record(
        BenchmarkCase(
            id="spanish-funding",
            query="Spanish startups with funding between 10M and 50M",
        ),
        load_catalog(CATALOG_PATH),
        latency_ms=8,
    )

    assert record.decision == "accepted"
    assert record.plan is not None
    assert record.plan.filters[0].mention == "Spanish"
    assert record.cala_query == (
        "startups.location=Spain.funding>10M.funding<50M.return(name, funding)"
    )


def test_mock_abstains_without_a_fixture() -> None:
    record = mock_generation_record(
        BenchmarkCase(id="unknown", query="Show me every company"),
        load_catalog(CATALOG_PATH),
        latency_ms=3,
    )

    assert record.decision == "abstained"
    assert record.cala_query is None
    assert record.abstention_reason == "mock_query_not_configured"


def test_mock_does_not_generalize_beyond_its_google_fixture() -> None:
    record = mock_generation_record(
        BenchmarkCase(
            id="google-employers",
            query="Companies employing former Google employees",
        ),
        load_catalog(CATALOG_PATH),
        latency_ms=3,
    )

    assert record.decision == "abstained"
    assert record.abstention_reason == "mock_query_not_configured"


def test_mock_abstains_from_ambiguous_funding_bounds() -> None:
    record = mock_generation_record(
        BenchmarkCase(
            id="ambiguous-funding",
            query="Spanish startups with funding either 10M or 50M",
        ),
        load_catalog(CATALOG_PATH),
        latency_ms=3,
    )

    assert record.decision == "abstained"
    assert record.abstention_reason == "mock_query_not_configured"


def test_demo_server_serves_harness_and_plans() -> None:
    server = create_server(port=0, mock_delay_ms=0)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base_url = f"http://127.0.0.1:{server.server_port}"
    try:
        with httpx.Client(base_url=base_url) as client:
            health = client.get("/health")
            harness = client.get("/demo/")
            restored_result = client.get("/playground/knowledge-query/mock-result")
            response = client.post(
                "/plan",
                json={
                    "id": "e2e-google",
                    "query": "Companies founded by former Google employees",
                },
            )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)

    assert health.json()["planner"] == "mock"
    assert "Cala FastPath harness" in harness.text
    assert "Cala FastPath harness" in restored_result.text
    assert response.status_code == 200
    assert response.json()["decision"] == "accepted"


def test_demo_server_rejects_extra_request_fields() -> None:
    server = create_server(port=0, mock_delay_ms=0)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        response = httpx.post(
            f"http://127.0.0.1:{server.server_port}/plan",
            json={"id": "bad", "query": "test", "unexpected": True},
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)

    assert response.status_code == 400
    assert response.json()["error"] == "invalid_request"


def test_demo_server_rejects_whitespace_only_query() -> None:
    server = create_server(port=0, mock_delay_ms=0)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        response = httpx.post(
            f"http://127.0.0.1:{server.server_port}/plan",
            json={"id": "blank", "query": "   "},
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)

    assert response.status_code == 400
    assert response.json()["error"] == "invalid_request"
