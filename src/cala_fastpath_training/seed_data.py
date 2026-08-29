from __future__ import annotations

from typing import Any

from .models import TrainingExample


def _row(
    example_id: str,
    group: str,
    language: str,
    text: str,
    plan: dict[str, Any],
) -> TrainingExample:
    return TrainingExample.model_validate(
        {"id": example_id, "group": group, "language": language, "text": text, "plan": plan}
    )


def bootstrap_examples() -> list[TrainingExample]:
    rows: list[TrainingExample] = []

    scenarios = [
        (
            "former-google-founders",
            {
                "operation": "knowledge_query",
                "root": "companies",
                "filters": [{"kind": "previous_job_eq", "mention": "Google", "value": "Google"}],
                "return": ["name", "founder"],
            },
            [
                (
                    "es",
                    "Compañías fundadas por antiguos empleados de Google, con nombre y fundador",
                ),
                ("es", "Dame el nombre y fundador de empresas creadas por exempleados de Google"),
                ("en", "Companies founded by former Google employees, returning name and founder"),
                (
                    "en",
                    "Show company names and founders where the founder previously worked at Google",
                ),
            ],
        ),
        (
            "spanish-funded-startups",
            {
                "operation": "knowledge_query",
                "root": "startups",
                "filters": [
                    {"kind": "location_eq", "mention": "España", "value": "Spain"},
                    {"kind": "funding_gt", "mention": "10M", "value": "10M"},
                    {"kind": "funding_lt", "mention": "50M", "value": "50M"},
                ],
                "return": ["name", "funding"],
            },
            [
                (
                    "es",
                    "Startups de España con financiación superior a 10M e inferior a 50M, "
                    "devuelve nombre y financiación",
                ),
                (
                    "es",
                    "Nombre y financiación de startups ubicadas en España que hayan levantado "
                    "más de 10M y menos de 50M",
                ),
                (
                    "en",
                    "Return name and funding for startups in España with funding above 10M "
                    "and below 50M",
                ),
            ],
        ),
        (
            "ai-employee-threshold",
            {
                "operation": "knowledge_query",
                "root": "companies",
                "filters": [
                    {"kind": "industry_eq", "mention": "AI", "value": "AI"},
                    {"kind": "employee_count_gt", "mention": "100", "value": "100"},
                ],
                "return": ["name", "employee_count"],
            },
            [
                (
                    "es",
                    "Empresas de AI con más de 100 empleados, devuelve nombre y número "
                    "de empleados",
                ),
                (
                    "es",
                    "Dame nombre y plantilla de compañías del sector AI que superen los "
                    "100 empleados",
                ),
                ("en", "AI companies with more than 100 employees, return name and employee count"),
            ],
        ),
        (
            "largest-funded-spanish-startups",
            {
                "operation": "knowledge_query",
                "root": "startups",
                "filters": [{"kind": "location_eq", "mention": "Spain", "value": "Spain"}],
                "return": ["name", "funding", "sector"],
                "order_by": "funding:desc",
                "limit": 5,
                "limit_mention": "5",
            },
            [
                (
                    "es",
                    "Las 5 startups con más financiación en Spain, devuelve nombre, "
                    "financiación y sector",
                ),
                ("en", "Top 5 startups by funding in Spain, return name, funding and sector"),
            ],
        ),
        (
            "openai-profile",
            {
                "operation": "retrieve_entity",
                "root": "companies",
                "entity": {"mention": "OpenAI"},
                "filters": [],
                "return": ["employee_count", "registered_address"],
            },
            [
                ("es", "Cuántos empleados tiene OpenAI y cuál es su domicilio registrado"),
                ("es", "Dame la plantilla y la dirección registral de OpenAI"),
                ("en", "Give me OpenAI employee count and registered address"),
            ],
        ),
        (
            "apple-profile",
            {
                "operation": "retrieve_entity",
                "root": "companies",
                "entity": {"mention": "Apple"},
                "filters": [],
                "return": ["legal_name", "registered_address", "employee_count"],
            },
            [
                ("es", "Nombre legal, domicilio registrado y número de empleados de Apple"),
                ("en", "Return Apple legal name, registered address and employee count"),
            ],
        ),
        (
            "unsupported-explanation",
            {
                "operation": "unsupported",
                "reason": "open_ended_explanation",
                "filters": [],
                "return": [],
            },
            [
                ("es", "Explícame por qué OpenAI ha tenido tanto impacto en la industria"),
                ("en", "Explain why OpenAI has had such a large impact on the industry"),
            ],
        ),
        (
            "unsupported-comparison",
            {
                "operation": "unsupported",
                "reason": "multi_step_reasoning",
                "filters": [],
                "return": [],
            },
            [
                ("es", "Compara las estrategias de OpenAI y Anthropic y dime cuál es mejor"),
                ("en", "Compare OpenAI and Anthropic strategies and decide which is better"),
            ],
        ),
    ]

    counter = 1
    for group, plan, texts in scenarios:
        for language, text in texts:
            rows.append(_row(f"seed-{counter:03d}", group, language, text, plan))
            counter += 1
    return rows
