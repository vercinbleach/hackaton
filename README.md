# Cala FastPath

Pipeline de entrenamiento para convertir consultas en lenguaje natural en planes estructurados de Cala mediante GLiNER2.

- [Entrenamiento](training/README.md)
- [Demo y evaluación](demo.md)

## Generar candidatos desde CLI

Configura `OPENAI_API_KEY` en `.env` y ejecuta las tres variantes sobre el mismo JSONL:

```powershell
uv sync --locked
uv run cala-fastpath generate
```

El comando guarda `openai`, `base` y `openai-skill` en `benchmark/runs/latest.jsonl`.
Esta fase solo genera planes y Cala QL. No puntúa ni llama a `/knowledge/query`.
Las variantes OpenAI usan `gpt-5.6-luna` con razonamiento `high` por defecto.

Para ejecutar una sola variante:

```powershell
uv run cala-fastpath generate --systems base
```
