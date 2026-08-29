# Pipeline de entrenamiento

Esta carpeta prepara y entrena el GLiNER2 LoRA de Cala FastPath. La demo vive fuera de esta pipeline.

## Contrato

Los ejemplos canónicos guardan el texto y un plan gold. La conversión para Pioneer usa:

- `labels` para operación, raíz, campos de retorno, orden y motivo de rechazo.
- `json_structures` para valores que aparecen literalmente en la consulta.

Pioneer exige que cada valor estructurado sea un fragmento exacto del texto. Por eso el modelo extrae `10M` y el compilador decide que corresponde a `funding>10M`.

## Preparación local

```powershell
uv sync --locked
uv run cala-fastpath bootstrap
uv run cala-fastpath validate training/data/examples.jsonl
uv run cala-fastpath build training/data/examples.jsonl
uv run pytest
```

El proyecto fija Python 3.13 mediante `.python-version` y resuelve todo desde
`pyproject.toml` y `uv.lock`. No hay un flujo alternativo con pip o `requirements.txt`.

El build crea:

```text
training/artifacts/v0/
  canonical/
  pioneer/
  manifest.json
  schema.json
```

El split se hace por `group`. Las paráfrasis de un mismo caso nunca aparecen en particiones distintas.
El `test` generado aquí es un split técnico del dataset de desarrollo; no es el holdout final. La
suite sellada vive fuera del training en `benchmark/data/holdout-v1.jsonl`.

## Entrenamiento en Pioneer

Configura `PIONEER_API_KEY` en el proceso. No guardes la clave en el repositorio.

```powershell
$env:PIONEER_API_KEY = "..."

uv run cala-fastpath models

uv run cala-fastpath upload training/artifacts/v0/pioneer/train.jsonl `
  --name cala-fastpath-train-v0 `
  --purpose training `
  --wait

uv run cala-fastpath upload training/artifacts/v0/pioneer/validation.jsonl `
  --name cala-fastpath-eval-v0 `
  --purpose evaluation `
  --wait

uv run cala-fastpath train `
  --dataset cala-fastpath-train-v0 `
  --model-name cala-fastpath-v0 `
  --base-model fastino/gliner2-multi-v1 `
  --epochs 3 `
  --learning-rate 5e-5
```

El comando devuelve el ID del job. Consulta su estado:

```powershell
uv run cala-fastpath status <job-id>
```

Cuando termine, evalúa el LoRA y el modelo base contra el mismo dataset:

```powershell
uv run cala-fastpath evaluate --model-id <job-id> --dataset cala-fastpath-eval-v0
uv run cala-fastpath evaluate --model-id fastino/gliner2-multi-v1 --dataset cala-fastpath-eval-v0
```

También se puede ejecutar el ciclo remoto completo con un solo comando:

```powershell
uv run cala-fastpath pipeline `
  --artifacts-dir training/artifacts/v0 `
  --prefix cala-fastpath-v0 `
  --base-model fastino/gliner2-multi-v1
```

Este comando sube los dos datasets, espera a que estén listos, entrena el LoRA y ejecuta la misma evaluación sobre el modelo base y el modelo entrenado.

## Calidad

```powershell
uv run ruff format --check .
uv run ruff check .
uv run pytest
```

## Antes del entrenamiento real

Los ejemplos generados por `bootstrap` solo prueban la tubería. Hay que ampliar y revisar el catálogo y el dataset antes de gastar créditos. El test final debe permanecer fuera del entrenamiento y de la selección del checkpoint.
