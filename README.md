# Cala FastPath

Cala FastPath explores a narrow question: can a small language model turn a natural-language
request into a constrained Cala QL query before Cala executes it?

This repository contains the data pipeline, deterministic compiler, benchmark runner, Pioneer
API client, and a Chrome extension demo. The public extension demo is controlled. It uses a local
mock planner so the JavaScript interaction can be demonstrated without claiming that the trained
model is ready.

## Current status

The JavaScript extension and local demo are runnable. The compiler produces Cala QL for the
supported demo cases and abstains on unknown requests.

The live Pioneer model path is not complete. During the hackathon, Pioneer accepted generation,
training, and inference requests, but several endpoints returned unusable data. The exact findings
are documented in [Pioneer endpoint findings](#pioneer-endpoint-findings).

Do not present the controlled demo as a live SLM result. The UI labels its planner as `Mock SLM`
for this reason.

## What the demo does

The extension adds a `FastPath` mode to Cala's Knowledge Query page.

```text
natural language
      |
      v
Chrome extension
      |
      v
localhost planner -> structured plan -> deterministic Cala QL compiler
      |
      v
Cala's native query submission
```

The planner runs at `http://127.0.0.1:8765/plan`. When it accepts a request, the extension replaces
the natural-language input with the compiled Cala QL and clicks Cala's existing Search button. On
the real Cala Console, Cala executes the query with the browser's signed-in session.

The extension does not read Cala credentials and does not call the Cala API directly. Its only
host permission is the local planner.

The standalone harness at `http://127.0.0.1:8765/demo/` is fully simulated. It reproduces the Cala
page, planner flow, result navigation, and recent-query list without contacting Cala.

## Controlled queries

The mock planner intentionally supports only these query families:

```text
Companies founded by former Google employees
Spanish startups with funding between 10M and 50M
Top 5 Spanish startups by funding
```

Equivalent Spanish variants are also supported. Unknown requests abstain instead of producing an
unconstrained query.

## Requirements

- Python 3.13
- [uv](https://docs.astral.sh/uv/)
- Chrome or another Chromium browser for the extension
- A Cala Console account only when testing on the real Cala page
- API keys only for the optional remote workflows

The extension uses plain Manifest V3 JavaScript. It has no Node.js, npm, bundler, or build step.

## Installation

Clone the repository and install the locked Python environment:

```powershell
git clone <repository-url>
cd hackaton
uv sync --locked
```

Create the local environment file:

```powershell
Copy-Item .env.example .env
```

On macOS or Linux:

```bash
cp .env.example .env
```

The controlled demo does not require any API key.

## Run the controlled browser demo

Start the local server:

```powershell
uv run python -m cala_fastpath_training.demo_server
```

Open:

```text
http://127.0.0.1:8765/demo/
```

Check the server separately if needed:

```powershell
Invoke-RestMethod http://127.0.0.1:8765/health
```

Expected response:

```json
{
  "status": "ok",
  "planner": "mock",
  "model": "mock/cala-fastpath-v0"
}
```

The optional mock delay can be changed without editing code:

```powershell
$env:CALA_FASTPATH_MOCK_DELAY_MS = "250"
uv run python -m cala_fastpath_training.demo_server
```

## Load the Chrome extension

Keep the local demo server running, then:

1. Open `chrome://extensions`.
2. Enable Developer mode.
3. Select **Load unpacked**.
4. Choose the repository's `extension` directory.
5. Open `https://console.cala.ai/playground/knowledge-query`.
6. Select `FastPath`, enter a supported query, and submit it.

The content script loads across the Cala Console so it survives client-side navigation, but it
activates only on Knowledge Query routes.

If the planner is unavailable or abstains, the extension does not call Cala.

## Configuration

Copy `.env.example` to `.env` and fill only the services you intend to use.

| Variable | Required for | Default |
| --- | --- | --- |
| `PIONEER_API_KEY` | Pioneer datasets, generation, training, evaluation, inference | None |
| `PIONEER_BASE_URL` | Pioneer API | `https://api.pioneer.ai` |
| `CALA_API_KEY` | Direct Cala API experiments | None |
| `CALA_BASE_URL` | Cala API | `https://api.cala.ai/v1` |
| `OPENAI_API_KEY` | OpenAI benchmark systems | None |
| `OPENAI_BASE_URL` | OpenAI API | `https://api.openai.com/v1` |
| `OPENAI_MODEL` | OpenAI planner baseline | `gpt-5.6-luna` |
| `OPENAI_REASONING_EFFORT` | OpenAI planner reasoning | `high` |
| `GLINER_BASE_MODEL` | Local GLiNER2 baseline | `fastino/gliner2-multi-v1` |
| `CALA_FASTPATH_MOCK_DELAY_MS` | Controlled demo latency | `650` |

Never put API keys in the extension. `.env` is ignored by Git.

## Data and compiler workflow

The canonical examples store natural language and a reviewed plan. The compiler converts a valid
plan into Cala QL. It does not ask a model to write executable Cala QL directly.

Validate and build the checked-in examples:

```powershell
uv run cala-fastpath validate training/data/examples.jsonl
uv run cala-fastpath build training/data/examples.jsonl --output-dir training/artifacts/v0
```

The build creates:

```text
training/artifacts/v0/
  canonical/
  pioneer/
  manifest.json
  schema.json
```

The split is grouped so paraphrases from the same family do not cross partitions. See
[training/README.md](training/README.md) for the training contract and
[demo.md](demo.md) for the broader evaluation design.

## Benchmark runner

Run a small local GLiNER2 baseline:

```powershell
uv run cala-fastpath generate benchmark/data/google-founder-projection-dev.jsonl `
  --systems base `
  --output benchmark/runs/base-dev.jsonl
```

Run the OpenAI planner baselines after setting `OPENAI_API_KEY`:

```powershell
uv run cala-fastpath generate benchmark/data/google-founder-projection-dev.jsonl `
  --systems openai,openai-skill `
  --output benchmark/runs/openai-dev.jsonl
```

`benchmark/data/google-founder-projection-dev.jsonl` is development regression data. It is not a
final test. `benchmark/data/holdout-v1.jsonl` is sealed and must remain untouched until the model,
schema, compiler, and thresholds are frozen.

## Pioneer API workflow

The CLI can inspect models, datasets, generation jobs, training jobs, and server-side logs:

```powershell
uv run cala-fastpath models
uv run cala-fastpath generation-status <generation-job-id>
uv run cala-fastpath dataset-preview <dataset-name> --version <version>
uv run cala-fastpath dataset-download <dataset-name> --version <version>
uv run cala-fastpath status <training-job-id>
uv run cala-fastpath training-logs <training-job-id>
```

Uploads must declare the actual dataset type:

```powershell
uv run cala-fastpath upload <dataset.jsonl> `
  --name <dataset-name> `
  --purpose training `
  --dataset-type classification `
  --wait
```

Supported upload types in the live Pioneer OpenAPI are `classification`, `ner`, `custom`, and
`decoder`.

## Pioneer endpoint findings

These observations come from direct API calls made during the hackathon. They explain why the
repository currently ships a controlled mock instead of claiming a working trained SLM.

### Synthetic classification

`POST /generate` accepted classification jobs and returned job IDs. Jobs using a large taxonomy,
a reduced three-label taxonomy, human-readable labels, positive classified examples, and the
`quality` profile all ended with:

```text
Generation produced no valid samples. Try adjusting your labels or adding more examples.
```

### Synthetic NER

One NER generation job reported `ready` with 95 samples. Dataset preview and download showed that
all 95 rows were exactly:

```json
{"entities": []}
```

The rows had no `text` field and no annotated spans. The training logs then reported:

```text
Loaded 95 samples
Converted 0 samples to GLiNER format
No valid samples in dataset 0, skipping
Training failed with error: No valid datasets found
```

A seeded NER retry also failed. A separate control using the documentation's simple `person`,
`company`, and `product` labels failed with the same no-valid-samples message.

### Label existing NER

`POST /generate/ner/label-existing` returned success and a count for 21 inputs, but every returned
item was an empty object. Asking Pioneer to save the response as a dataset failed validation because
the generated rows did not contain `text` and `entities`.

### Mixed training

Pioneer rejected a training job that combined separate classification and NER datasets:

```text
Datasets are not compatible for training.
Expected compatibility 'classification:text:na', got ['ner:text:na'].
```

This means the current hosted flow cannot simply combine the two generated datasets into one LoRA
job.

### Classification inference

A classification LoRA completed and reached `deployed` with 17 registered labels. Both single-label
and multi-label calls to `POST /inference` returned:

```json
{
  "categories": []
}
```

The response remained empty with threshold `0`. The uploaded training data also had a design flaw:
multi-label examples were flattened into repeated single-label rows with identical text and
different labels. That should be corrected before another training run, but it does not explain why
single-label inference returned no winning category at threshold `0`.

### Practical consequence

The compiler and extension flow are demonstrable. The hosted SLM path is not. A future live-model
demo needs all of the following before replacing the mock:

1. Pioneer generation or uploaded rows that retain `text` and valid annotations.
2. A training representation that matches the intended classification mode.
3. A deployed model that returns at least one category in single-label inference.
4. A benchmark run that produces a valid plan and reaches the compiler.

## Partners and services

The repository distinguishes runtime integrations from hackathon partner acknowledgements.

| Partner or service | Role in this project | Runtime dependency |
| --- | --- | --- |
| Cala | Target console, Knowledge Query experience, and Cala QL execution | Yes for the real-console demo |
| Fastino Labs / Pioneer | GLiNER2 models, synthetic-data experiments, LoRA training, hosted inference | Optional and currently blocked by the findings above |
| OpenAI | SOTA planner baseline and skill-conditioned benchmark path | Optional |
| Aikido | Hackathon security side-challenge partner | No |
| fal | Hackathon side-challenge partner | No |
| Entire | Hackathon partner for session, prompt, tool-call, and commit provenance | No |

Aikido, fal, and Entire are acknowledged because they are hackathon partners shown in the event
brief. The current extension does not call their APIs or include their SDKs.

## Security boundaries

- The demo server binds only to `127.0.0.1` or `localhost`.
- The extension permits only session storage and `http://127.0.0.1:8765/*`.
- The content script validates planner responses before submitting Cala QL.
- Abstentions never call Cala.
- Planner requests have a 15-second browser timeout.
- The local server limits request bodies and query length.
- API keys remain in the Python process and never enter extension storage.

## Project structure

```text
benchmark/       Development suites, sealed holdout, benchmark skill
demo_harness/    Standalone simulated Cala page
extension/       Manifest V3 JavaScript extension
src/             Data pipeline, compiler, API clients, demo server
tests/           Python and extension package checks
training/        Catalog, canonical examples, training documentation
```

## Verification

Run the repository checks:

```powershell
uv run ruff format --check .
uv run ruff check .
uv run pytest
```

For the extension interaction, start the demo server and manually verify one accepted query and one
abstained query in the harness. The real-console path requires a signed-in Cala browser session.

## Troubleshooting

### Planner unavailable

Confirm that the server is running and that `http://127.0.0.1:8765/health` returns `ok`. Reload the
extension after restarting Chrome.

### FastPath controls do not appear

Confirm that the page URL starts with `https://console.cala.ai/playground/knowledge-query` and that
the unpacked extension is enabled.

### FastPath stops instead of submitting

The controlled planner abstains on queries outside its small allowlist. Use one of the controlled
queries listed above.

### Pioneer says a dataset is ready but training rejects it

Inspect the stored rows and server logs instead of trusting metadata alone:

```powershell
uv run cala-fastpath dataset-preview <name> --version <version>
uv run cala-fastpath dataset-download <name> --version <version>
uv run cala-fastpath training-logs <job-id>
```

The observed NER failure was caused by 95 stored rows without `text` or entity annotations.

## License and data handling

No license file is currently included. Treat the repository as hackathon code until ownership and
licensing are documented. Do not commit API keys, downloaded private datasets, or model artifacts.
