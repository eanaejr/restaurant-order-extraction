# Order extraction service

A small FastAPI service that is one component of a phone-based restaurant
order assistant: given a sentence a guest actually says (in Croatian), it
extracts **what** was ordered and **in what quantity**, using a free Gemini
model together with the restaurant menu (`jelovnik.json`).

```
POST /order {"text": "dvije margarite i jednu colu"}

{"items": [{"id": "margarita", "quantity": 2},
           {"id": "coca_cola", "quantity": 1}],
 "unavailable": []}
```

Items that exist on the menu come back in `items`, referenced by menu `id`.
Anything the guest asks for that is **not** on the menu comes back separately
in `unavailable` — it is never dropped and never silently replaced with a
similar menu item:

```
POST /order {"text": "dva hamburgera i jednu margaritu"}

{"items": [{"id": "margarita", "quantity": 1}],
 "unavailable": [{"text": "hamburger", "quantity": 2}]}
```

## How the service is meant to be used

The service is deliberately **stateless** and answers only to its caller —
it never talks to the guest or to the restaurant:

```
guest (phone) <-> AI assistant (STT/TTS, conversation, session state)
                        |  POST /order {"text": "..."}
                        v
                 this service (extraction only)
                        |
                        v
          {"items": [...], "unavailable": [...]}
```

`unavailable` is the signal for the calling assistant to continue the
conversation, e.g. "we don't have hamburgers — would you like a pizza
instead?". That conversation, and any state, belongs to the assistant, not
to this service; that keeps the service easy to test, scale and verify.

## Running

Requires Python 3.10+.

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env               # then open .env and paste your key
uvicorn app.main:app --reload      # interactive API docs at http://127.0.0.1:8000/docs
```

The API key is read only from the `GEMINI_API_KEY` environment variable —
never from the code, never from the repository. For local development, the
gitignored `.env` file at the project root conveniently populates those
variables; real environment variables always take precedence over `.env`,
so exporting `GEMINI_API_KEY` works exactly the same way in production.
`.env.example` is the committed template that documents the variables
without containing any secrets. Without a key the service still starts,
and `POST /order` answers `500` with a clear message.

## Configuration

| Variable | Default | Meaning |
|---|---|---|
| `GEMINI_API_KEY` | — (required) | Free key from [Google AI Studio](https://aistudio.google.com/apikey). |
| `GEMINI_MODEL` | `gemini-3.8-flash` | Any Gemini model with a free tier, e.g. `gemini-3.5-flash` or `gemini-3.5-flash-lite`. |
| `GEMINI_THINKING_LEVEL` | `low` | `minimal`/`low`/`medium`/`high`. `minimal` is rejected by `gemini-3.7/3.8-flash`. |
| `GEMINI_FALLBACK_MODELS` | `gemini-3.5-flash-lite, gemini-3.7-flash` | Free-tier models tried in order (each with one retry) when the primary model fails, e.g. on a transient `503 high demand`. Set to an empty value to disable. |
| `GEMINI_TEMPERATURE` | not set | **Legacy 2.5 models only** (see below). |

Any of these can also live in the gitignored `.env` file at the project root
for local development; real environment variables always win over `.env`.
The generation parameters live in one function, `app/gemini.py: build_config`,
so it is easy to review exactly what is (and is not) sent to the model.

**Why there is no temperature by default:** on Gemini 3.x models the backend
ignores `temperature`, `top_p` and `top_k` (they are deprecated for that
family). Determinism is instead achieved the recommended way: structured
output (`response_mime_type` + `response_schema`) plus a fixed
`thinking_level`. We therefore never send a temperature to 3.x models;
`GEMINI_TEMPERATURE` exists only if you point `GEMINI_MODEL` at an older
2.5 model.

**Why `thinking_level=low`:** order extraction is simple instruction
following, not deep reasoning. `low` minimises latency and token usage
without hurting quality; the schema already constrains the output shape.

The generation parameters live in one function, `app/gemini.py: build_config`,
so it is easy to review exactly what is (and is not) sent to the model.

## How the extraction works

1. `POST /order` validates the body: a blank or missing `text` answers `422`.
2. The menu and the guest utterance go to the Gemini model together with a
   system prompt (`app/prompts/system_prompt.md`) that fixes the rules:
   match Croatian word forms to menu ids, never invent ids, never replace a
   missing item with a similar one, unknown requests go to `unavailable`
   with the guest's own words.
3. The model must answer in the exact JSON shape of the `GeminiOrder`
   Pydantic model (`response_schema` + `response_mime_type: application/json`).
   Per Google's best practice the JSON format is not duplicated inside the
   prompt; the field descriptions in `app/schemas.py` carry the semantics.
4. The answer is still validated in Python — never trust the model:
   quantities must be ≥ 1, hallucinated ids are moved to `unavailable`,
   duplicates are merged. Unknown requests come back in their base form
   (e.g. "dva hamburgera" → `hamburger`), never replaced by a similar item.

## Failure modes (what happens when things go wrong)

| Failure | Behaviour |
|---|---|
| A model is overloaded (Google's newest models occasionally answer `503 high demand`) | short backoff, one retry, then the next free model from the fallback chain — proven live |
| Free-tier rate limit (`429`) | longer backoff between attempts, fallback chain still tried; if every model answers `429`: `503` with a `Retry-After` header |
| API key invalid/blocked (`401`/`403`) | no pointless retries or fallbacks — the key is the problem, not the models: `500` with a clear "check GEMINI_API_KEY" message |
| Model returns malformed or nonsensical JSON | strict validation rejects it (retry + fallback), else `502` with the whole chain named |
| Blank/whitespace text, missing field | `422` |
| Utterance is a question or small talk, not an order | empty `items` and `unavailable` — nothing is invented |
| Gemini (or the network) is fully down | whole chain exhausted → `502`; the calling assistant tells the guest — conversation handling belongs to the caller, this service stays stateless |
| This service itself is down | the caller gets a connection error and degrades on its own (an order-by-phone assistant should say "technical difficulties", not hang up) |

The failure classification lives in `app/gemini.py: _failure_kind`, the
retry/fallback policy in `extract_order`; the endpoint maps the three error
classes to `500` / `503` / `502`.

## Verifying that it works

With the service running and `GEMINI_API_KEY` exported:

```bash
python check_examples.py
```

sends the three mandatory sentences from the task and compares the answers
with the expected results, plus two protocol checks (blank text → `422`,
`GET /health`). It prints `PASS`/`FAIL` per case and exits non-zero on any
failure, so it can also serve as a smoke test after deployment.

Without an API key (and without any network), the test suite covers the
endpoint behaviour, input validation, error paths, post-processing and the
exact parameters sent to Gemini:

```bash
python -m pytest
```

Manual checks, including behaviour on unusual input (a question instead of
an order returns empty lists; gibberish lands in `unavailable` at worst):

```bash
curl -s -X POST localhost:8000/order -H 'Content-Type: application/json' \
     -d '{"text": "imate li nešto bez mesa za nas dvoje?"}'
curl -s -X POST localhost:8000/order -H 'Content-Type: application/json' \
     -d '{"text": "dvije margarite i jednu colu"}'
curl -s -X POST localhost:8000/order -H 'Content-Type: application/json' -d '{"text": ""}'   # 422
```

## What could be improved

The biggest gain would be a small evaluation set of recorded guest sentences
(run nightly, e.g. with recorded Gemini responses) to catch prompt or model
regressions instead of relying on three examples. On the reliability side:
exponential backoff instead of a single retry, request logging/observability,
and a circuit breaker for the Gemini API. Natural next steps from the task
itself: answering "do you have something meat-free?" with suggestions from
the menu, handling the guest changing their mind mid-sentence, and
suggesting alternatives for unavailable items — all of which fit into the
same `response_schema` mechanism.
