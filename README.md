# Order extraction service

A small FastAPI service that is one component of a phone-based restaurant
order assistant: given a sentence a guest actually says (in Croatian), it
extracts **what** was ordered and **in what quantity**, using a free Gemini
model together with the restaurant menu (`jelovnik.json`).

```
POST /order {"text": "dvije margarite i jednu colu"}

{"items": [{"id": "margarita", "quantity": 2},
           {"id": "coca_cola", "quantity": 1}],
 "unavailable": [],
 "suggestions": []}
```

Items that exist on the menu come back in `items`, referenced by menu `id`.
Anything the guest asks for that is **not** on the menu comes back separately
in `unavailable` — it is never dropped and never silently replaced with a
similar menu item:

```
POST /order {"text": "dva hamburgera i jednu margaritu"}

{"items": [{"id": "margarita", "quantity": 1}],
 "unavailable": [{"text": "hamburger", "quantity": 2}],
 "suggestions": []}
```

**Response format extension — `suggestions`.** The proposed format is
extended by exactly one field, with this reason: the optional case from the
task where the guest asks what is available ("imate li nešto bez mesa za nas
dvoje?") needs somewhere to go — it is not an order, so `items` and
`unavailable` stay empty and the matching menu ids come back in
`suggestions` (meat-free **food**, no drinks), with nothing invented:

```
POST /order {"text": "imate li nešto bez mesa za nas dvoje?"}

{"items": [], "unavailable": [],
 "suggestions": ["margarita", "vegetariana", "quattro_formaggi", "mijesana_salata"]}
```

The second optional case — the guest changing their mind mid-sentence —
needs no format change at all ("tri margarite i colu, ma ne, ipak dvije
margarite" → margarita × 2, the un-corrected cola stands). All five
sentences from the task are asserted by `check_examples.py`.

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
| `GEMINI_SIMULATE_FAILURE` | not set | Fault injection for demos: `auth` / `quota` / `transient` — the endpoint answers exactly as it would for that real failure (500 / 503+Retry-After / 502), deterministically. Never enable in production. |
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

## How the extraction works

1. `POST /order` validates the body: a blank or missing `text` answers `422`.
2. The menu and the guest utterance go to the Gemini model together with a
   system prompt (`app/prompts/system_prompt.md`) that fixes the rules:
   match Croatian word forms to menu ids, never invent ids, never replace a
   missing item with a similar one, unknown requests go to `unavailable`
   with the guest's own words, a question about what is available (e.g.
   "bez mesa?") is answered with `suggestions` instead of an order, and when
   the guest changes their mind mid-sentence the final statement wins for
   the corrected part while everything un-corrected stands.
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
| A call stalls instead of erroring (observed live) | cut off after 60 s — an explicit per-call HTTP timeout (the SDK has none by default), then handled like any transient failure: retry, next model |
| Blank/whitespace text, missing field | `422` |
| Utterance is a question about what is available (e.g. "bez mesa?") | empty order + matching menu ids in `suggestions` |
| Utterance is small talk, not an order at all | empty `items`, `unavailable` and `suggestions` — nothing is invented |
| The guest changes their mind mid-sentence | the final statement wins for the corrected part, everything un-corrected stands |
| Gemini (or the network) is fully down | whole chain exhausted → `502`; the calling assistant tells the guest — conversation handling belongs to the caller, this service stays stateless |
| This service itself is down | the caller gets a connection error and degrades on its own (an order-by-phone assistant should say "technical difficulties", not hang up) |

The failure classification lives in `app/gemini.py: _failure_kind`, the
retry/fallback policy in `extract_order`; the endpoint maps the three error
classes to `500` / `503` / `502`.

Every failure mode can be shown deterministically, without waiting for a
real outage — fault injection via `GEMINI_SIMULATE_FAILURE`:

```bash
GEMINI_SIMULATE_FAILURE=quota .venv/bin/uvicorn app.main:app --port 8101
curl -si -X POST localhost:8101/order -H 'Content-Type: application/json' \
     -d '{"text": "dvije margarite"}'    # -> 503 + Retry-After
```

or the whole story in one command (see `demo.py`): the five sentences from
the task, the protocol edge cases, unusual input, and all three failure
modes with their exact answers.

## Verifying that it works

With the service running and `GEMINI_API_KEY` exported:

```bash
python check_examples.py
```

sends the five sentences from the task (the three mandatory ones plus the
two optional cases) and compares the answers with the expected results,
plus two protocol checks (blank text → `422`, `GET /health`). It prints
`PASS`/`FAIL` per case and exits non-zero on any failure, so it can also
serve as a smoke test after deployment.

For presentations, `demo.py` goes further — the same five sentences
plus protocol edge cases, unusual input (a question about an item,
politeness, gibberish) and **all three failure modes shown deterministically**
via `GEMINI_SIMULATE_FAILURE` (rejected key → `500`, exhausted free-tier
quota → `503` + `Retry-After`, Gemini unreachable → `502`):

```bash
python demo.py        # needs the normal service running (with a real key)
```

Without an API key (and without any network), the test suite covers the
endpoint behaviour, input validation, error paths, post-processing and the
exact parameters sent to Gemini:

```bash
python -m pytest
```

Manual checks, including behaviour on unusual input (the meat-free question
returns `suggestions`, not an order; gibberish lands in `unavailable` at
worst):

```bash
curl -s -X POST localhost:8000/order -H 'Content-Type: application/json' \
     -d '{"text": "imate li nešto bez mesa za nas dvoje?"}'
curl -s -X POST localhost:8000/order -H 'Content-Type: application/json' \
     -d '{"text": "dvije margarite i jednu colu"}'
curl -s -X POST localhost:8000/order -H 'Content-Type: application/json' -d '{"text": ""}'   # 422
```

## What could be improved

The biggest gain would be a small evaluation set of recorded guest sentences
with expected results, run regularly — a pass rate would turn "it seems to
work" into a measured number and catch prompt regressions the way
`check_examples.py` cannot. On the reliability side: exponential backoff and
a circuit breaker for the Gemini API. The natural next extension from the
task: suggesting alternatives for unavailable items through the same
`suggestions` mechanism.
