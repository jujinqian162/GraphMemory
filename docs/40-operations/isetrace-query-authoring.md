# ISETrace LLM query authoring

`scripts/generate_isetrace_llm_queries.py` is a temporary offline dataset-authoring utility. It is not a Prefect stage and does not participate in the maintained experiment workflow. The sample commands below are pilot-only; the final natural-query test must consume frozen `QuerySpec` records from the fixed test split documented in [`isetrace-split.md`](isetrace-split.md), not resample the sample or full raw shards directly.

The script reuses the authoritative domain path:

```text
ISETrace JSONL
  -> CanonicalTrajectory
  -> ProvenanceGraph
  -> MotifSpec
  -> LLM-authored query text
```

The LLM never chooses answer nodes, support nodes, exact evidence source spans, dependency edges, or graph topology. Before any prompt is assembled, each `MotifQueryTarget` is frozen with answer/support output IDs and exact `(event_id, json_pointer, char_start, char_end)` spans from the selected provenance content units. The LLM writes only query text and an audit answer. `call_result` motifs are excluded by default so trivial single-call queries do not dominate the output.

## v5 one-shot authoring

Prompt v5 keeps the v4 one-stage authoring design and upgrades the stored label contract. Each request contains one shared
trajectory episode context and up to eight tasks by default. The episode context
contains user intents plus a bounded pool of up to 24 ToolOutput excerpts selected
from task evidence, temporal neighbors, and same-tool distractors. Tool-call
arguments are not repeated in the shared context. Each task marks its own answer
and support aliases because the same event may be an answer for one task and
context for another.

The system prompt includes explicit positive and negative examples. A good query
uses a broader user-purpose or non-answer event as its episode anchor while asking
for the core facts contained in the marked answer events. It explicitly rejects
questions about byte counts, generic write receipts, exact invocation parameters,
and other operational details that are easy to solve through lexical matching but
have little memory value.

By default, every task returns two materially different query candidates while
sharing one reference answer and grounding self-check:

```text
8 tasks/call x 2 queries/task = up to 16 candidate queries/call
```

The limits are 12 tasks per call and 3 queries per task. Local validation checks
the requested count, context anchors, forbidden literals, duplicate wording, and
mechanical paraphrases. Multiple queries from one task share the same motif and
gold labels; treat them as authoring candidates rather than independent benchmark
samples. The reviewed benchmark should normally retain one query per task.

## Runtime configuration

Create a local `.env` file:

```dotenv
MODEL_ID=...
API_KEY=...
BASE_URL=...
```

`.env` is ignored by Git. The script uses the OpenAI Responses protocol with strict JSON-schema output, `store=false`, one stable requested prompt-cache key, and up to eight same-trajectory authoring tasks per request by default. Raw API responses are cached locally by a complete request digest (endpoint, model, prompt, schema, and packet), so repeating the same run reuses responses instead of paying for duplicate annotation. Changing the model or prompt produces a different digest automatically.

## Inspect packets without network access

```bash
uv run python scripts/generate_isetrace_llm_queries.py \
  --source data/isetrace/sample/trajectories-00000.sample.jsonl \
  --output data/isetrace/query-authoring/pilot.jsonl \
  --limit 20 \
  --dry-run
```

This writes `pilot.jsonl.packets.jsonl`. Inspect these packets before spending API calls.

## Generate provisional records

```bash
uv run python scripts/generate_isetrace_llm_queries.py \
  --source data/isetrace/sample/trajectories-00000.sample.jsonl \
  --output data/isetrace/query-authoring/pilot.jsonl \
  --limit 100 \
  --per-trajectory 8 \
  --tasks-per-call 8 \
  --queries-per-task 2 \
  --api-retries 3 \
  --validation-retries 2
```

A `tqdm` bar tracks accepted query candidates. `--limit` is the requested number
of accepted query records, not the number of motif tasks. With the default
`--queries-per-task 2`, one accepted task yields two records. After a final
rejection the script continues sampling later motifs when the source contains
enough candidates.

Transient HTTP 429/5xx, network, timeout, malformed JSON, and non-object API responses are retried with bounded exponential backoff. An explicit model rejection is terminal: an ambiguous or semantically unrelated motif is not forced through a rewrite. Missing structured fields, failed answer-event self-checks, duplicate wording, leakage, and other local validation failures are returned to the model with the failure reason for a bounded rewrite. Each rewrite has a new request digest and is cached independently.

Outputs:

- `pilot.jsonl`: accepted `ProvenanceQueryExample` records;
- `pilot.jsonl.rejected.jsonl`: model and deterministic-validation rejections;
- `.pilot-cache/`: request-digest keyed raw Responses payloads, including cached rewrite attempts.

The output records preserve requested/reported model IDs, prompt version, request digest, response/cache metadata, token usage, gateway-instruction digest when present, and the motif-selected exact answer/support spans. Records produced by v3/v4 that contain only output IDs are deliberately invalid under the current contract; preparation never expands an output ID into a whole-output gold span.

## Annotation provenance

Every generated record is explicitly marked:

```text
annotation_method = llm_generated
human_review_status = unreviewed
```

LLM-generated records must not be described as human-annotated. They may be described as human-reviewed only after a real reviewer checks them and the stored status is updated to `accepted` or `edited`; rejected records must remain excluded. The audit answer emitted by the LLM is diagnostic only and never replaces motif-derived gold labels.

Prompt v5 supplies one bounded, shared trajectory context per request, marks
answer/support aliases per task, requests multiple distinct queries per task, and
includes concrete good/bad examples that favor meaningful user memory needs over
answer-token copying or tool-receipt trivia. It still supplies aliased dependency
direction and requires exact answer-event grounding and all-answer-event coverage.
Local checks reject leaked authoring aliases, typed answer literals, trivial
byte-count/invocation questions, invalid context anchors, duplicate wording, and
mechanical same-task paraphrases.

These checks reduce authoring errors but do not prove semantic grounding or make
two variants from one motif statistically independent. The utility remains suitable
for creating a review pool, not for claiming that all accepted records have passed
counterfactual support-minimality or independent human validation.
