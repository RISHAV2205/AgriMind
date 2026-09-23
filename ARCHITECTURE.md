# AgriMind Architecture

## 1. Goal

AgriMind is a multi-agent agricultural decision-support system. A farmer can submit
text, voice, images, and farm context; specialist agents produce evidence-backed
findings; one orchestrator converts those findings into a clear, safe action plan.

The first implemented capability is tomato leaf disease classification. It should
remain an independent, versioned ML capability rather than being embedded in a
general-purpose chat agent.

## 2. Recommended system shape

```text
Web / mobile client
        |
        v
API gateway + authentication
        |
        v
Conversation / task orchestrator
   |       |        |          |
   v       v        v          v
Disease  Weather  Crop-care  Market agents
agent    agent    agent      agent
   \       |        |          /
    \------ shared farm context ------/
                  |
                  v
        Recommendation + safety layer
                  |
                  v
     PostgreSQL / object storage / cache
```

Use a **supervisor-and-specialists** pattern:

- The orchestrator decides which agents to call and passes structured context.
- Specialist agents do one job and return typed results with confidence, sources,
  assumptions, and warnings.
- The recommendation layer combines results, checks safety rules, and writes the
  farmer-facing answer. Specialists should not directly issue final instructions.

This is more reliable, testable, and affordable than unrestricted agent-to-agent
chat.

## 3. Agent responsibilities

| Agent | Inputs | Output | Initial implementation |
|---|---|---|---|
| Orchestrator | User question, farm profile, conversation state | Execution plan and final response | LLM with deterministic routing rules |
| Disease agent | Leaf image, crop, location, growth stage | Disease probabilities, image-quality result, model version | Existing ResNet classifier exposed as a service |
| Weather agent | Farm location, time horizon | Forecast, risk events, farming implications | Weather provider adapter + rules |
| Crop-care agent | Crop, disease result, stage, local conditions | Non-chemical-first treatment options, prevention, escalation criteria | Curated agronomy knowledge base + LLM retrieval |
| Irrigation agent | Soil/crop profile, weather, irrigation history | Suggested irrigation window and amount/range | Rule-based first; sensor integrations later |
| Market agent | Crop, nearby market, harvest date, quantity | Price trend, market comparison, sell/hold rationale | Market-data adapter + analytics |
| Safety agent | Proposed recommendation, location | Blocked/revised unsafe advice and warnings | Deterministic policy/rules; optional LLM review |

## 4. Request flow

1. The client creates a `farm_query` with text/voice, optional image, and farm ID.
2. The API validates uploads and loads the farm profile (crop, district, acreage,
   language, planting date, and consented data).
3. The orchestrator classifies intent and creates an execution plan. It can run
   independent calls, such as disease and weather, concurrently.
4. Each agent returns a structured result; no free-form hidden reasoning is stored.
5. The recommendation layer joins evidence, applies safety rules, and produces an
   answer in the farmer's chosen language.
6. Store the query, agent outputs, final response, feedback, and model/knowledge
   versions for audit and continuous improvement.

For a diseased-leaf request, the execution plan is normally:

```text
image -> Disease agent -> Crop-care agent
                    \-> Weather agent (if location is available)
all evidence -> Safety layer -> farmer action plan
```

## 5. Service boundaries and repository layout

Start as a **modular monolith**. It is faster to build, deploy, and debug than
microservices; split high-load components later.

```text
AgriMind/
  apps/
    api/                 # FastAPI routes, auth, request validation
    worker/              # Long-running/background tasks
  core/
    orchestration/       # Routing, execution plans, response synthesis
    schemas/             # Pydantic request/result contracts
    safety/              # Agronomy and product-safety checks
    knowledge/           # Retrieval and citation handling
  agents/
    disease_agent/       # Existing ML model, training and inference
    weather_agent/
    crop_care_agent/
    irrigation_agent/
    market_agent/
  integrations/          # Weather, market, maps, notification adapters
  tests/
  docs/
```

Keep `agents/disease_agent/src` for training and model-specific inference. Add a
thin adapter in `agents/disease_agent/service.py` that accepts and returns the
shared schemas. Load the model once when the worker starts, not once per request.

## 6. Core data contracts

Every agent should return a contract similar to:

```json
{
  "agent": "disease",
  "status": "success",
  "confidence": 0.91,
  "data": {"label": "Tomato_Late_blight"},
  "evidence": [{"type": "model", "version": "resnet18-2026-09"}],
  "warnings": ["Confirm with a clear image of an affected leaf."],
  "requires_human_review": false
}
```

Important entities:

- `FarmProfile`: owner, location at district/coordinates level, crops, fields,
  planting dates, language, soil/sensor configuration.
- `FarmQuery`: input text, media references, current crop/field, timestamp.
- `AgentRun`: selected agent, input/output JSON, latency, status, versions.
- `Recommendation`: prioritized actions, rationale, citations, warnings,
  acknowledgement/feedback.
- `ModelRegistry`: model version, classes, metrics, training data lineage,
  approval status.

## 7. Storage and infrastructure

- **PostgreSQL**: users, farms, queries, agent runs, recommendations, feedback.
- **Object storage**: uploaded images, audio, trained model artifacts. Store object
  references in PostgreSQL, not blobs.
- **Redis**: cache weather/market responses; queue background jobs.
- **Worker queue**: image inference, notification delivery, report generation, and
  slow external API calls.
- **Vector index**: curated agricultural guidance with document metadata, region,
  crop, review date, and source. It is a retrieval component, not a source of truth.

Deploy API and worker separately. The disease model can initially run in the worker
process; move it to its own inference service only when GPU scaling or independent
deployment demands it.

## 8. Safety, quality, and trust

- Never present a classifier result as certain. Include confidence and request a
  better image or expert review below a defined threshold.
- Separate diagnosis from pesticide guidance. Product, dose, and legality must be
  validated for the farmer's region and crop; when unavailable, provide only
  general integrated-pest-management guidance and escalation steps.
- Show the evidence used: model version, weather timestamp, and knowledge sources.
- Validate image type/size, scan uploads, enforce authentication and rate limits.
- Encrypt private farmer data, minimize location precision where possible, and keep
  explicit consent for data used in model improvement.
- Log structured agent outcomes, latency, errors, confidence calibration, and user
  feedback. Avoid storing hidden chain-of-thought.

## 9. Phased implementation

### Phase 1 — usable disease assistant

Build FastAPI, image upload, a `/v1/diagnose` endpoint, the disease adapter,
PostgreSQL records, and a simple farmer response. Add image-quality checks,
confidence thresholds, and model-version logging.

### Phase 2 — multi-agent recommendations

Add farm profiles, the orchestrator, weather adapter, crop-care retrieval, safety
rules, multilingual answers, and feedback collection.

### Phase 3 — operational farm intelligence

Add irrigation planning, market insights, notifications, sensor data, scheduled
risk checks, analytics, and human agronomist escalation.

## 10. Initial API surface

```text
POST /v1/queries                 Submit text, voice, and/or image query
POST /v1/diagnoses/leaf          Direct leaf diagnosis (useful for the UI)
GET  /v1/queries/{id}            Query, agent statuses, final recommendation
GET  /v1/farms/{id}/profile      Farm context
PUT  /v1/farms/{id}/profile      Update farm context
POST /v1/recommendations/{id}/feedback
GET  /health
```

## 11. First technology choices

- Python 3.11+, FastAPI, Pydantic, SQLAlchemy/Alembic.
- PyTorch/TorchVision for the existing disease model.
- PostgreSQL, Redis, S3-compatible object storage.
- React/Next.js or Flutter client, depending on whether web-first or mobile-first
  is the product priority.
- Docker Compose for local development; managed containers and managed PostgreSQL
  for production.

The key rule: use deterministic code for routing, data validation, safety limits,
and calculations. Use LLMs for language understanding, retrieval-assisted
explanations, and converting verified results into a farmer-friendly response.
