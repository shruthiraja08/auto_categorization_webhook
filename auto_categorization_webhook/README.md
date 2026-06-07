# Auto Categorization Webhook

A production-ready webhook that automatically classifies support tickets into predefined categories, subcategories, priorities, and confidence scores using OpenAI's `gpt-4o` with few-shot learning and structured outputs.

## Architecture & Workflow

1. **API Layer (`app.py`)**: A FastAPI application providing `/classify` and `/health` endpoints. It enforces `X-API-Key` authentication and standardizes error responses.
2. **Orchestration Layer (`classifier.py`)**: Manages the interaction with the OpenAI API. Uses `AsyncOpenAI` for non-blocking I/O, `tenacity` for exponential backoff on transient errors, and structured output parsing to enforce the `ClassificationResult` schema.
3. **Prompting Layer (`prompt_builder.py`)**: Loads `samples.json` at startup to dynamically construct few-shot examples, teaching the LLM the expected taxonomy and JSON structure.
4. **Data/Config Layers (`schemas.py`, `config.py`)**: Defines Pydantic models for strictly validated request/response boundaries and uses `pydantic-settings` to load configuration from the environment/`.env` file.

## Requirements

- Python 3.11+
- OpenAI API Key (`OPENAI_API_KEY`)
- Webhook Secret Key (`API_KEY`)

## Local Setup

1. **Clone & Environment**:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .\.venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. **Configuration**:
   Copy the example environment file and fill in your keys.
   ```bash
   cp .env.example .env
   ```

3. **Run the Application**:
   ```bash
   uvicorn app:app --reload
   ```

## Docker Setup

A multi-stage Dockerfile is included for lean, production-ready deployments.

```bash
docker build -t auto-categorization-webhook .
docker run -p 8000:8000 --env-file .env auto-categorization-webhook
```

## API Usage

### 1. Health Check
```bash
curl http://localhost:8000/health
```

### 2. Classify Ticket
Requires the `X-API-Key` header.

```bash
curl -X POST http://localhost:8000/classify \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your_super_secret_webhook_key_123" \
  -d '{
    "ticket_id": "T-1045",
    "title": "Cannot access billing portal",
    "description": "Every time I click on the billing tab it redirects me back to the dashboard."
  }'
```

**Example Response:**
```json
{
  "request_id": "b9a7c3a0-4f51-4d1d-91b3-6c84f9a0c4f3",
  "ticket_id": "T-1045",
  "category": "Billing",
  "subcategory": "Portal Access",
  "priority": "High",
  "confidence": 0.96,
  "low_confidence": false,
  "latency_ms": 840,
  "classified_at": "2024-05-12T10:45:00Z",
  "model": "gpt-4o-2024-05-13"
}
```

## Logging & Audit

- **Application Logs**: Streamed to `stdout` in a consistent format.
- **Audit Logs**: Every prediction is written as a JSONL record to `logs/predictions.log` asynchronously, ensuring disk I/O does not block the FastAPI event loop.

## Testing

The project includes a comprehensive `pytest` suite with `pytest-asyncio` and `pytest-mock`.

```bash
pytest tests/
```

Test coverage targets >80% and covers the HTTP API, OpenAI mock interactions, few-shot prompt building, and config validation edge cases.
