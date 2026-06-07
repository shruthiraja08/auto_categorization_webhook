# auto_categorization_webhook
# Auto Categorization Webhook

## Overview

Auto Categorization Webhook is a FastAPI-based intelligent ticket classification system that automatically categorizes support tickets using OpenAI GPT-4o. The system analyzes ticket titles and descriptions and predicts the appropriate category, subcategory, priority level, and confidence score.

This project helps organizations reduce manual effort in ticket triaging and improves response efficiency by automating support ticket classification.

---

## Features

* Automatic ticket categorization
* Subcategory prediction
* Priority assignment (Low, Medium, High)
* Confidence score generation
* OpenAI GPT-4o integration
* Secure API key authentication
* Structured JSON responses
* Audit logging for predictions
* FastAPI REST API endpoints
* Docker support for deployment
* Unit testing with Pytest

---

## Project Structure

```text
auto_categorization_webhook/
│
├── app.py
├── classifier.py
├── config.py
├── prompt_builder.py
├── schemas.py
├── samples.json
├── requirements.txt
├── Dockerfile
├── README.md
│
├── logs/
├── tests/
├── ui/
└── .env
```

---

## Technologies Used

* Python 3.11+
* FastAPI
* OpenAI GPT-4o
* Pydantic
* AsyncOpenAI
* Uvicorn
* Pytest
* Docker

---

## System Workflow

1. User submits a support ticket.
2. FastAPI receives the request.
3. Prompt Builder prepares few-shot examples.
4. OpenAI GPT-4o analyzes the ticket.
5. The system predicts:

   * Category
   * Subcategory
   * Priority
   * Confidence Score
6. Response is returned as JSON.
7. Prediction details are logged for auditing.

---

## API Endpoints

### Health Check

```http
GET /health
```

### Ticket Classification

```http
POST /classify
```

Request Example:

```json
{
  "ticket_id": "T-1045",
  "title": "Cannot access billing portal",
  "description": "Every time I click on the billing tab it redirects me back to the dashboard."
}
```

Response Example:

```json
{
  "ticket_id": "T-1045",
  "category": "Billing",
  "subcategory": "Portal Access",
  "priority": "High",
  "confidence": 0.96
}
```

---

## Installation

### Clone Repository

```bash
git clone https://github.com/shruthiraja08/auto_categorization_webhook.git
cd auto_categorization_webhook
```

### Create Virtual Environment

```bash
python -m venv .venv
```

### Activate Environment

Windows:

```bash
.venv\Scripts\activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run Application

```bash
uvicorn app:app --reload
```

---

## Testing

Run all tests using:

```bash
pytest tests/
```

---

## Future Enhancements

* Multi-language ticket classification
* Dashboard for analytics
* Email integration
* Real-time monitoring
* Ticket routing automation

---
