# NL2SQL API

A lightweight REST API that converts natural language questions into SQL queries using HuggingFace causal language models. Built with FastAPI and the HuggingFace `transformers` library.

---

## Requirements

- Python 3.10+
- A HuggingFace account and API token (for gated/private models)
- GPU recommended for reasonable inference speed

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Running the Server

```bash
uvicorn main:app --host 0.0.0.0 --port 5000 --reload
```

The API will be available at `http://localhost:5000`.  

---

## API Reference

### `POST /generate`

Generates a SQL query from a natural language question.

**Request body** (`application/json`):

| Field | Type | Required | Description |
|---|---|---|---|
| `model_name` | string | Yes | HuggingFace model ID (e.g. `MPX0222forHF/SQL-R1-7B`) |
| `question` | string | Yes | Natural language question to convert to SQL |
| `ddl` | string | Yes | DDL schema string (CREATE TABLE statements) |
| `dialect` | string | No | SQL dialect — Currently supports `SQLite` and `MySQL` only|
| `data` | string | No | Optional sample data for additional context |

**Response body**:

| Field | Type | Description |
|---|---|---|
| `sql` | string | Generated and formatted SQL query |
| `model_name` | string | Model used for generation |
| `question` | string | Original question |
| `dialect` | string | SQL dialect used |
| `elapsed` | float | Inference time in seconds |

---

## Example Usage

### Python (`requests`)

```python
import requests

api_url = "http://localhost:5000/generate"

with open("schema.txt", "r") as f:
    schema = f.read()

payload = {
    "model_name": "MPX0222forHF/SQL-R1-7B",
    "question": "Who is the best performing student by peer review score?",
    "ddl": schema,
    "dialect": "SQLite"
}

response = requests.post(api_url, json=payload)
print(response.json()["sql"])
```

### cURL

```bash
curl -X POST "http://localhost:5000/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "model_name": "MPX0222forHF/SQL-R1-7B",
    "question": "Who is the best performing student by peer review score?",
    "ddl": "CREATE TABLE Employees (...); CREATE TABLE Performance_Metrics (...);",
    "dialect": "SQLite"
  }'
```

### Sample Response

```json
{
  "sql": "WITH RankedStudents AS (\n  SELECT e.employee_id, ...\n)\nSELECT ...\nFROM RankedStudents\nWHERE rank = 1;",
  "model_name": "MPX0222forHF/SQL-R1-7B",
  "question": "Who is the best performing student by peer review score?",
  "dialect": "SQLite",
  "elapsed": 14.32
}
```

---

## Project Structure

```
.
├── main.py            # FastAPI app and route definitions
├── nl2sql.py          # Core logic: model loading, prompt building, inference, SQL extraction
├── requirements.txt   # Python dependencies
└── API_Usage.ipynb    # Example notebook for testing the API
```

---

## Notes

- **Model loading**: Models are loaded fresh on every request. For production use, consider caching loaded models in memory (e.g. using a dict keyed by `model_name`).
- **HuggingFace token**: The token is currently hardcoded in `nl2sql.py`. It is recommended to move it to an environment variable or a `.env` file before sharing or deploying.
- **Inference time**: Generation can take more 10 minutes. A CUDA-enabled GPU is strongly recommended.
- **Schema quality**: Query accuracy is directly tied to the completeness and clarity of the DDL passed in the `ddl` field. Include all relevant tables and foreign key relationships.
