from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import os
import re
import json
import time
import httpx
import mysql.connector
from dotenv import load_dotenv

load_dotenv()

from nl2sql import load_model, build_prompt, tokenize_prompt, generate_output, extract_sql, translate_sql, format_sql


app = FastAPI(
    title="NL2SQL API",
    description="Convert NL questions to SQL queries using HuggingFace models.",
    version="2.0.0"
)

_origins_env = os.getenv("ALLOWED_ORIGINS", "*")
allowed_origins = ["*"] if _origins_env == "*" else [o.strip() for o in _origins_env.split(",")]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── MySQL Config ──
DB_CONFIG = {
    "host": os.getenv("DB_HOST", "127.0.0.1"),
    "port": int(os.getenv("DB_PORT", 3306)),
    "user": os.getenv("DB_USER", "nsrivast"),
    "password": os.getenv("DB_PASSWORD", "ns#601"),
    "database": os.getenv("DB_NAME", "doppw"),
}

# ── OpenRouter Config ──
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "nvidia/nemotron-3-super-120b-a12b:free")
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

# ── DDL schema for NL2SQL ──
ALL_PENSIONERS_DDL = """
CREATE TABLE all_pensioners (
    state TEXT COMMENT 'State name in UPPERCASE e.g. KARNATAKA, UTTAR PRADESH, MADHYA PRADESH, TAMIL NADU, ANDHRA PRADESH, MAHARASHTRA, DELHI, BIHAR, ASSAM, ODISHA, WEST BENGAL, RAJASTHAN, GUJARAT, PUNJAB, KERALA, JHARKHAND, HARYANA, CHHATTISGARH, TELANGANA, JAMMU AND KASHMIR, GOA, UTTARAKHAND, HIMACHAL PRADESH, TRIPURA, MEGHALAYA, MANIPUR, NAGALAND, ARUNACHAL PRADESH, MIZORAM, SIKKIM, PUDUCHERRY, CHANDIGARH, ANDAMAN AND NICOBAR ISLANDS, DADRA AND NAGAR HAVELI, DAMAN AND DIU, LAKSHADWEEP, LADAKH',
    district TEXT COMMENT 'District name in UPPERCASE',
    pensioner_pincode TEXT,
    branch_pincode TEXT,
    bank_name TEXT COMMENT 'Bank name in lowercase e.g. state bank of india, punjab national bank, union bank of india, canara bank, bank of baroda',
    pensioner_type TEXT COMMENT 'Values: central, state, other (stored in lowercase)',
    pensioner_subtype TEXT,
    YOB INT COMMENT 'Year of birth of pensioner',
    lc_date TEXT COMMENT 'Life Certificate date - NOT NULL means DLC completed, NULL means DLC pending',
    pensioner_DLC_type TEXT COMMENT 'DLC verification method: p=fingerprint, f=face, i=iris',
    last_year_lc_type TEXT COMMENT 'Previous year LC type: DLC or PLC'
);"""

# ── OpenRouter system prompt ──
SYSTEM_PROMPT = """You are Alan AI, a STRICTLY DLC Pension Dashboard assistant. You ONLY help with DLC (Digital Life Certificate) pension data queries.

Database schema:
""" + ALL_PENSIONERS_DDL + """

SUPPORTED QUERY DIMENSIONS (ONLY these are valid filter axes):
- State: filter by state name (e.g. Bihar, UP, Maharashtra)
- District: filter by district name within a state
- Pincode: filter by 6-digit pensioner pincode
- Bank: filter by bank_name (e.g. SBI, PNB, Canara)
- Age category: derived from YOB — Under 60 (YOB>YEAR-60), 60-70, 70-80, 80-90, 90-100, 100+
- Pensioner type: pensioner_type (central / state / other)
- Pensioner subtype: pensioner_subtype (autonomous, civil, railway, postal, defence, telecom)

STRICT RULES:
1. ONLY answer questions that filter by the SUPPORTED DIMENSIONS above.
2. If the query mentions an institution/organization/college/department NOT mappable to state/district/pincode/bank/type — respond with intent=chat and explain which dimensions ARE supported. Example: "IIT Kanpur", "Ministry of Finance", "AIIMS" → these are not queryable dimensions. Reply: "I can only answer queries based on state, district, pincode, bank, age category, or pensioner type. IIT Kanpur is not a queryable dimension in this dataset."
3. For ANY off-topic question (general knowledge, personal chat, jokes, etc.) — politely decline. Set intent=chat.
4. Simple greetings (hi, hello, namaste) are OK — respond briefly.
5. Use UPPER(state) for state comparisons e.g. WHERE UPPER(state) = 'BIHAR'
6. DLC done = COUNT(lc_date), pending = COUNT(*)-COUNT(lc_date), completion_rate = ROUND(COUNT(lc_date)*100.0/COUNT(*),1)
7. Conversion potential formula: ROUND(SUM(CASE WHEN lc_date IS NULL AND last_year_lc_type='DLC' THEN 0 ELSE 1 END)*100.0/COUNT(*),1)
   Use ALWAYS when asked about "conversion potential", "plc to dlc", or "conversion rate".
8. For "top" queries include counts AND completion rate, order by completion_rate DESC
9. Limit to 10 unless user specifies otherwise
10. NEVER generate INSERT, UPDATE, DELETE, DROP, ALTER
11. Reply in same language as user (Hindi/Hinglish/English)
12. Keep replies concise

You MUST respond with ONLY a JSON object, nothing else:
{"intent":"chat","reply":"your message","sql":null}
or
{"intent":"data","reply":"fetching data...","sql":"SELECT ..."}"""


async def call_openrouter(question: str) -> dict:
    """Call OpenRouter API with the user's question. Returns parsed JSON response."""
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://cdis.iitk.ac.in/dlc-dashboard-test",
        "X-Title": "DLC Pension Dashboard"
    }

    payload = {
        "model": OPENROUTER_MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question}
        ],
        "temperature": 0.2,
        "max_tokens": 512
    }

    async with httpx.AsyncClient(timeout=60.0) as client:
        resp = await client.post(OPENROUTER_URL, headers=headers, json=payload)
        resp.raise_for_status()
        data = resp.json()

    content = data["choices"][0]["message"]["content"]

    # Parse the JSON response from the model
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        # Try to extract JSON from the response if it's wrapped in markdown
        json_match = re.search(r'\{.*\}', content, re.DOTALL)
        if json_match:
            parsed = json.loads(json_match.group())
        else:
            parsed = {"intent": "chat", "reply": content, "sql": None}

    return parsed


def execute_sql(sql: str) -> list:
    """Execute a SELECT query on MySQL and return rows."""
    conn = mysql.connector.connect(**DB_CONFIG)
    cursor = conn.cursor(dictionary=True)
    cursor.execute(sql)
    rows = cursor.fetchall()
    cursor.close()
    conn.close()
    return rows


NL2SQL_MODEL = os.getenv("NL2SQL_MODEL", "MPX0222forHF/SQL-R1-7B")


class NL2SQLRequest(BaseModel):
    model_name: str
    question: str
    ddl: str
    data: Optional[str] = None
    dialect: str = 'SQLite'

class NL2SQLResponse(BaseModel):
    sql: str
    model_name: str
    question: str
    dialect: str
    elapsed: float

class AskAIRequest(BaseModel):
    question: str


@app.post("/generate", tags=["NL2SQL"])
def generate_sql_endpoint(req: NL2SQLRequest):
    tokenizer, model = load_model(req.model_name)
    prompt = build_prompt(req.question, req.ddl, req.dialect)
    tokenized_prompt = tokenize_prompt(tokenizer, model, prompt)
    decoded_output, elapsed_time = generate_output(tokenizer, model, tokenized_prompt)
    sql = extract_sql(decoded_output)
    tsql = translate_sql(sql, req.dialect)
    fsql = format_sql(tsql)
    response = NL2SQLResponse(sql=fsql,
                              model_name=req.model_name,
                              question=req.question,
                              dialect=req.dialect,
                              elapsed=elapsed_time)
    return response


# ── Pre-built queries for common patterns ──
STATE_ALIASES = {
    'uttarpradesh': 'UTTAR PRADESH', 'up': 'UTTAR PRADESH', 'uttar pradesh': 'UTTAR PRADESH',
    'madhyapradesh': 'MADHYA PRADESH', 'mp': 'MADHYA PRADESH', 'madhya pradesh': 'MADHYA PRADESH',
    'andhrapradesh': 'ANDHRA PRADESH', 'ap': 'ANDHRA PRADESH', 'andhra pradesh': 'ANDHRA PRADESH',
    'tamilnadu': 'TAMIL NADU', 'tn': 'TAMIL NADU', 'tamil nadu': 'TAMIL NADU',
    'westbengal': 'WEST BENGAL', 'wb': 'WEST BENGAL', 'west bengal': 'WEST BENGAL',
    'himachalpradesh': 'HIMACHAL PRADESH', 'hp': 'HIMACHAL PRADESH', 'himachal pradesh': 'HIMACHAL PRADESH',
    'arunachalpradesh': 'ARUNACHAL PRADESH', 'arunachal pradesh': 'ARUNACHAL PRADESH',
    'j&k': 'JAMMU AND KASHMIR', 'jk': 'JAMMU AND KASHMIR', 'jammu kashmir': 'JAMMU AND KASHMIR', 'jammu and kashmir': 'JAMMU AND KASHMIR',
    'karnataka': 'KARNATAKA', 'maharashtra': 'MAHARASHTRA', 'delhi': 'DELHI',
    'bihar': 'BIHAR', 'assam': 'ASSAM', 'odisha': 'ODISHA', 'orissa': 'ODISHA',
    'rajasthan': 'RAJASTHAN', 'gujarat': 'GUJARAT', 'punjab': 'PUNJAB',
    'kerala': 'KERALA', 'jharkhand': 'JHARKHAND', 'haryana': 'HARYANA',
    'chhattisgarh': 'CHHATTISGARH', 'chattisgarh': 'CHHATTISGARH',
    'telangana': 'TELANGANA', 'goa': 'GOA', 'uttarakhand': 'UTTARAKHAND',
    'tripura': 'TRIPURA', 'meghalaya': 'MEGHALAYA', 'manipur': 'MANIPUR',
    'nagaland': 'NAGALAND', 'mizoram': 'MIZORAM', 'sikkim': 'SIKKIM',
    'puducherry': 'PUDUCHERRY', 'chandigarh': 'CHANDIGARH',
}

def resolve_state(text):
    t = text.lower().strip()
    for alias in sorted(STATE_ALIASES.keys(), key=len, reverse=True):
        # Short aliases (2-3 chars) need word boundary to avoid false matches
        # e.g. "mp" in "computer", "up" in "update", "goa" in "goal"
        if len(alias) <= 3:
            if re.search(r'\b' + re.escape(alias) + r'\b', t):
                return STATE_ALIASES[alias]
        else:
            if alias in t:
                return STATE_ALIASES[alias]
    return None

def extract_limit(text):
    m = re.search(r'(?:top|first|best|bottom|worst|last)\s+(\d+)', text.lower())
    return int(m.group(1)) if m else None

def match_common_query(question: str):
    """Check if question matches a common pattern and return pre-built SQL."""
    q = question.lower().strip()
    limit = extract_limit(q) or 10
    state = resolve_state(q)

    # Detect institution/org queries that are NOT valid dimensions — return sentinel
    INSTITUTION_PATTERNS = r'\b(iit|aiims|nit|iisc|iim|iiit|cpao|epfo|ministry|mantralaya|department|mantri|samiti|sabha|university|college|institute|organisation|organization)\b'
    if re.search(INSTITUTION_PATTERNS, q) and not state:
        return "__INVALID_DIMENSION__"

    if re.search(r'(how\s+many|total|count).*(pensioner|record)', q) and not state and not re.search(r'(state|district|bank|wise)', q):
        return "SELECT COUNT(*) AS total_pensioners FROM all_pensioners"

    # Conversion potential query for a specific state
    if state and re.search(r'(conversion\s*potential|plc\s*to\s*dlc|conversion\s*rate)', q):
        return f"""
            SELECT COUNT(*) AS total_pensioners,
                   COUNT(lc_date) AS dlc_done,
                   COUNT(*) - COUNT(lc_date) AS dlc_pending,
                   ROUND(SUM(CASE WHEN lc_date IS NULL AND last_year_lc_type='DLC' THEN 0 ELSE 1 END)*100.0/COUNT(*),1) AS conversion_potential_pct
            FROM all_pensioners WHERE UPPER(state) = '{state}'
        """

    # Conversion potential overall (no state)
    if re.search(r'(overall|india|national|total).*(conversion\s*potential|plc\s*to\s*dlc)', q) or \
       (re.search(r'(conversion\s*potential|plc\s*to\s*dlc)', q) and not state):
        return """
            SELECT COUNT(*) AS total_pensioners,
                   COUNT(lc_date) AS dlc_done,
                   COUNT(*) - COUNT(lc_date) AS dlc_pending,
                   ROUND(SUM(CASE WHEN lc_date IS NULL AND last_year_lc_type='DLC' THEN 0 ELSE 1 END)*100.0/COUNT(*),1) AS conversion_potential_pct
            FROM all_pensioners WHERE state IS NOT NULL AND state != 'null'
        """

    if state and re.search(r'(how\s+many|total|count|number).*(pensioner|record)', q) and not re.search(r'district', q):
        return f"SELECT COUNT(*) AS total_pensioners, COUNT(lc_date) AS dlc_done, ROUND(COUNT(lc_date)*100.0/COUNT(*),1) AS completion_rate FROM all_pensioners WHERE UPPER(state) = '{state}'"

    if state and re.search(r'district', q):
        order = 'ASC' if re.search(r'(least|lowest|minimum|worst|kam|sabse\s*kam|bottom)', q) else 'DESC'
        return f"""
            SELECT district, COUNT(*) AS total_pensioners, COUNT(lc_date) AS dlc_done,
                   ROUND(COUNT(lc_date) * 100.0 / COUNT(*), 1) AS completion_rate
            FROM all_pensioners WHERE UPPER(state) = '{state}' AND district IS NOT NULL AND district != 'null'
            GROUP BY district ORDER BY completion_rate {order}, total_pensioners DESC LIMIT {limit}
        """

    # Pincodes in a state — best or worst
    if state and re.search(r'pincode', q):
        order = 'ASC' if re.search(r'(least|lowest|minimum|worst|kam|sabse\s*kam|bottom)', q) else 'DESC'
        return f"""
            SELECT pensioner_pincode AS pincode, COUNT(*) AS total_pensioners, COUNT(lc_date) AS dlc_done,
                   ROUND(COUNT(lc_date) * 100.0 / COUNT(*), 1) AS completion_rate
            FROM all_pensioners WHERE UPPER(state) = '{state}'
              AND pensioner_pincode IS NOT NULL AND pensioner_pincode != '' AND pensioner_pincode != 'null'
            GROUP BY pensioner_pincode ORDER BY completion_rate {order}, total_pensioners DESC LIMIT {limit}
        """

    if re.search(r'top\s*\d*\s*(state|performing\s+state|best\s+state)', q):
        return f"""
            SELECT state, COUNT(*) AS total_pensioners, COUNT(lc_date) AS dlc_done,
                   ROUND(COUNT(lc_date) * 100.0 / COUNT(*), 1) AS completion_rate
            FROM all_pensioners WHERE state IS NOT NULL AND state != 'null'
            GROUP BY state ORDER BY completion_rate DESC, total_pensioners DESC LIMIT {limit}
        """

    if re.search(r'top\s*\d*\s*(bank|performing\s+bank|best\s+bank)', q):
        return f"""
            SELECT bank_name, COUNT(*) AS total_pensioners, COUNT(lc_date) AS dlc_done,
                   ROUND(COUNT(lc_date) * 100.0 / COUNT(*), 1) AS completion_rate
            FROM all_pensioners WHERE bank_name IS NOT NULL AND bank_name != 'null'
            GROUP BY bank_name ORDER BY completion_rate DESC, total_pensioners DESC LIMIT {limit}
        """

    if re.search(r'state[\s-]?wise', q) and re.search(r'(conversion\s*potential|plc\s*to\s*dlc)', q):
        return """
            SELECT state, COUNT(*) AS total_pensioners,
                   COUNT(lc_date) AS dlc_done,
                   COUNT(*) - COUNT(lc_date) AS dlc_pending,
                   ROUND(SUM(CASE WHEN lc_date IS NULL AND last_year_lc_type='DLC' THEN 0 ELSE 1 END)*100.0/COUNT(*),1) AS conversion_potential_pct
            FROM all_pensioners WHERE state IS NOT NULL AND state != 'null'
            GROUP BY state ORDER BY conversion_potential_pct DESC
        """

    if re.search(r'state[\s-]?wise', q):
        return """
            SELECT state, COUNT(*) AS total_pensioners, COUNT(lc_date) AS dlc_done,
                   COUNT(*) - COUNT(lc_date) AS dlc_pending,
                   ROUND(COUNT(lc_date) * 100.0 / COUNT(*), 1) AS completion_rate
            FROM all_pensioners WHERE state IS NOT NULL AND state != 'null'
            GROUP BY state ORDER BY completion_rate DESC
        """

    if state:
        return f"""
            SELECT COUNT(*) AS total_pensioners, COUNT(lc_date) AS dlc_done,
                   COUNT(*) - COUNT(lc_date) AS dlc_pending,
                   ROUND(COUNT(lc_date) * 100.0 / COUNT(*), 1) AS completion_rate
            FROM all_pensioners WHERE UPPER(state) = '{state}'
        """

    return None


def fallback_local_model(question: str):
    """Use local SQL-R1-7B model as fallback when OpenRouter is unavailable."""
    tokenizer, model = load_model(NL2SQL_MODEL)
    prompt = build_prompt(question, ALL_PENSIONERS_DDL, "MySQL")
    tokenized_prompt = tokenize_prompt(tokenizer, model, prompt)
    decoded_output, elapsed_time = generate_output(tokenizer, model, tokenized_prompt)
    sql = extract_sql(decoded_output)
    tsql = translate_sql(sql, "MySQL")
    fsql = format_sql(tsql)
    return fsql, elapsed_time


@app.post("/ask-ai", tags=["Ask AI"])
async def ask_ai(req: AskAIRequest):
    question = req.question.strip()
    if not question:
        return {"success": False, "message": "Question is required"}

    try:
        # ── Step 1: Try common query patterns first (instant, no API call) ──
        common_sql = match_common_query(question)
        if common_sql == "__INVALID_DIMENSION__":
            return {
                "success": True,
                "answer_type": "text",
                "message": "I can only answer queries based on: state, district, pincode, bank name, age category, or pensioner type (central/state/other). The term you mentioned is not a supported filter dimension in this dataset. Please try asking about a state, district, bank, or pincode instead.",
                "data": []
            }
        if common_sql:
            rows = execute_sql(common_sql)
            answer_type = "table"
            message = None
            if len(rows) == 0:
                answer_type = "text"
                message = "No results found for your query."
            elif len(rows) == 1 and len(rows[0]) <= 4:
                answer_type = "single_value"
                parts = []
                for key, val in rows[0].items():
                    label = key.replace('_', ' ').title()
                    parts.append(f"{label}: {val:,}" if isinstance(val, (int, float)) else f"{label}: {val}")
                message = " | ".join(parts)

            return {
                "success": True, "answer_type": answer_type, "message": message,
                "sql": common_sql, "data": rows, "total_rows": len(rows), "elapsed": 0.0
            }

        # ── Step 2: Use OpenRouter (free LLM) if API key is configured ──
        if OPENROUTER_API_KEY:
            try:
                start = time.time()
                ai_response = await call_openrouter(question)
                elapsed = round(time.time() - start, 2)

                intent = ai_response.get("intent", "chat")
                reply = ai_response.get("reply", "")
                sql = ai_response.get("sql")

                # Chat intent — return the friendly reply directly
                if intent == "chat" or not sql:
                    return {
                        "success": True, "answer_type": "text",
                        "message": reply or "I'm Alan AI, your DLC Pension assistant. Ask me anything about pensioner data!",
                        "sql": None, "data": None, "total_rows": 0, "elapsed": elapsed
                    }

                # Data intent — execute the SQL
                sql = sql.strip()
                if not sql.upper().startswith("SELECT"):
                    return {
                        "success": True, "answer_type": "text",
                        "message": "I can only answer data retrieval questions.",
                        "sql": sql, "data": None, "total_rows": 0, "elapsed": elapsed
                    }

                rows = execute_sql(sql)
                answer_type = "table"
                message = reply

                if len(rows) == 0:
                    answer_type = "text"
                    message = reply or "No results found for your query."
                elif len(rows) == 1 and len(rows[0]) == 1:
                    answer_type = "single_value"
                    key = list(rows[0].keys())[0]
                    val = rows[0][key]
                    message = f"{key}: {val:,}" if isinstance(val, (int, float)) else f"{key}: {val}"

                return {
                    "success": True, "answer_type": answer_type, "message": message,
                    "sql": sql, "data": rows, "total_rows": len(rows), "elapsed": elapsed
                }

            except Exception as openrouter_err:
                print(f"OpenRouter failed, falling back to local model: {openrouter_err}")
                # Fall through to local model

        # ── Step 3: Fallback to local SQL-R1-7B model ──
        fsql, elapsed_time = fallback_local_model(question)

        if not fsql or not fsql.strip():
            return {
                "success": True, "answer_type": "text",
                "message": "Sorry, I could not generate a query for this question. Please try rephrasing.",
                "sql": None, "data": None
            }

        if not fsql.strip().upper().startswith("SELECT"):
            return {
                "success": True, "answer_type": "text",
                "message": "I can only answer data retrieval questions.",
                "sql": fsql, "data": None
            }

        rows = execute_sql(fsql)
        answer_type = "table"
        message = None

        if len(rows) == 0:
            answer_type = "text"
            message = "No results found for your query."
        elif len(rows) == 1 and len(rows[0]) == 1:
            answer_type = "single_value"
            key = list(rows[0].keys())[0]
            val = rows[0][key]
            message = f"{key}: {val:,}" if isinstance(val, (int, float)) else f"{key}: {val}"

        return {
            "success": True, "answer_type": answer_type, "message": message,
            "sql": fsql, "data": rows, "total_rows": len(rows), "elapsed": elapsed_time
        }

    except mysql.connector.Error as db_err:
        return {
            "success": True, "answer_type": "text",
            "message": "I generated a query but it had an error. Please try rephrasing your question.",
            "sql": None, "data": None
        }
    except Exception as e:
        return {
            "success": False,
            "message": f"Something went wrong: {str(e)}"
        }
