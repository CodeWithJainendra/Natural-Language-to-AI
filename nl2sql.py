from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from sqlparse import format as sqlparse
import time
import os
import torch
from dotenv import load_dotenv
import sqlglot

load_dotenv()

_model_cache = {}

def load_model(model_name: str):
    if model_name in _model_cache:
        return _model_cache[model_name]

    os.environ['HF_TOKEN'] = os.getenv('HF_TOKEN', '')  # set HF_TOKEN in your .env

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
    )
    _model_cache[model_name] = (tokenizer, model)
    return tokenizer, model

def build_prompt(question: str, schema: str, dialect: str) -> list[dict]:
    system_msg = (
        f"You are an expert {dialect} query writer. "
        f"You write correct queries grounded strictly in the schema provided.\n\n"
        f"Rules you must follow:\n"
        f"1. Only use tables and columns that are explicitly defined in the DDL statements below.\n"
        f"2. Never invent, alias-as-new, or infer columns that do not exist in the schema.\n"
        f"3. Only use built-in functions and syntax that are native to {dialect}. "
        f"Do NOT use functions from other database engines. Use the correct {dialect} equivalents instead.\n"
        f"4. Never use SQL reserved keywords (e.g. rank, order, group, select) as bare column names "
        f"— always backtick them if they appear in the schema.\n"
        f"5. IMPORTANT: When using GROUP BY with ORDER BY on an aggregate (COUNT, SUM, etc.), ALWAYS include that aggregate in the SELECT list with a descriptive alias. "
        f"For example, if ordering by COUNT(*), write: SELECT state, COUNT(*) AS total_pensioners FROM ... GROUP BY state ORDER BY total_pensioners DESC. "
        f"Never use an aggregate only in ORDER BY without also selecting it.\n"
        f"6. When the user asks for 'top' or 'best' or 'highest' items, ALWAYS include ALL relevant numeric columns (counts, percentages, ratios) in the SELECT so the result is meaningful. "
        f"A result with only names and no numbers is never acceptable. "
        f"When user says 'top states' or 'top banks' without specifying what metric, default to ranking by DLC completion rate DESC and include: total pensioners (COUNT(*)), DLC done (COUNT(lc_date)), and completion rate ((COUNT(lc_date)*100.0/COUNT(*))).\n"
        f"7. If the question cannot be answered from the schema, say so instead of guessing.\n"
        f"8. Do not add any comments (inline or block) to the query.\n"
        f"9. ALWAYS use UPPER() for case-insensitive string comparisons on state, district, and bank_name columns. For example: WHERE UPPER(state) = UPPER('uttar pradesh').\n"
        f"10. When the user mentions a state or district name, use the correct full name with proper spacing (e.g. 'UTTAR PRADESH' not 'UTTARPRADESH', 'MADHYA PRADESH' not 'MADHYAPRADESH').\n"
        f"11. DLC completion rate = (COUNT(lc_date) / COUNT(*)) * 100. Use this formula whenever the user asks about completion rate or percentage.\n\n"
        f"DDL statements:\n{schema}"
    )
    user_msg = f"Generate a {dialect} query to answer this question: '{question}'"
 
    return [
        {"role": "system", "content": system_msg},
        {"role": "user",   "content": user_msg},
    ]

def tokenize_prompt(tokenizer, model, prompt:list[dict]):
    tokenized_prompt = tokenizer.apply_chat_template(
                    prompt,
                    add_generation_prompt=True,
                    tokenize=True,
                    return_dict=True,
                    return_tensors="pt",
                ).to(model.device)
    return tokenized_prompt

def generate_output(tokenizer, model, tokenized_prompt):
    start = time.time()
    output = model.generate(**tokenized_prompt, max_new_tokens=4000)
    decoded_output = tokenizer.decode(
                        output[0][tokenized_prompt["input_ids"].shape[-1]:],
                        skip_special_tokens=True
                    )
    end = time.time()
    return decoded_output, round(end-start, 2)

def extract_sql(decoded_output):
    sql = decoded_output.split('```sql')[-1].split('```')[0].replace('\n', ' ')
    return sql

def translate_sql(sql, dialect):
    if dialect == 'MySQL':
        tsql = sqlglot.transpile(sql, read='sqlite', write='mysql')[0]
    elif dialect == 'SQLite':
        tsql = sql
    else:
        tsql = sql
    return tsql

def format_sql(sql):
    fsql = sqlparse(sql, reindent=True, keyword_case='upper')
    return fsql