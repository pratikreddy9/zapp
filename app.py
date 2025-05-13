import json, streamlit as st
from typing import List, Optional
from pymongo import MongoClient
from pydantic import BaseModel, Field

from langchain_openai import ChatOpenAI
from langchain.memory import ConversationSummaryBufferMemory
from langchain_core.tools import Tool                   # new unified core
from langchain.agents import (                          # v0.4.x agent API
    OpenAIFunctionsAgent,
    AgentExecutor
)

# ── PAGE CONFIG ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="Latest‑LC Resume Chatbot", layout="wide")
st.title("⚡️ LangChain‑0.4 Resume Assistant")

# ── DB SEARCH LOGIC (unchanged) ──────────────────────────────────────────────
def mongo_search(filters: dict) -> list:
    db = MongoClient(
        host="notify.pesuacademy.com", port=27017,
        username="admin", password=st.secrets["MONGO_PASS"],
        authSource="admin"
    )["resumes_database"]["resumes"]

    ctry = filters.get("country", "").strip().lower()
    skills = filters.get("skills", [])
    titles = filters.get("job_titles", [])
    min_exp = int(filters.get("min_experience_years", 0))
    top_k  = int(filters.get("top_k", 50))

    q = {}
    if ctry:
        q["country"] = {"$regex": f"^{ctry}$", "$options": "i"}
    if skills:
        q["$or"] = [{"skills.skillName": {"$in": skills}},
                    {"keywords": {"$in": skills}}]

    f = []
    if titles:
        f.append({"jobExperiences.title": {"$in": titles}})
    if min_exp:
        f.append({
            "$expr": {"$gte": [
                {"$toInt": {"$ifNull": [{"$first": "$jobExperiences.duration"}, "0"]}},
                min_exp]}})
    if f: q["$and"] = f

    rows = list(db.find(q, {"_id": 0, "embedding": 0}).limit(top_k))
    db.client.close()
    return rows

# ── Pydantic schema for the tool (latest LC pattern) ─────────────────────────
class QueryFilters(BaseModel):
    country: str = Field(..., description="Country name (normalized)")
    min_experience_years: Optional[int] = Field(0, description="Minimum years")
    max_experience_years: Optional[int] = Field(None, description="Maximum years")
    job_titles: List[str] = Field(..., description="List of job titles")
    skills: List[str] = Field(..., description="List of skills")
    top_k: Optional[int] = Field(50, description="Number of results")

def query_db_tool(json_str: str) -> str:
    """
    Expects JSON string that matches QueryFilters.
    Saves rows in session for UI and returns RESULT::N.
    """
    try:
        filters = QueryFilters.model_validate_json(json_str).model_dump()
    except Exception as err:
        return f"ERROR:: {err}"
    rows = mongo_search(filters)
    st.session_state.last_rows  = rows
    st.session_state.last_query = filters
    return f"RESULT::{len(rows)}"

tool = Tool.from_function(
    name="query_db",
    description=(
        "Search the resume MongoDB. "
        "Input **must** be a JSON string conforming to QueryFilters schema."
    ),
    func=query_db_tool
)

# ── MASTER PROMPT (your big schema prompt) ───────────────────────────────────
MASTER_PROMPT = """
You are a powerful resume filtering assistant. Your job is to convert natural language user queries into a **structured MongoDB query**, fetch matching resumes, and respond with a JSON output that always includes:

{
    "message": "... natural language message summarizing the search ...",
    "query_parameters": {
        "country": "...",
        "min_experience_years": ...,
        "max_experience_years": ...,
        "job_titles": [...],
        "skills": [...],
        "top_k": ...
    },
    "results_count": ...,
    "results": [ ... array of matching resumes ... ],
    "completed_at": "ISO timestamp"
}

Resume collection structure:

- resumeId (str, always present)
- name (str)
- email (str)
- contactNo (str)
- address (nullable str)
- country (nullable str) → we normalize using .strip().lower()

- educationalQualifications (list of dicts)
    - degree (nullable str)
    - field (nullable str)
    - institution (nullable str)
    - graduationYear (nullable int)

- jobExperiences (list of dicts)
    - title (nullable str)
    - duration (nullable str, can be numeric or text)

- keywords (list of str)
- skills (list of dicts)
    - skillName (str)

Country Variants Knowledge:

The following country variants are normalized via .strip().lower() and must be considered equivalent:

Indonesia: ["Indonesia"]
Vietnam: ["Vietnam", "Viet Nam", "Vn", "Vietnamese"]
United States: ["United States", "Usa", "Us"]
Malaysia: ["Malaysia"]
India: ["India", "Ind"]
Singapore: ["Singapore"]
Philippines: ["Philippines", "The Philippines"]
Australia: ["Australia"]
New Zealand: ["New Zealand"]
Germany: ["Germany"]
Saudi Arabia: ["Saudi Arabia", "Ksa"]
Japan: ["Japan"]
Hong Kong: ["Hong Kong", "Hong Kong Sar"]
Thailand: ["Thailand"]
United Arab Emirates: ["United Arab Emirates", "Uae"]

Filtering Rules:

- Country: match via normalized .strip().lower() against country.
- Experience: check jobExperiences[].duration. If numeric string, convert to int; if malformed/null, skip.
- Skills: MUST check in BOTH:
    1️⃣ skills[].skillName (list of dicts)
    2️⃣ keywords[] (list of strings)
- Keywords: match against keywords list.

✅ For skills: when filtering resumes, **always match the target skill against BOTH the `skills[].skillName` and the `keywords` list.** This ensures no relevant candidates are missed.

ALWAYS return a complete JSON response in the above format, even during intermediate steps if you need to refine your query.
✅ Skill & Title Normalization Rules:

To improve search accuracy, when building the MongoDB query:

- **Skills:**
    - Always expand skill names to include common variants, synonyms, and different casings.
    - Examples:
        - "SQL" → ["SQL", "sql", "mysql", "microsoft sql server"]
        - "JavaScript" → ["JavaScript", "javascript", "js", "java script"]
        - "C#" → ["C#", "c sharp", "csharp"]
        - "HTML" → ["HTML", "html", "hypertext markup language"]

- **Job Titles:**
    - Always expand job titles to include common abbreviations and different spacings.
    - Examples:
        - "Software Developer" → ["Software Developer", "software dev", "softwaredeveloper", "software engineer"]
        - "Backend Developer" → ["Backend Developer", "backend dev", "back-end developer", "server-side developer"]
        - "Frontend Developer" → ["Frontend Developer", "frontend dev", "front-end developer"]

- **Case Insensitivity:**
    - All matches are case-insensitive (MongoDB uses `$regex` with `"i"` option).

👉 **IMPORTANT:**
- Expand these fields **directly inside the `query_parameters` JSON** (the `skills` and `job_titles` arrays).
- This ensures the backend can directly use these expanded arrays without needing additional normalization.

✅ Always include **all relevant variants** to avoid missing good matches.

"""

EVALUATOR_PROMPT = """
You are a resume scoring assistant. You receive:

1️⃣ A natural language query (the user's request),
2️⃣ A list of resumes that have already been pre-filtered by country, skills, experience, and job title.

🎯 Your task is to:

- Review the resumes based on the query.
- Select and return ONLY the top 10 best-matching `resumeId`s.

✅ Output format (JSON):

{
    "top_resume_ids": [ ... up to 10 resumeId strings ... ],
    "completed_at": "ISO timestamp"
}

👉 NOTE:
- ONLY return the `resumeId` values in the array.
- Do NOT return full resumes or any extra text.
"""

# ── LLM + MEMORY ─────────────────────────────────────────────────────────────
llm = ChatOpenAI(model="gpt-4o", temperature=0)
memory = ConversationSummaryBufferMemory(
    llm=llm, max_token_limit=1500, memory_key="chat_history"
)

# ── Agent construction (v0.4.x pattern) ──────────────────────────────────────
functions_agent = OpenAIFunctionsAgent.from_llm_and_tools(
    llm=llm,
    tools=[tool],
    system_message=MASTER_PROMPT
)

agent = AgentExecutor(
    agent=functions_agent,
    tools=[tool],
    memory=memory,
    verbose=False
)

# ── Streamlit state ──────────────────────────────────────────────────────────
if "chat" not in st.session_state:        st.session_state.chat = []
if "last_rows" not in st.session_state:   st.session_state.last_rows = []
if "last_query" not in st.session_state:  st.session_state.last_query = {}

# ── Display chat history ─────────────────────────────────────────────────────
for m in st.session_state.chat:
    st.chat_message(m["role"]).markdown(m["content"])

# ── Display last result cards ────────────────────────────────────────────────
if st.session_state.last_rows:
    ctry = st.session_state.last_query.get("country", "?").title()
    st.markdown(f"### {len(st.session_state.last_rows)} matches for {ctry}")
    for r in st.session_state.last_rows:
        with st.container(border=True):
            st.markdown(f"**{r['name']}** — {r.get('country','N/A')}")
            st.markdown(
                f"📧 {r.get('email','N/A')} &nbsp;&nbsp; "
                f"🛠 {', '.join(s.get('skillName','') for s in r.get('skills',[]))}",
                unsafe_allow_html=True
            )

# ── User input and agent run ────────────────────────────────────────────────
user_text = st.chat_input("Ask about candidates…")
if user_text:
    st.chat_message("user").markdown(user_text)
    st.session_state.chat.append({"role": "user", "content": user_text})

    assistant_reply = agent.run(user_text)

    st.chat_message("assistant").markdown(assistant_reply)
    st.session_state.chat.append({"role": "assistant", "content": assistant_reply})
