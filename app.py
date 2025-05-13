"""
Resume‑filtering chatbot with memory
LangChain 0.3.25 • OpenAI 1.78.1 • Streamlit 1.34+
"""

import os
import json
from datetime import datetime
from typing import List, Optional, Dict, Any

import streamlit as st
from pymongo import MongoClient
from pymongo.errors import PyMongoError

from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain.agents import create_openai_tools_agent, AgentExecutor
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationBufferMemory

# ── CONFIG ─────────────────────────────────────────────────────────────
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
MONGO_CFG = {
    "host": "notify.pesuacademy.com",
    "port": 27017,
    "username": "admin",
    "password": st.secrets["MONGO_PASS"],
    "authSource": "admin",
}
MODEL_NAME = "gpt-4o"
EVAL_MODEL_NAME = "gpt-4o"
TOP_K_DEFAULT = 50
DB_NAME = "resumes_database"
COLL_NAME = "resumes"

# ── MONGO ──────────────────────────────────────────────────────────────
def get_mongo_client() -> MongoClient:
    return MongoClient(**MONGO_CFG)

# ── NORMALIZATION ──────────────────────────────────────────────────────
COUNTRY_EQUIV = {
    "indonesia": ["indonesia"],
    "vietnam": ["vietnam", "viet nam", "vn", "vietnamese"],
    "united states": ["united states", "usa", "us"],
    "malaysia": ["malaysia"],
    "india": ["india", "ind"],
    "singapore": ["singapore"],
    "philippines": ["philippines", "the philippines"],
    "australia": ["australia"],
    "new zealand": ["new zealand"],
    "germany": ["germany"],
    "saudi arabia": ["saudi arabia", "ksa"],
    "japan": ["japan"],
    "hong kong": ["hong kong", "hong kong sar"],
    "thailand": ["thailand"],
    "united arab emirates": ["united arab emirates", "uae"],
}
SKILL_VARIANTS = {
    "sql": ["sql", "mysql", "microsoft sql server"],
    "javascript": ["javascript", "js", "java script"],
    "c#": ["c#", "c sharp", "csharp"],
    "html": ["html", "hypertext markup language"],
}
TITLE_VARIANTS = {
    "software developer": [
        "software developer",
        "software dev",
        "softwaredeveloper",
        "software engineer",
    ],
    "backend developer": [
        "backend developer",
        "backend dev",
        "back-end developer",
        "server-side developer",
    ],
    "frontend developer": [
        "frontend developer",
        "frontend dev",
        "front-end developer",
    ],
}
def expand(values: List[str], table: Dict[str, List[str]]) -> List[str]:
    out = set()
    for v in values:
        v_low = v.strip().lower()
        out.update(table.get(v_low, []))
        out.add(v)
    return list(out)

# ── LLM‑BASED RESUME SCORER ────────────────────────────────────────────
import openai
_openai_client = openai.OpenAI(api_key=OPENAI_API_KEY)

EVALUATOR_PROMPT = """
You are a resume scoring assistant. Return only the 10 best resumeIds.

JSON format:
{
  "top_resume_ids": [...],
  "completed_at": "ISO"
}
"""

def score_resumes(query: str, resumes: List[Dict[str, Any]]) -> List[str]:
    chat = _openai_client.chat.completions.create(
        model=EVAL_MODEL_NAME,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": EVALUATOR_PROMPT},
            {
                "role": "user",
                "content": f"Query: {query}\n\nResumes: {json.dumps(resumes)}",
            },
        ],
    )
    content = json.loads(chat.choices[0].message.content)
    return content.get("top_resume_ids", [])

# ── TOOL: query_db ─────────────────────────────────────────────────────
@tool
def query_db(
    query: str,
    country: Optional[str] = None,
    min_experience_years: Optional[int] = None,
    max_experience_years: Optional[int] = None,
    job_titles: Optional[List[str]] = None,
    skills: Optional[List[str]] = None,
    top_k: int = TOP_K_DEFAULT,
) -> Dict[str, Any]:
    """Filter MongoDB resumes and return top 10 matches."""
    try:
        mongo_q: Dict[str, Any] = {}

        if country:
            variants = COUNTRY_EQUIV.get(country.strip().lower(), [country])
            mongo_q["country"] = {"$in": variants}

        if skills:
            expanded_skills = expand(skills, SKILL_VARIANTS)
            mongo_q["$or"] = [
                {"skills.skillName": {"$in": expanded_skills}},
                {"keywords": {"$in": expanded_skills}},
            ]

        and_clauses = []
        if job_titles:
            expanded_titles = expand(job_titles, TITLE_VARIANTS)
            and_clauses.append({"jobExperiences.title": {"$in": expanded_titles}})

        if isinstance(min_experience_years, int) and min_experience_years > 0:
            and_clauses.append(
                {
                    "$expr": {
                        "$gte": [
                            {
                                "$toInt": {
                                    "$ifNull": [{"$first": "$jobExperiences.duration"}, "0"]
                                }
                            },
                            min_experience_years,
                        ]
                    }
                }
            )
        if and_clauses:
            mongo_q["$and"] = and_clauses

        with get_mongo_client() as client:
            coll = client[DB_NAME][COLL_NAME]
            candidates = list(
                coll.find(mongo_q, {"_id": 0, "embedding": 0}).limit(top_k)
            )

        best_ids = score_resumes(query, candidates)
        best_resumes = [r for r in candidates if r["resumeId"] in best_ids]

        return {
            "message": f"{len(best_resumes)} resumes after scoring.",
            "results_count": len(best_resumes),
            "results": best_resumes,
            "completed_at": datetime.utcnow().isoformat(),
        }

    except PyMongoError as err:
        return {"error": f"DB error: {str(err)}"}
    except Exception as exc:
        return {"error": str(exc)}

# ── AGENT + MEMORY ─────────────────────────────────────────────────────
llm = ChatOpenAI(model=MODEL_NAME, api_key=OPENAI_API_KEY, temperature=0)

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful HR assistant. Use the `query_db` tool whenever the "
            "user asks for candidates or filtering. Otherwise, answer normally.",
        ),
        MessagesPlaceholder(variable_name="chat_history"),       # <-- memory
        ("user", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),    # <-- tool log
    ]
)

# We keep one ConversationBufferMemory per Streamlit session
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
    )

if "agent_executor" not in st.session_state:
    agent = create_openai_tools_agent(llm, [query_db], prompt)
    st.session_state.agent_executor = AgentExecutor(
        agent=agent,
        tools=[query_db],
        memory=st.session_state.memory,
        verbose=True,
    )

# ── STREAMLIT UI ───────────────────────────────────────────────────────
st.title("🧠 Resume Filtering Chatbot")

user_input = st.chat_input("Ask me to find resumes...")
if user_input:
    with st.spinner("Thinking..."):
        result = st.session_state.agent_executor.invoke({"input": user_input})
        # tiny guard: executor returns {'output': '...'} when llm answers
        bot_reply = result.get("output", result)
        st.session_state.memory.save_context({"input": user_input}, {"output": bot_reply})

# render chat history (newest last)
for msg in st.session_state.memory.chat_memory.messages:
    if msg.type == "human":
        st.chat_message("user").write(msg.content)
    else:
        st.chat_message("assistant").write(msg.content)
