import json, streamlit as st
from datetime import datetime
from pymongo import MongoClient

# LangChain 0.3.x packages
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationSummaryBufferMemory
from langchain.tools import Tool
from langchain.agents import create_openai_functions_agent, AgentExecutor

# ── STREAMLIT CONFIG ──────────────────────────────────────────────────────────
st.set_page_config(page_title="LangChain Resume Chatbot", layout="wide")
st.title("💼 LangChain Resume Assistant")

# ── LLM (GPT‑4o) ──────────────────────────────────────────────────────────────
llm = ChatOpenAI(
    model="gpt-4o",
    api_key=st.secrets["OPENAI_API_KEY"],
    temperature=0
)

# ── YOUR MASTER & EVALUATOR PROMPTS ───────────────────────────────────────────
MASTER_PROMPT = """<-- paste the big prompt you provided here, unchanged -->"""
EVALUATOR_PROMPT = """<-- paste evaluator prompt if you later need it -->"""

# ── MONGO SEARCH FUNCTION (exact logic) ───────────────────────────────────────
def mongo_search(filters: dict) -> list:
    db = MongoClient(
        host="notify.pesuacademy.com", port=27017,
        username="admin", password=st.secrets["MONGO_PASS"],
        authSource="admin"
    )["resumes_database"]["resumes"]

    country = filters.get("country", "").strip().lower()
    titles  = filters.get("job_titles", [])
    skills  = filters.get("skills", [])
    min_exp = int(filters.get("min_experience_years", 0))
    top_k   = int(filters.get("top_k", 50))

    q = {}
    if country:
        q["country"] = {"$regex": f"^{country}$", "$options": "i"}
    if skills:
        q["$or"] = [{"skills.skillName":{"$in":skills}},
                    {"keywords":{"$in":skills}}]

    f = []
    if titles:
        f.append({"jobExperiences.title":{"$in":titles}})
    if min_exp:
        f.append({
            "$expr":{"$gte":[
                {"$toInt":{"$ifNull":[{"$first":"$jobExperiences.duration"},"0"]}},
                min_exp]}})
    if f: q["$and"] = f

    rows = list(db.find(q, {"_id":0,"embedding":0}).limit(top_k))
    db.client.close()
    return rows

# ── LANGCHAIN TOOL (expects JSON string) ──────────────────────────────────────
def query_db_tool(json_filters: str) -> str:
    """
    LangChain Tool wrapper. Input MUST be a JSON string representing filters.
    Returns a short confirmation; rows are saved in Streamlit state for UI.
    """
    try:
        filters = json.loads(json_filters)
    except Exception as e:
        return f"ERROR:: bad JSON ({e})"

    rows = mongo_search(filters)
    st.session_state.last_rows  = rows
    st.session_state.last_query = filters
    return f"RESULT::{len(rows)}"

tool = Tool(
    name="query_db",
    func=query_db_tool,
    description=(
        "Search resumes in MongoDB. "
        "Input: a JSON string with keys "
        "country, min_experience_years, max_experience_years, "
        "job_titles (list), skills (list), top_k."
    )
)

# ── MEMORY ────────────────────────────────────────────────────────────────────
if "memory" not in st.session_state:
    st.session_state.memory = ConversationSummaryBufferMemory(
        llm=llm, max_token_limit=1500, memory_key="chat_history"
    )

# ── AGENT INITIALISATION (OpenAI functions agent) ────────────────────────────
agent_chain = create_openai_functions_agent(
    llm=llm,
    tools=[tool],
    memory=st.session_state.memory,
    system_message=MASTER_PROMPT
)
agent = AgentExecutor(agent=agent_chain, tools=[tool], memory=st.session_state.memory)

# ── CHAT STORAGE ─────────────────────────────────────────────────────────────
if "chat" not in st.session_state: st.session_state.chat = []
if "last_rows" not in st.session_state: st.session_state.last_rows = []
if "last_query" not in st.session_state: st.session_state.last_query = {}

# ── RENDER CHAT HISTORY ──────────────────────────────────────────────────────
for msg in st.session_state.chat:
    st.chat_message(msg["role"]).markdown(msg["content"])

# ── RENDER LAST RESULT CARDS (persist) ───────────────────────────────────────
if st.session_state.last_rows:
    ctry = st.session_state.last_query.get("country","?").title()
    st.markdown(f"### 🗂️ {len(st.session_state.last_rows)} matches for {ctry}")
    for r in st.session_state.last_rows:
        with st.container(border=True):
            st.markdown(f"**{r['name']}** — {r.get('country','N/A')}")
            st.markdown(
                f"📧 {r.get('email','N/A')} &nbsp;&nbsp; "
                f"🛠 {', '.join(s.get('skillName','') for s in r.get('skills',[]))}",
                unsafe_allow_html=True
            )

# ── USER INPUT ───────────────────────────────────────────────────────────────
user_text = st.chat_input("Ask me about candidates…")
if user_text:
    st.chat_message("user").markdown(user_text)
    st.session_state.chat.append({"role": "user", "content": user_text})

    # Agent run (auto handles tool)
    reply = agent.run(user_text)

    st.chat_message("assistant").markdown(reply)
    st.session_state.chat.append({"role": "assistant", "content": reply})
