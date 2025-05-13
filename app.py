import json, streamlit as st
from pymongo import MongoClient
from openai import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationSummaryBufferMemory
from langchain.agents import initialize_agent, AgentType, Tool

# ── STREAMLIT CONFIG ──────────────────────────────────────────────────────────
st.set_page_config(page_title="LangChain Resume Chatbot", layout="wide")
st.title("🤖 LangChain Resume Assistant")

# ── OPENAI + LANGCHAIN LLM ────────────────────────────────────────────────────
llm = ChatOpenAI(
    model="gpt-4o",
    api_key=st.secrets["OPENAI_API_KEY"],
    temperature=0
)

# ── MONGO SEARCH FUNCTION (your original logic) ───────────────────────────────
def mongo_search(filters: dict) -> list:
    db = MongoClient(
        host="notify.pesuacademy.com", port=27017,
        username="admin", password=st.secrets["MONGO_PASS"],
        authSource="admin"
    )["resumes_database"]["resumes"]

    country = filters.get("country", "").strip().lower()
    min_exp = int(filters.get("min_experience_years", 0))
    titles  = filters.get("job_titles", [])
    skills  = filters.get("skills", [])
    top_k   = int(filters.get("top_k", 50))

    q = {}
    if country:
        q["country"] = {"$regex": f"^{country}$", "$options": "i"}
    if skills:
        q["$or"] = [{"skills.skillName": {"$in": skills}},
                    {"keywords": {"$in": skills}}]

    filters_and = []
    if titles:
        filters_and.append({"jobExperiences.title": {"$in": titles}})
    if min_exp:
        filters_and.append({
            "$expr": {"$gte": [
                {"$toInt": {"$ifNull": [{"$first": "$jobExperiences.duration"}, "0"]}},
                min_exp]}})
    if filters_and:
        q["$and"] = filters_and

    rows = list(db.find(q, {"_id": 0, "embedding": 0}).limit(top_k))
    db.client.close()
    return rows

# ── LANGCHAIN TOOL (Sous‑Chef) ────────────────────────────────────────────────
def _tool_func(json_str: str) -> str:
    """LangChain Tool wrapper. Input MUST be JSON string of filters."""
    try:
        filters = json.loads(json_str)
    except Exception as e:
        return f"ERROR:: bad JSON ({e})"
    rows = mongo_search(filters)
    # Save rows in session for UI
    st.session_state.last_rows = rows
    st.session_state.last_country = filters.get("country", "unknown")
    return f"RESULT::{len(rows)}"

query_tool = Tool(
    name="query_db",
    func=_tool_func,
    description=(
        "Search resumes. "
        "Input MUST be a JSON string with keys: country, min_experience_years, "
        "max_experience_years, job_titles, skills, top_k"
    )
)

# ── MEMORY & AGENT INITIALISATION ─────────────────────────────────────────────
if "memory" not in st.session_state:
    st.session_state.memory = ConversationSummaryBufferMemory(
        llm=llm,
        max_token_limit=1500,
        memory_key="chat_history"
    )

agent = initialize_agent(
    tools=[query_tool],
    llm=llm,
    agent=AgentType.OPENAI_MULTI_FUNCTIONS,
    memory=st.session_state.memory,
    verbose=False,
    agent_kwargs={
        "system_message":(
            "You are the Manager agent. Converse naturally. "
            "When enough info, call the tool 'query_db' by passing exactly a JSON string "
            "of filters. Otherwise ask clarifying questions. "
            "After receiving RESULT::N from tool, interpret saved rows in Streamlit "
            "(the frontend will display them) and explain to the user."
        )
    }
)

# ── STREAMLIT STATE FOR CHAT & RESULTS ────────────────────────────────────────
if "chat_log" not in st.session_state: st.session_state.chat_log = []
if "last_rows" not in st.session_state: st.session_state.last_rows = []
if "last_country" not in st.session_state: st.session_state.last_country = ""

# ── RENDER CHAT LOG ───────────────────────────────────────────────────────────
for turn in st.session_state.chat_log:
    st.chat_message(turn["role"]).markdown(turn["content"])

# ── RENDER LAST RESULT CARDS (persist) ────────────────────────────────────────
if st.session_state.last_rows:
    st.markdown(f"### 📦 {len(st.session_state.last_rows)} matches for "
                f"{st.session_state.last_country.title()}")
    for r in st.session_state.last_rows:
        with st.container(border=True):
            st.markdown(f"**{r['name']}** — {r.get('country','N/A')}")
            st.markdown(
                f"📧 {r.get('email','N/A')} &nbsp;&nbsp; "
                f"🛠 {', '.join(s.get('skillName','') for s in r.get('skills',[]))}",
                unsafe_allow_html=True
            )

# ── USER INPUT & AGENT RUN ───────────────────────────────────────────────────
user_msg = st.chat_input("Ask something about candidates…")
if user_msg:
    st.chat_message("user").markdown(user_msg)
    st.session_state.chat_log.append({"role": "user", "content": user_msg})

    # LangChain agent handles tool invocation
    agent_reply = agent.run(user_msg)

    # If tool returned RESULT::, agent_reply will still be a string
    st.chat_message("assistant").markdown(agent_reply)
    st.session_state.chat_log.append({"role": "assistant", "content": agent_reply})
