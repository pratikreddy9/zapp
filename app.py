import json, streamlit as st
from pymongo import MongoClient
from openai import OpenAI

# ── PAGE CONFIG ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="Next‑Gen Resume Chatbot", layout="wide")
oa = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# ── SYSTEM PROMPTS ────────────────────────────────────────────────────────────
MANAGER_SYS = """
You are the Manager agent. Keep a friendly, analytical tone.
If you need data, output ONLY this JSON:

{ "action":"query_db", "filters":{ country,min_experience_years,max_experience_years,
  job_titles,skills,top_k } }

No extra keys or comments.
If data (rows) is provided, analyse and answer naturally.
The word JSON appears here so OpenAI accepts json_object.
"""
RESPONDER_SYS = "You are the Manager agent. Analyse the given rows and answer clearly."

# ── TOOLS SCHEMA (visible to Manager) ─────────────────────────────────────────
TOOLS = [{
    "type": "function",
    "function": {
        "name": "query_db",
        "description": "Search resumes via structured filters.",
        "parameters": {
            "type": "object",
            "properties": {
                "country": {"type": "string"},
                "min_experience_years": {"type": "integer"},
                "max_experience_years": {"type": "integer"},
                "job_titles": {"type": "array", "items": {"type": "string"}},
                "skills": {"type": "array", "items": {"type": "string"}},
                "top_k": {"type": "integer"}
            },
            "required": ["country", "skills", "job_titles"]
        }
    }
}]

# ── MONGO FUNCTION (Sous‑Chef) ────────────────────────────────────────────────
def query_db(filters: dict) -> list:
    coll = MongoClient(
        host="notify.pesuacademy.com", port=27017,
        username="admin", password=st.secrets["MONGO_PASS"],
        authSource="admin"
    )["resumes_database"]["resumes"]

    q = {}
    c = filters.get("country", "").strip().lower()
    if c:
        q["country"] = {"$regex": f"^{c}$", "$options": "i"}
    skills = filters.get("skills", [])
    if skills:
        q["$or"] = [{"skills.skillName": {"$in": skills}}, {"keywords": {"$in": skills}}]

    f = []
    titles = filters.get("job_titles", [])
    if titles:
        f.append({"jobExperiences.title": {"$in": titles}})
    min_exp = int(filters.get("min_experience_years", 0))
    if min_exp:
        f.append({"$expr": {"$gte": [
            {"$toInt": {"$ifNull": [{"$first": "$jobExperiences.duration"}, "0"]}},
            min_exp]}})
    if f:
        q["$and"] = f

    top_k = int(filters.get("top_k", 50))
    rows = list(coll.find(q, {"_id": 0, "embedding": 0}).limit(top_k))
    coll.database.client.close()
    return rows

# ── MEMORY HELPERS ────────────────────────────────────────────────────────────
def summarise(rows, max_show=5):
    return ", ".join(r["name"] for r in rows[:max_show]) if rows else "0 rows"

def add_summary(country, rows):
    st.session_state.memory.append({
        "country": country.lower(),
        "count": len(rows),
        "summary": summarise(rows)
    })

def memory_context(max_items=3):
    ctx = []
    for m in st.session_state.memory[-max_items:]:
        ctx.append(f"{m['country'].title()}: {m['count']} matches (e.g. {m['summary']})")
    return "\n".join(ctx)

# ── SESSION STATE INIT ────────────────────────────────────────────────────────
if "chat"   not in st.session_state: st.session_state.chat = []   # conversation turns
if "memory" not in st.session_state: st.session_state.memory = [] # past search summaries
if "busy"   not in st.session_state: st.session_state.busy = False

# ── RENDER ALL CHAT TURNS ─────────────────────────────────────────────────────
st.title("🤖  Two‑Agent Resume Assistant")
for t in st.session_state.chat:
    st.chat_message(t["role"]).markdown(t["content"])

# ── USER INPUT ────────────────────────────────────────────────────────────────
prompt = st.chat_input("Ask your question…")
if prompt and not st.session_state.busy:
    st.session_state.busy = True
    st.session_state.chat.append({"role": "user", "content": prompt})
    st.chat_message("user").markdown(prompt)

    # Build context for Manager
    ctx = [{"role": "system", "content": MANAGER_SYS}]
    mem = memory_context()
    if mem:
        ctx.append({"role": "assistant", "content": "Previous summaries:\n" + mem})
    ctx += [{"role": m["role"], "content": m["content"]}
            for m in st.session_state.chat[-6:]]

    # ── Manager CALL ─────────────────────────────────────────────────────────
    mgr = oa.chat.completions.create(
        model="gpt-4o",
        messages=ctx,
        tools=TOOLS,
        tool_choice="auto",
        response_format={"type": "json_object"}
    ).choices[0].message

    # If Manager just chats
    if not mgr.tool_calls:
        st.chat_message("assistant").markdown(mgr.content)
        st.session_state.chat.append({"role": "assistant", "content": mgr.content})
        st.session_state.busy = False
        st.stop()

    # ── Manager issued query_db JSON ─────────────────────────────────────────
    filters = json.loads(mgr.tool_calls[0].function.arguments)
    rows = query_db(filters)
    add_summary(filters.get("country", "unknown"), rows)

    # call Manager again as responder with rows
    responder_msgs = [
        {"role": "system", "content": RESPONDER_SYS},
        {"role": "user", "content": json.dumps({
            "question": prompt,
            "rows": rows[:20]             # truncate for token cost
        })}
    ]
    resp = oa.chat.completions.create(
        model="gpt-4o",
        messages=responder_msgs
    ).choices[0].message.content

    # show answer
    st.chat_message("assistant").markdown(resp)
    st.session_state.chat.append({"role": "assistant", "content": resp})

    # show cards
    if rows:
        st.markdown(f"### 📦 {len(rows)} matches for {filters.get('country','?').title()}")
        for r in rows:
            with st.container(border=True):
                st.markdown(f"**{r['name']}** — {r.get('country','N/A')}")
                st.markdown(
                    f"📧 {r.get('email','N/A')} &nbsp;&nbsp; "
                    f"🛠 {', '.join(s.get('skillName','') for s in r.get('skills',[]))}",
                    unsafe_allow_html=True
                )

    st.session_state.busy = False
