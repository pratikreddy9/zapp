import json, streamlit as st
from pymongo import MongoClient
from openai import OpenAI

# ── CONFIG ────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Resume Chat", layout="wide")
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

MASTER_PROMPT = """
You are a resume‑search assistant. Ask follow‑up questions when needed; otherwise
call the tool `search_resumes` with the JSON filters described below.

Resume fields: resumeId, name, email, contactNo, address, country (lower‑case),
educationalQualifications[], jobExperiences[], keywords[], skills[].skillName.

Matching rules: country case‑insensitive; duration numeric; skills must match
BOTH skills[].skillName and keywords[].  Expand common variants
(SQL→mysql, JS, etc.). Always respond in valid JSON.
"""

# ── MONGO TOOL ────────────────────────────────────────────────────────────────
def get_mongo():
    return MongoClient(
        host="notify.pesuacademy.com",
        port=27017,
        username="admin",
        password=st.secrets["MONGO_PASS"],
        authSource="admin"
    )

def search_resumes(params: dict) -> list:
    db = get_mongo()["resumes_database"]["resumes"]
    top_k = int(params.get("top_k", 50))
    country = params.get("country", "").strip().lower()
    skills  = params.get("skills", [])
    titles  = params.get("job_titles", [])
    min_exp = int(params.get("min_experience_years", 0))

    q = {}
    if country:
        q["country"] = {"$regex": f"^{country}$", "$options": "i"}
    if skills:
        q["$or"] = [{"skills.skillName": {"$in": skills}},
                    {"keywords": {"$in": skills}}]

    f = []
    if titles:
        f.append({"jobExperiences.title": {"$in": titles}})
    if min_exp:
        f.append({"$expr": {"$gte": [
            {"$toInt": {"$ifNull": [{"$first": "$jobExperiences.duration"}, "0"]}},
            min_exp]}})
    if f:
        q["$and"] = f

    res = list(db.find(q, {"_id":0,"embedding":0}).limit(top_k))
    get_mongo().close()
    return res

# ── STREAMLIT SESSION ─────────────────────────────────────────────────────────
if "history" not in st.session_state:      # chat log
    st.session_state.history = []
if "busy" not in st.session_state:         # prevents double calls
    st.session_state.busy = False

# ── RENDER CHAT HISTORY ───────────────────────────────────────────────────────
for m in st.session_state.history:
    st.chat_message(m["role"]).markdown(m["content"] if isinstance(m["content"], str) else "")

# ── INPUT BOX ────────────────────────────────────────────────────────────────
prompt = st.chat_input("Ask something about resumes…")
if prompt and not st.session_state.busy:
    st.session_state.busy = True
    st.session_state.history.append({"role":"user","content":prompt})
    st.chat_message("user").markdown(prompt)

    # Build conversation for GPT
    messages = [{"role":"system","content":MASTER_PROMPT}]
    for m in st.session_state.history:
        messages.append({"role":m["role"],"content":m["content"] if isinstance(m["content"],str) else ""})

    tool_schema = [{
        "type":"function",
        "function":{
            "name":"search_resumes",
            "description":"Search resumes with structured filters.",
            "parameters":{
                "type":"object",
                "properties":{
                    "country":{"type":"string"},
                    "min_experience_years":{"type":"integer"},
                    "max_experience_years":{"type":"integer"},
                    "job_titles":{"type":"array","items":{"type":"string"}},
                    "skills":{"type":"array","items":{"type":"string"}},
                    "top_k":{"type":"integer"}
                },
                "required":["country","skills","job_titles"]
            }
        }
    }]

    # ── GPT CALL ────────────────────────────────────────────────────────────
    rsp = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        tools=tool_schema,
        tool_choice="auto",
        response_format={"type":"json_object"}
    ).choices[0].message

    # assistant speaks (even if also calling a tool)
    if rsp.content:
        st.chat_message("assistant").markdown(rsp.content)
        st.session_state.history.append({"role":"assistant","content":rsp.content})

    # tool call?
    if rsp.tool_calls:
        args = json.loads(rsp.tool_calls[0].function.arguments)
        rows = search_resumes(args)

        # show cards
        if rows:
            card_container = st.container()
            with card_container:
                st.markdown("### 🔍 Matches")
                for r in rows:
                    with st.container(border=True):
                        st.markdown(f"**{r.get('name','Unnamed')}**")
                        st.markdown(f"📧 {r.get('email','N/A')} &nbsp;&nbsp; 🌍 {r.get('country','N/A')}")
                        st.markdown(f"🛠️ {', '.join(s.get('skillName','') for s in r.get('skills',[]))}")

        # store last results if you want
        st.session_state.history.append({"role":"assistant","content":f"Found {len(rows)} resumes."})

    st.session_state.busy = False
