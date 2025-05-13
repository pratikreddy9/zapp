import json, streamlit as st
from pymongo import MongoClient
from openai import OpenAI

# ── PAGE CONFIG ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="Resume Chatbot", layout="wide")
openai_client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# ── SYSTEM PROMPT (condensed) ─────────────────────────────────────────────────
MASTER_PROMPT = """
You are a conversational resume‑search assistant.
• If info is missing, ask follow‑ups.
• Else call the tool `search_resumes` with JSON args:
  {country,min_experience_years,max_experience_years,job_titles,skills,top_k}.
Rules: country case‑insensitive; skills must match BOTH skills.skillName & keywords.
Expand common variants (SQL→mysql, JS→javascript, Software Developer→software engineer).
Always output valid JSON when calling the tool; otherwise reply naturally.
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
    country = params.get("country", "").strip().lower()
    min_exp = int(params.get("min_experience_years", 0))
    titles  = params.get("job_titles", [])
    skills  = params.get("skills", [])
    top_k   = int(params.get("top_k", 50))

    q = {}
    if country:
        q["country"] = {"$regex": f"^{country}$", "$options": "i"}
    if skills:
        q["$or"] = [{"skills.skillName":{"$in":skills}}, {"keywords":{"$in":skills}}]

    f = []
    if titles:
        f.append({"jobExperiences.title":{"$in":titles}})
    if min_exp:
        f.append({"$expr":{"$gte":[
            {"$toInt":{"$ifNull":[{"$first":"$jobExperiences.duration"},"0"]}},
            min_exp]}})
    if f: q["$and"] = f

    rows = list(db.find(q, {"_id":0,"embedding":0}).limit(top_k))
    get_mongo().close()
    return rows

# ── SESSION STATE ─────────────────────────────────────────────────────────────
if "chat"     not in st.session_state: st.session_state.chat = []          # all turns
if "searches" not in st.session_state: st.session_state.searches = []      # [{country,rows}]
if "busy"     not in st.session_state: st.session_state.busy = False

# ── HELPERS ───────────────────────────────────────────────────────────────────
def save_search(country: str, rows: list):
    st.session_state.searches.append({"country": country.lower(), "rows": rows})

def context_summaries(max_items=3) -> str:
    summaries = []
    for item in st.session_state.searches[-max_items:]:
        names = ", ".join(r["name"] for r in item["rows"][:5])
        summaries.append(f"{item['country'].title()}: {len(item['rows'])} matches (e.g. {names})")
    return "\n".join(summaries)

def render_cards(rows: list):
    for r in rows:
        with st.container(border=True):
            st.markdown(f"**{r.get('name','Unnamed')}** — {r.get('country','N/A')}")
            st.markdown(
                f"📧 {r.get('email','N/A')} &nbsp;&nbsp; "
                f"📱 {r.get('contactNo','N/A')}<br>"
                f"🛠 {', '.join(s.get('skillName','') for s in r.get('skills',[]))}",
                unsafe_allow_html=True
            )

# ── RENDER CHAT HISTORY ──────────────────────────────────────────────────────
st.title("🤖 Resume‑Search Assistant")
for turn in st.session_state.chat:
    st.chat_message(turn["role"]).markdown(turn["content"])

# ── RENDER ALL SEARCHES (persist) ────────────────────────────────────────────
if st.session_state.searches:
    st.markdown("### 📦 Searches so far")
    for idx, item in enumerate(st.session_state.searches, 1):
        st.markdown(f"**Result #{idx} – {item['country'].title()} ({len(item['rows'])} matches)**")
        render_cards(item["rows"])

# ── USER INPUT ───────────────────────────────────────────────────────────────
prompt = st.chat_input("Ask something about candidates…")
if prompt and not st.session_state.busy:
    st.session_state.busy = True
    st.session_state.chat.append({"role":"user","content":prompt})
    st.chat_message("user").markdown(prompt)

    # Build GPT context: system + memory summaries + last 6 turns
    messages = [{"role":"system","content":MASTER_PROMPT}]
    mem = context_summaries()
    if mem:
        messages.append({"role":"assistant","content":"Memory:\n"+mem})
    messages += [{"role":m["role"],"content":m["content"]} for m in st.session_state.chat[-6:]]

    tool_schema = [{
        "type":"function",
        "function":{
            "name":"search_resumes",
            "description":"Search resumes with filters",
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
    rsp = openai_client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        tools=tool_schema,
        tool_choice="auto",
        response_format={"type":"json_object"}
    ).choices[0].message

    # Assistant reply text
    if rsp.content:
        st.chat_message("assistant").markdown(rsp.content)
        st.session_state.chat.append({"role":"assistant","content":rsp.content})

    # If GPT called the tool
    if rsp.tool_calls:
        args = json.loads(rsp.tool_calls[0].function.arguments)
        rows = search_resumes(args)
        save_search(args.get("country","unknown"), rows)

        confirm = f"Here are the top {len(rows)} matches for {args.get('country','?').title()}."
        st.chat_message("assistant").markdown(confirm)
        st.session_state.chat.append({"role":"assistant","content":confirm})
        render_cards(rows)

    st.session_state.busy = False
