import json, streamlit as st
from pymongo import MongoClient
from openai import OpenAI

# ── STREAMLIT CONFIG ──────────────────────────────────────────────────────────
st.set_page_config(page_title="Resume Chatbot", layout="wide")
openai_client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# ── SYSTEM PROMPT (condensed but complete) ────────────────────────────────────
MASTER_PROMPT = """
You are a conversational resume‑search assistant.  
• If more info is needed, ask clarifying questions.  
• Otherwise call the tool `search_resumes` with:
  {country,min_experience_years,max_experience_years,job_titles,skills,top_k}.  
Rules: country case‑insensitive; skills must match BOTH skills.skillName & keywords.  
Expand variants (SQL→mysql, JS→javascript, Software Developer→software engineer…).  
Respond in valid JSON when calling the tool; otherwise speak naturally.
"""

# ── MONGO SEARCH TOOL ─────────────────────────────────────────────────────────
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
    country = params.get("country","").strip().lower()
    min_exp = int(params.get("min_experience_years", 0))
    skills  = params.get("skills",[])
    titles  = params.get("job_titles",[])

    query = {}
    if country:
        query["country"] = {"$regex": f"^{country}$", "$options": "i"}
    if skills:
        query["$or"] = [{"skills.skillName":{"$in":skills}}, {"keywords":{"$in":skills}}]

    f=[]
    if titles:
        f.append({"jobExperiences.title":{"$in":titles}})
    if min_exp:
        f.append({"$expr":{"$gte":[
            {"$toInt":{"$ifNull":[{"$first":"$jobExperiences.duration"},"0"]}},
            min_exp]}})
    if f: query["$and"]=f

    rows=list(db.find(query,{"_id":0,"embedding":0}).limit(top_k))
    get_mongo().close()
    return rows

# ── SESSION STATE INIT ────────────────────────────────────────────────────────
if "chat"     not in st.session_state: st.session_state.chat=[]
if "searches" not in st.session_state: st.session_state.searches=[]  # list(dict)

if "busy" not in st.session_state:     st.session_state.busy=False

# ── RENDER ENTIRE CHAT HISTORY ────────────────────────────────────────────────
for turn in st.session_state.chat:
    st.chat_message(turn["role"]).markdown(turn["content"])

# ── RENDER ALL SAVED SEARCH RESULT CARDS ──────────────────────────────────────
for idx,search in enumerate(st.session_state.searches,1):
    st.markdown(f"### 🔎 Result #{idx}: {search['summary']}")
    for r in search["rows"]:
        with st.container(border=True):
            st.markdown(f"**{r.get('name','Unnamed')}** — {r.get('country','N/A')}")
            st.markdown(
                f"📧 {r.get('email','N/A')}   📱 {r.get('contactNo','N/A')}<br>"
                f"🛠 {', '.join(s.get('skillName','') for s in r.get('skills',[]))}",
                unsafe_allow_html=True
            )

# ── USER INPUT ───────────────────────────────────────────────────────────────
prompt = st.chat_input("Ask the assistant…")
if prompt and not st.session_state.busy:
    # add user turn
    st.session_state.chat.append({"role":"user","content":prompt})
    st.chat_message("user").markdown(prompt)
    st.session_state.busy=True

    # build context: system + last 3 turns
    context=[{"role":"system","content":MASTER_PROMPT}]
    context += [{"role":m["role"],"content":m["content"]}
                for m in st.session_state.chat[-6:]]  # last 3 user+assistant pairs

    # tool schema
    tools=[{
        "type":"function",
        "function":{
            "name":"search_resumes",
            "description":"Search resumes using structured filters.",
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
        messages=context,
        tools=tools,
        tool_choice="auto",
        response_format={"type":"json_object"}
    ).choices[0].message

    # assistant natural reply
    if rsp.content:
        st.chat_message("assistant").markdown(rsp.content)
        st.session_state.chat.append({"role":"assistant","content":rsp.content})

    # tool execution
    if rsp.tool_calls:
        args=json.loads(rsp.tool_calls[0].function.arguments)
        rows=search_resumes(args)
        summary=f"{len(rows)} resumes for {args.get('country','?')}"

        # save search
        st.session_state.searches.append({"summary":summary,"rows":rows})

        # brief assistant confirmation
        confirm=f"Here are the top {len(rows)} matches."
        st.chat_message("assistant").markdown(confirm)
        st.session_state.chat.append({"role":"assistant","content":confirm})

        # immediately render cards for new search
        for r in rows:
            with st.container(border=True):
                st.markdown(f"**{r.get('name','Unnamed')}** — {r.get('country','N/A')}")
                st.markdown(
                    f"📧 {r.get('email','N/A')}   📱 {r.get('contactNo','N/A')}<br>"
                    f"🛠 {', '.join(s.get('skillName','') for s in r.get('skills',[]))}",
                    unsafe_allow_html=True
                )

    st.session_state.busy=False
