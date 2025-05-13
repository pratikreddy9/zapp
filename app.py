import json, streamlit as st
from pymongo import MongoClient
from openai import OpenAI

st.set_page_config(page_title="Resume Chat (2‑Agent)", layout="wide")
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# ── PROMPTS ──────────────────────────────────────────────────────────────────
PLANNER_SYS = """
You are the Planner agent. If you have enough info, output ONLY:
{"action":"query_db","filters":{...}}   (schema: country,min_experience_years,
max_experience_years,job_titles,skills,top_k)
Otherwise reply conversationally.
"""
RESPONDER_SYS = "You are the Responder agent.  Use the rows to answer."

# ── MONGO SEARCH ─────────────────────────────────────────────────────────────
def mongo_search(f):
    db = MongoClient(
        host="notify.pesuacademy.com", port=27017,
        username="admin", password=st.secrets["MONGO_PASS"],
        authSource="admin"
    )["resumes_database"]["resumes"]

    q = {}
    if c:=f.get("country"): q["country"]={"$regex":f"^{c.strip().lower()}$","$options":"i"}
    if sk:=f.get("skills"):
        q["$or"]=[{"skills.skillName":{"$in":sk}},{"keywords":{"$in":sk}}]
    filters=[]
    if jt:=f.get("job_titles"): filters.append({"jobExperiences.title":{"$in":jt}})
    if me:=f.get("min_experience_years"):
        filters.append({
            "$expr":{"$gte":[{"$toInt":{"$ifNull":[{"$first":"$jobExperiences.duration"},"0"]}},me]}})
    if filters: q["$and"]=filters

    rows=list(db.find(q,{"_id":0,"embedding":0}).limit(int(f.get("top_k",50))))
    db.client.close(); return rows

# ── SESSION STATE ────────────────────────────────────────────────────────────
if "chat" not in st.session_state: st.session_state.chat=[]
if "memory_rows" not in st.session_state: st.session_state.memory_rows=[]

# ── DISPLAY CHAT ─────────────────────────────────────────────────────────────
st.title("🤖 2‑Layer Resume Assistant")
for t in st.session_state.chat:
    st.chat_message(t["role"]).markdown(t["content"])

# ── USER INPUT ───────────────────────────────────────────────────────────────
prompt=st.chat_input("Ask…")
if prompt:
    st.session_state.chat.append({"role":"user","content":prompt})
    st.chat_message("user").markdown(prompt)

    # ① Planner call
    planner_messages=[{"role":"system","content":PLANNER_SYS}]
    planner_messages += [{"role":m["role"],"content":m["content"]} for m in st.session_state.chat[-6:]]
    plan = client.chat.completions.create(
        model="gpt-4o", messages=planner_messages, response_format={"type":"json_object"}
    ).choices[0].message.content

    # Did Planner emit JSON?
    try:
        obj=json.loads(plan)
        if obj.get("action")=="query_db":
            filters=obj["filters"]
            rows=mongo_search(filters)
            st.session_state.memory_rows=rows  # save for responder

            # ② Responder call
            responder_messages=[
                {"role":"system","content":RESPONDER_SYS},
                {"role":"user","content":json.dumps({"question":prompt,"rows":rows[:20]})}
            ]
            resp=client.chat.completions.create(
                model="gpt-4o",messages=responder_messages
            ).choices[0].message.content

            st.chat_message("assistant").markdown(resp)
            st.session_state.chat.append({"role":"assistant","content":resp})

            # show cards
            for r in rows:
                with st.container(border=True):
                    st.markdown(f"**{r['name']}** — {r['country']}")
        else:
            raise ValueError
    except Exception:
        # Planner just replied conversationally
        st.chat_message("assistant").markdown(plan)
        st.session_state.chat.append({"role":"assistant","content":plan})
