import json
import streamlit as st
from datetime import datetime
from pymongo import MongoClient
from openai import OpenAI

# ========== CONFIG ==========
st.set_page_config(page_title="Resume Search Chat", layout="wide")
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# ========== PROMPT ==========
MASTER_PROMPT = """
You are a resume search assistant. When a user types a natural query, return either:
- A follow-up question if more info is needed
- Or a tool call to search MongoDB using structured JSON

The database contains resumes with:
- Fields: resumeId, name, email, contactNo, address, country
- Lists: educationalQualifications[], jobExperiences[], keywords[], skills[]

country is case-insensitive and normalized using .strip().lower().
Experience is in jobExperiences[].duration (convert numeric strings to int).
skills must match both skills[].skillName and keywords[].
Always return structured JSON using the schema if tool call is needed.

Normalize and expand:
- "SQL" → ["SQL", "sql", "mysql", "microsoft sql server"]
- "JavaScript" → ["JavaScript", "javascript", "js", "java script"]
- "Software Developer" → ["Software Developer", "software dev", "softwaredeveloper", "software engineer"]

Return structured JSON with key:
"query_parameters": {
  "country": "...",
  "min_experience_years": ...,
  "max_experience_years": ...,
  "job_titles": [...],
  "skills": [...],
  "top_k": ...
}
"""

# ========== MONGO SEARCH ==========
def get_mongo_client():
    return MongoClient(
        host="notify.pesuacademy.com",
        port=27017,
        username="admin",
        password=st.secrets["MONGO_PASS"],
        authSource="admin"
    )

def search_resumes(params):
    client = get_mongo_client()
    coll = client["resumes_database"]["resumes"]

    country = params.get("country")
    min_exp = params.get("min_experience_years", 0)
    job_titles = params.get("job_titles", [])
    skills = params.get("skills", [])
    top_k = params.get("top_k", 50)

    if isinstance(min_exp, str) and min_exp.isdigit():
        min_exp = int(min_exp)

    query = {}
    if country:
        query["country"] = {"$regex": f"^{country.strip().lower()}$", "$options": "i"}
    if skills:
        query["$or"] = [{"skills.skillName": {"$in": skills}}, {"keywords": {"$in": skills}}]

    filters = []
    if job_titles:
        filters.append({"jobExperiences.title": {"$in": job_titles}})
    if min_exp:
        filters.append({
            "$expr": {
                "$gte": [
                    {"$toInt": {"$ifNull": [{"$first": "$jobExperiences.duration"}, "0"]}},
                    min_exp
                ]
            }
        })
    if filters:
        query["$and"] = filters

    results = list(coll.find(query, {"_id": 0, "embedding": 0}).limit(top_k))
    client.close()
    return results

# ========== STATE INIT ==========
if "chat" not in st.session_state:
    st.session_state.chat = []
if "results" not in st.session_state:
    st.session_state.results = []

# ========== UI CHAT RENDER ==========
st.title("🧠 Resume Agent")

for msg in st.session_state.chat:
    style = "background:#ffecec;" if msg["role"] == "user" else "background:#eaffea;"
    st.markdown(
        f"<div style='{style} padding:10px; border-radius:8px; margin:5px 0'>{msg['content']}</div>",
        unsafe_allow_html=True
    )

# ========== RESUME CARDS ==========
if st.session_state.results:
    st.markdown("### ✅ Matches")
    for r in st.session_state.results:
        st.markdown(
            f"""
            <div style='border:1px solid #ccc;padding:10px;border-radius:8px;margin:5px'>
            <b>{r.get("name", "Unnamed")}</b><br>
            📧 {r.get("email", "N/A")}<br>
            📱 {r.get("contactNo", "N/A")}<br>
            🌍 {r.get("country", "N/A")}<br>
            🛠️ Skills: {", ".join([s.get("skillName", "") for s in r.get("skills", [])])}<br>
            🔑 Keywords: {", ".join(r.get("keywords", []))}
            </div>
            """,
            unsafe_allow_html=True
        )

# ========== INPUT ==========
user_input = st.text_input("Type your query:")

if user_input:
    st.session_state.chat.append({"role": "user", "content": user_input})

    messages = [{"role": "system", "content": MASTER_PROMPT}]
    for m in st.session_state.chat:
        messages.append({"role": m["role"], "content": m["content"]})

    tool_schema = [{
        "type": "function",
        "function": {
            "name": "search_resumes",
            "description": "Search resumes using filters",
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

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            tools=tool_schema,
            tool_choice="auto",
            response_format={"type": "json_object"}
        )

        reply = response.choices[0].message

        if reply.tool_calls:
            args = json.loads(reply.tool_calls[0].function.arguments)
            results = search_resumes(args)
            st.session_state.results = results
            st.session_state.chat.append({"role": "assistant", "content": f"🔍 Found {len(results)} matches."})
        else:
            st.session_state.chat.append({"role": "assistant", "content": reply.content})

        st.rerun()

    except Exception as e:
        st.session_state.chat.append({"role": "assistant", "content": f"❌ Error: {str(e)}"})
        st.rerun()
