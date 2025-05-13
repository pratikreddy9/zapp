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
You are a resume search assistant. Given a user query, either ask clarifying questions or call a tool that searches MongoDB resumes using:

{
  "country": "...",
  "min_experience_years": ...,
  "max_experience_years": ...,
  "job_titles": [...],
  "skills": [...],
  "top_k": ...
}

Resume structure includes:
- Fields: resumeId, name, email, contactNo, address, country (normalized lowercase)
- Lists: educationalQualifications[], jobExperiences[], keywords[], skills[].skillName

Match skills with both `skills[].skillName` and `keywords`. Normalize casing and spacing.

Expand:
- "SQL" → ["SQL", "sql", "mysql", "microsoft sql server"]
- "Python" → ["Python", "python"]
- "Software Developer" → ["Software Developer", "software dev", "softwaredeveloper", "software engineer"]

Always use `response_format={"type": "json_object"}`.
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

    if isinstance(top_k, str) and top_k.isdigit():
        top_k = int(top_k)

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

# ========== STREAMLIT STATE ==========
if "chat" not in st.session_state:
    st.session_state.chat = []
if "results" not in st.session_state:
    st.session_state.results = []
if "last_query_pending" not in st.session_state:
    st.session_state.last_query_pending = None

# ========== CHAT UI ==========
st.title("🤖 Resume Agent")

for msg in st.session_state.chat:
    bg = "#ffecec" if msg["role"] == "user" else "#eaffea"
    st.markdown(
        f"<div style='background:{bg};padding:10px;border-radius:8px;margin:5px 0'>{msg['content']}</div>",
        unsafe_allow_html=True
    )

if st.session_state.results:
    st.markdown("### 🔍 Matching Resumes")
    for r in st.session_state.results:
        st.markdown(
            f"""
            <div style='border:1px solid #ccc;padding:10px;border-radius:8px;margin:5px'>
            <b>{r.get("name", "Unnamed")}</b><br>
            📧 {r.get("email", "N/A")}<br>
            📱 {r.get("contactNo", "N/A")}<br>
            🌍 {r.get("country", "N/A")}<br>
            🛠 Skills: {", ".join(s.get("skillName", "") for s in r.get("skills", []))}<br>
            🔑 Keywords: {", ".join(r.get("keywords", []))}
            </div>
            """,
            unsafe_allow_html=True
        )

# ========== INPUT ==========
user_input = st.text_input("Ask your query:")
if user_input:
    st.session_state.chat.append({"role": "user", "content": user_input})
    st.session_state.last_query_pending = user_input
    st.rerun()

# ========== GPT + TOOL CALL ==========
if st.session_state.last_query_pending:
    query = st.session_state.last_query_pending
    st.session_state.last_query_pending = None

    messages = [{"role": "system", "content": MASTER_PROMPT}]
    for m in st.session_state.chat:
        messages.append({"role": m["role"], "content": m["content"]})

    tools = [{
        "type": "function",
        "function": {
            "name": "search_resumes",
            "description": "Search resumes using structured filters",
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
            tools=tools,
            tool_choice="auto",
            response_format={"type": "json_object"}
        )

        msg = response.choices[0].message

        if msg.tool_calls:
            # Show GPT's message if available
            if msg.content:
                st.session_state.chat.append({"role": "assistant", "content": msg.content})

            args = json.loads(msg.tool_calls[0].function.arguments)
            results = search_resumes(args)

            # Save results + reply
            st.session_state.results = results
            summary = f"🔍 Found {len(results)} resumes matching your criteria."
            st.session_state.chat.append({"role": "assistant", "content": summary})

        else:
            st.session_state.chat.append({"role": "assistant", "content": msg.content})

        st.rerun()

    except Exception as e:
        st.session_state.chat.append({"role": "assistant", "content": f"❌ Error: {str(e)}"})
        st.rerun()
