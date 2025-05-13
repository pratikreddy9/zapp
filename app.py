import json
import streamlit as st
from datetime import datetime
from pymongo import MongoClient
from openai import OpenAI

# ========== INIT ==========
st.set_page_config(page_title="Resume Chat", layout="wide")
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# ========== MONGO SETUP ==========
def get_mongo_client():
    return MongoClient(
        host="notify.pesuacademy.com",
        port=27017,
        username="admin",
        password=st.secrets["MONGO_PASS"],
        authSource="admin"
    )

# ========== RESUME FILTER FUNCTION ==========
def search_resumes(filters):
    mongo = get_mongo_client()
    coll = mongo["resumes_database"]["resumes"]

    country = filters.get("country")
    min_exp = filters.get("min_experience_years", 0)
    job_titles = filters.get("job_titles", [])
    skills = filters.get("skills", [])
    top_k = filters.get("top_k", 50)

    if isinstance(min_exp, str) and min_exp.isdigit():
        min_exp = int(min_exp)

    query = {}
    if country:
        query["country"] = {"$regex": f"^{country.strip().lower()}$", "$options": "i"}
    if skills:
        query["$or"] = [{"skills.skillName": {"$in": skills}}, {"keywords": {"$in": skills}}]

    job_exp_filters = []
    if job_titles:
        job_exp_filters.append({"jobExperiences.title": {"$in": job_titles}})
    if min_exp:
        job_exp_filters.append({
            "$expr": {
                "$gte": [
                    {"$toInt": {"$ifNull": [{"$first": "$jobExperiences.duration"}, "0"]}},
                    min_exp
                ]
            }
        })

    if job_exp_filters:
        query["$and"] = job_exp_filters

    results = list(coll.find(query, {"_id": 0, "embedding": 0}).limit(top_k))
    mongo.close()
    return results

# ========== UI STATE INIT ==========
if "chat" not in st.session_state:
    st.session_state.chat = []
if "pending_input" not in st.session_state:
    st.session_state.pending_input = None

# ========== SHOW CHAT ==========
for msg in st.session_state.chat:
    role = msg["role"]
    if role == "user":
        st.markdown(f"<div style='background:#fce2e2;padding:10px;border-radius:8px;margin:5px 0;text-align:right'>{msg['content']}</div>", unsafe_allow_html=True)
    elif role == "assistant" and isinstance(msg["content"], str):
        st.markdown(f"<div style='background:#e2fce2;padding:10px;border-radius:8px;margin:5px 0;text-align:left'>{msg['content']}</div>", unsafe_allow_html=True)
    elif role == "assistant" and isinstance(msg["content"], list):
        for res in msg["content"]:
            with st.container():
                st.markdown(f"""
                    <div style='border:1px solid #ccc;padding:10px;border-radius:8px;margin-bottom:10px'>
                    <b>{res.get("name", "Unnamed")}</b><br>
                    📧 {res.get("email", "N/A")}<br>
                    📍 {res.get("country", "N/A")}<br>
                    🛠 Skills: {", ".join(k.get("skillName", "") for k in res.get("skills", []))}<br>
                    🔑 Keywords: {", ".join(res.get("keywords", []))}
                    </div>
                """, unsafe_allow_html=True)

# ========== INPUT BOX ==========
user_input = st.text_input("You:", key="chat_input")

if user_input:
    st.session_state.chat.append({"role": "user", "content": user_input})
    st.session_state.pending_input = user_input
    st.experimental_rerun()

# ========== GPT CALL ==========
if st.session_state.pending_input:
    user_input = st.session_state.pending_input
    st.session_state.pending_input = None

    tool_def = [{
        "type": "function",
        "function": {
            "name": "search_resumes",
            "description": "Search resumes from the MongoDB using given filters.",
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

    messages = [
        {"role": "system", "content": "You're a resume filtering agent. If enough info is present, call the function. Always output valid JSON."}
    ] + [{"role": m["role"], "content": m["content"]} for m in st.session_state.chat if m["role"] in ("user", "assistant")]

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            tools=tool_def,
            tool_choice="auto",
            response_format={"type": "json_object"}
        )

        msg = response.choices[0].message

        if msg.tool_calls:
            args = json.loads(msg.tool_calls[0].function.arguments)
            results = search_resumes(args)

            if not results:
                st.session_state.chat.append({"role": "assistant", "content": "No matching resumes found."})
            else:
                st.session_state.chat.append({"role": "assistant", "content": results})

        elif msg.content:
            st.session_state.chat.append({"role": "assistant", "content": msg.content})

        st.experimental_rerun()

    except Exception as e:
        st.session_state.chat.append({"role": "assistant", "content": f"❌ Error: {str(e)}"})
        st.experimental_rerun()
