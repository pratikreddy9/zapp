import json
import requests
import streamlit as st
from pymongo import MongoClient
from datetime import datetime

st.set_page_config(page_title="Resume Chat Agent", layout="wide")

# ✅ MongoDB Setup
def get_mongo_client():
    return MongoClient(
        host="notify.pesuacademy.com",
        port=27017,
        username="admin",
        password=st.secrets["MONGO_PASS"],
        authSource="admin"
    )

# ✅ OpenAI Setup
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
OPENAI_MODEL = "gpt-4o"
OPENAI_URL = "https://api.openai.com/v1/chat/completions"

# ✅ Resume Search Logic (Your OG logic)
def search_resumes_via_mongo(filters):
    mongo_client = get_mongo_client()
    resumes_collection = mongo_client["resumes_database"]["resumes"]

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
                    {
                        "$toInt": {
                            "$ifNull": [{"$first": "$jobExperiences.duration"}, "0"]
                        }
                    },
                    min_exp
                ]
            }
        })

    if job_exp_filters:
        query["$and"] = job_exp_filters

    results = list(resumes_collection.find(query, {"_id": 0, "embedding": 0}).limit(top_k))
    mongo_client.close()
    return results

# ✅ Chat memory
if "messages" not in st.session_state:
    st.session_state.messages = []

# ✅ Render chat history
for msg in st.session_state.messages:
    box = "#ffe0e0" if msg["role"] == "user" else "#e0ffe0"
    st.markdown(f"<div style='background-color:{box};padding:10px;border-radius:10px;margin:5px 0'>{msg['content']}</div>", unsafe_allow_html=True)

# ✅ User input
user_input = st.text_input("You:", key="input")
if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})

    # Build full message history
    history = [{"role": "system", "content": "You are a resume assistant that can call a function to search MongoDB when enough info is present."}]
    history += [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages]

    # Function schema
    tools = [{
        "type": "function",
        "function": {
            "name": "search_resumes",
            "description": "Search resumes based on structured filters",
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

    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": OPENAI_MODEL,
        "messages": history,
        "tools": tools,
        "tool_choice": "auto",
        "response_format": "json"
    }

    response = requests.post(OPENAI_URL, headers=headers, json=payload)
    response.raise_for_status()
    data = response.json()
    msg_obj = data["choices"][0]["message"]

    # If GPT wants to call search
    if "tool_calls" in msg_obj:
        tool_call = msg_obj["tool_calls"][0]
        args = json.loads(tool_call["function"]["arguments"])
        results = search_resumes_via_mongo(args)
        reply = f"✅ Found {len(results)} matching resumes.\n\nExamples:\n" + "\n".join([r["name"] for r in results[:5]])
    else:
        reply = msg_obj["content"]

    st.session_state.messages.append({"role": "assistant", "content": reply})
    st.rerun()
