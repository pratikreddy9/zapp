import json
import streamlit as st
from datetime import datetime
from pymongo import MongoClient
from openai import OpenAI

# ========== INIT ==========
st.set_page_config(page_title="Resume Search Chat", layout="wide")
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

# ========== FUNCTION TOOL ==========
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

# ========== CHAT UI ==========
if "chat" not in st.session_state:
    st.session_state.chat = []

for msg in st.session_state.chat:
    box = "#fff3f3" if msg["role"] == "user" else "#f3fff3"
    st.markdown(f"<div style='background-color:{box};padding:10px;border-radius:8px;margin:5px 0'>{msg['content']}</div>", unsafe_allow_html=True)

user_input = st.text_input("You:", key="chat_input")

if user_input:
    st.session_state.chat.append({"role": "user", "content": user_input})

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

    messages = [{"role": "system", "content": "You're a resume filtering agent. Use the function only if all required parameters are present."}]
    for m in st.session_state.chat:
        messages.append({"role": m["role"], "content": m["content"]})

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            tools=tool_def,
            tool_choice="auto",
            response_format="json"
        )

        msg_obj = response.choices[0].message

        if msg_obj.tool_calls:
            tool_call = msg_obj.tool_calls[0]
            args = json.loads(tool_call.function.arguments)
            results = search_resumes(args)
            names = [r.get("name", "Unnamed") for r in results[:5]]
            reply = f"✅ Found {len(results)} resumes. Sample:\n- " + "\n- ".join(names)
        else:
            reply = msg_obj.content

        st.session_state.chat.append({"role": "assistant", "content": reply})
        st.rerun()

    except Exception as e:
        st.session_state.chat.append({"role": "assistant", "content": f"❌ Error: {str(e)}"})
        st.rerun()
