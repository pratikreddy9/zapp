"""
Resume‑filtering chatbot with conversation memory and grid display
LangChain 0.3.25 • OpenAI 1.78.1 • Streamlit 1.34+
"""

import os, json, re
from datetime import datetime
from typing import List, Optional, Dict, Any

import streamlit as st
from pymongo import MongoClient
from pymongo.errors import PyMongoError

from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain.agents import create_openai_tools_agent, AgentExecutor
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationBufferMemory
import openai

# ── CONFIG ─────────────────────────────────────────────────────────────
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
MONGO_CFG = {
    "host": "notify.pesuacademy.com",
    "port": 27017,
    "username": "admin",
    "password": st.secrets["MONGO_PASS"],
    "authSource": "admin",
}
MODEL_NAME = "gpt-4o"
EVAL_MODEL_NAME = "gpt-4o"
TOP_K_DEFAULT = 50
DB_NAME = "resumes_database"
COLL_NAME = "resumes"

# ── MONGO ──────────────────────────────────────────────────────────────
def get_mongo_client() -> MongoClient:
    return MongoClient(**MONGO_CFG)

# ── NORMALIZATION ──────────────────────────────────────────────────────
COUNTRY_EQUIV = {
    "indonesia": ["indonesia"],
    "vietnam": ["vietnam", "viet nam", "vn", "vietnamese"],
    "united states": ["united states", "usa", "us"],
    "malaysia": ["malaysia"],
    "india": ["india", "ind"],
    "singapore": ["singapore"],
    "philippines": ["philippines", "the philippines"],
    "australia": ["australia"],
    "new zealand": ["new zealand"],
    "germany": ["germany"],
    "saudi arabia": ["saudi arabia", "ksa"],
    "japan": ["japan"],
    "hong kong": ["hong kong", "hong kong sar"],
    "thailand": ["thailand"],
    "united arab emirates": ["united arab emirates", "uae"],
}
SKILL_VARIANTS = {
    "sql": ["sql", "mysql", "microsoft sql server"],
    "javascript": ["javascript", "js", "java script"],
    "c#": ["c#", "c sharp", "csharp"],
    "html": ["html", "hypertext markup language"],
}
TITLE_VARIANTS = {
    "software developer": [
        "software developer",
        "software dev",
        "softwaredeveloper",
        "software engineer",
    ],
    "backend developer": [
        "backend developer",
        "backend dev",
        "back-end developer",
        "server-side developer",
    ],
    "frontend developer": [
        "frontend developer",
        "frontend dev",
        "front-end developer",
    ],
}
def expand(values: List[str], table: Dict[str, List[str]]) -> List[str]:
    out = set()
    for v in values:
        v_low = v.strip().lower()
        out.update(table.get(v_low, []))
        out.add(v)
    return list(out)

# ── LLM‑BASED RESUME SCORER ────────────────────────────────────────────
_openai_client = openai.OpenAI(api_key=OPENAI_API_KEY)
EVALUATOR_PROMPT = """
You are a resume scoring assistant. Return only the 10 best resumeIds.

JSON format:
{
  "top_resume_ids": [...],
  "completed_at": "ISO"
}
"""
def score_resumes(query: str, resumes: List[Dict[str, Any]]) -> List[str]:
    chat = _openai_client.chat.completions.create(
        model=EVAL_MODEL_NAME,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": EVALUATOR_PROMPT},
            {
                "role": "user",
                "content": f"Query: {query}\n\nResumes: {json.dumps(resumes)}",
            },
        ],
    )
    content = json.loads(chat.choices[0].message.content)
    return content.get("top_resume_ids", [])

# ── TOOL: query_db ─────────────────────────────────────────────────────
@tool
def query_db(
    query: str,
    country: Optional[str] = None,
    min_experience_years: Optional[int] = None,
    max_experience_years: Optional[int] = None,
    job_titles: Optional[List[str]] = None,
    skills: Optional[List[str]] = None,
    top_k: int = TOP_K_DEFAULT,
) -> Dict[str, Any]:
    """Filter MongoDB resumes and return top 10 matches."""
    try:
        mongo_q: Dict[str, Any] = {}
        if country:
            mongo_q["country"] = {"$in": COUNTRY_EQUIV.get(country.strip().lower(), [country])}
        if skills:
            expanded = expand(skills, SKILL_VARIANTS)
            mongo_q["$or"] = [
                {"skills.skillName": {"$in": expanded}},
                {"keywords": {"$in": expanded}},
            ]
        and_clauses = []
        if job_titles:
            and_clauses.append({"jobExperiences.title": {"$in": expand(job_titles, TITLE_VARIANTS)}})
        if isinstance(min_experience_years, int) and min_experience_years > 0:
            and_clauses.append(
                {
                    "$expr": {
                        "$gte": [
                            {"$toInt": {"$ifNull": [{"$first": "$jobExperiences.duration"}, "0"]}},
                            min_experience_years,
                        ]
                    }
                }
            )
        if and_clauses:
            mongo_q["$and"] = and_clauses
        with get_mongo_client() as client:
            coll = client[DB_NAME][COLL_NAME]
            candidates = list(coll.find(mongo_q, {"_id": 0, "embedding": 0}).limit(top_k))
        best_ids = score_resumes(query, candidates)
        best_resumes = [r for r in candidates if r["resumeId"] in best_ids]
        return {
            "message": f"{len(best_resumes)} resumes after scoring.",
            "results_count": len(best_resumes),
            "results": best_resumes,
            "completed_at": datetime.utcnow().isoformat(),
        }
    except PyMongoError as err:
        return {"error": f"DB error: {str(err)}"}
    except Exception as exc:
        return {"error": str(exc)}

# ── RESUME GRID DISPLAY ───────────────────────────────────────────────────
def extract_resume_data(resume):
    """
    Extract and normalize resume data for display.
    Adapt this function based on your actual resume data structure.
    """
    # Basic info
    name = resume.get("name", "Unknown")
    email = resume.get("email", "")
    phone = resume.get("contactNo", "")
    location = resume.get("location", resume.get("country", ""))
    
    # Skills
    skills = []
    if "skills" in resume:
        if isinstance(resume["skills"], list):
            for skill in resume["skills"]:
                if isinstance(skill, dict) and "skillName" in skill:
                    skills.append(skill["skillName"])
                elif isinstance(skill, str):
                    skills.append(skill)
    
    # Experience
    experience = []
    if "jobExperiences" in resume and isinstance(resume["jobExperiences"], list):
        for job in resume["jobExperiences"]:
            if isinstance(job, dict) and "title" in job:
                experience.append(job["title"])
                
    # If we couldn't extract structured experience, look for raw text
    if not experience and "experience" in resume:
        if isinstance(resume["experience"], str):
            experience = [item.strip() for item in resume["experience"].split(",")]
    
    return {
        "name": name,
        "email": email,
        "phone": phone,
        "location": location,
        "skills": skills,
        "experience": experience
    }

def display_resume_grid(resumes):
    """Display resumes in a 3x3 grid layout with styled cards."""
    if not resumes:
        st.warning("No resumes found matching the criteria.")
        return
    
    # Custom CSS for the resume cards
    st.markdown("""
    <style>
    .resume-card {
        border: 1px solid #e1e4e8;
        border-radius: 10px;
        padding: 16px;
        margin-bottom: 15px;
        background-color: white;
        box-shadow: 0 3px 8px rgba(0,0,0,0.05);
        height: 100%;
        transition: transform 0.2s, box-shadow 0.2s;
    }
    .resume-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 5px 15px rgba(0,0,0,0.1);
    }
    .resume-name {
        font-weight: bold;
        font-size: 18px;
        margin-bottom: 8px;
        color: #24292e;
    }
    .resume-location {
        color: #586069;
        font-size: 14px;
        margin-bottom: 10px;
    }
    .resume-contact {
        margin-bottom: 8px;
        font-size: 14px;
        color: #444d56;
    }
    .resume-section-title {
        font-weight: 600;
        margin-top: 12px;
        margin-bottom: 6px;
        font-size: 15px;
        color: #24292e;
    }
    .resume-experience {
        font-size: 14px;
        color: #444d56;
        margin-bottom: 4px;
    }
    .skill-tag {
        display: inline-block;
        background-color: #f1f8ff;
        color: #0366d6;
        border-radius: 12px;
        padding: 3px 10px;
        margin: 3px;
        font-size: 12px;
        font-weight: 500;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Calculate grid layout
    num_resumes = len(resumes)
    rows = (num_resumes + 2) // 3  # Ceiling division for number of rows
    
    for row in range(rows):
        cols = st.columns(3)
        for col in range(3):
            idx = row * 3 + col
            if idx < num_resumes:
                resume = resumes[idx]
                
                # Get normalized resume data
                data = extract_resume_data(resume)
                
                with cols[col]:
                    html = f"""
                    <div class="resume-card">
                        <div class="resume-name">{data['name']}</div>
                        <div class="resume-location">📍 {data['location']}</div>
                        <div class="resume-contact">📧 {data['email']}</div>
                        <div class="resume-contact">📱 {data['phone']}</div>
                    """
                    
                    # Add experience section
                    if data['experience']:
                        html += f'<div class="resume-section-title">Experience</div>'
                        for exp in data['experience'][:3]:  # Limit to 3 experiences
                            html += f'<div class="resume-experience">• {exp}</div>'
                    
                    # Add skills section
                    if data['skills']:
                        html += f'<div class="resume-section-title">Skills</div><div>'
                        for skill in data['skills'][:7]:  # Limit to 7 skills
                            html += f'<span class="skill-tag">{skill}</span>'
                        html += '</div>'
                    
                    html += '</div>'
                    st.markdown(html, unsafe_allow_html=True)

# ── RESUME EXTRACTION FROM ASSISTANT RESPONSES ─────────────────────────
def extract_resumes_from_response(response_text):
    """
    Extract resume data from the assistant's response text.
    This handles the case where the resumes might be formatted
    as text in the response rather than structured data.
    """
    # Simple regex pattern to identify candidate blocks
    # Adjust based on your typical output format
    candidate_pattern = r'([A-Z][a-z]+ [A-Z][a-z]+(?: [A-Z][a-z]+)?)\s*\n\s*Email: ([^\n]+)\s*\nContact No: ([^\n]+)\s*\nLocation: ([^\n]+)\s*\nExperience: ([^\n]+)\s*\nSkills: ([^\n]+)'
    
    matches = re.findall(candidate_pattern, response_text)
    
    resumes = []
    for match in matches:
        name, email, contact, location, experience, skills = match
        resumes.append({
            "name": name,
            "email": email,
            "contactNo": contact,
            "location": location,
            "experience": experience,
            "skills": skills.split(', ')
        })
    
    return resumes

# ── AGENT + MEMORY ─────────────────────────────────────────────────────
llm = ChatOpenAI(model=MODEL_NAME, api_key=OPENAI_API_KEY, temperature=0)
agent_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a helpful HR assistant helping to find top candidates.
               Use the `query_db` tool whenever the user asks for candidates or filtering.
               
               When listing candidates, always format the data consistently for each candidate:
               1. Name
               2. Email
               3. Contact No
               4. Location (country or city)
               5. Experience (list all relevant positions)
               6. Skills (list all technical skills)
               
               This formatting helps the UI display candidates properly in a grid format.
               
               Otherwise, answer normally to other questions.
            """,
        ),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
)

# Initialize memory and agent
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(
        memory_key="chat_history", return_messages=True
    )
if "agent_executor" not in st.session_state:
    agent = create_openai_tools_agent(llm, [query_db], agent_prompt)
    st.session_state.agent_executor = AgentExecutor(
        agent=agent, tools=[query_db], memory=st.session_state.memory, verbose=True
    )

# Track if we're currently displaying resumes
if "displaying_resumes" not in st.session_state:
    st.session_state.displaying_resumes = False
if "latest_resumes" not in st.session_state:
    st.session_state.latest_resumes = []

# ── STREAMLIT UI ───────────────────────────────────────────────────────
st.set_page_config(page_title="Resume Filtering Chatbot", layout="wide")

# Apply custom CSS
st.markdown("""
<style>
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    .main-header {
        font-family: 'Helvetica Neue', sans-serif;
        color: #1E3A8A;
        margin-bottom: 20px;
    }
    .header-container {
        display: flex;
        align-items: center;
        margin-bottom: 20px;
    }
    .header-emoji {
        font-size: 36px;
        margin-right: 10px;
    }
    .header-text {
        font-size: 24px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<div class="header-container"><div class="header-emoji">🧠</div><div class="header-text">Resume Filtering Chatbot</div></div>', unsafe_allow_html=True)

# Sidebar with settings
with st.sidebar:
    st.header("Settings")
    debug_mode = st.checkbox("Debug Mode", value=False)
    if st.button("Clear Chat History"):
        st.session_state.memory.clear()
        st.session_state.displaying_resumes = False
        st.session_state.latest_resumes = []
        st.rerun()

# Update agent settings
if "agent_executor" in st.session_state:
    st.session_state.agent_executor.verbose = debug_mode

# Handle user input
user_input = st.chat_input("Ask me to find resumes...")
if user_input:
    # Display user message
    st.chat_message("user").write(user_input)
    
    # Process with agent
    with st.spinner("Thinking..."):
        try:
            response = st.session_state.agent_executor.invoke({"input": user_input})
            agent_response = response.get("output", "")
            
            # Check if the response contains resume data
            found_resumes = []
            
            # Try to find structured resume data
            if isinstance(response, dict):
                # Look through the whole response recursively for results
                def find_results(obj):
                    if isinstance(obj, dict):
                        if "results" in obj and isinstance(obj["results"], list):
                            return obj["results"]
                        for k, v in obj.items():
                            result = find_results(v)
                            if result:
                                return result
                    elif isinstance(obj, list):
                        for item in obj:
                            result = find_results(item)
                            if result:
                                return result
                    return None
                
                results = find_results(response)
                if results:
                    found_resumes = results
            
            # If no structured data, try to extract from text
            if not found_resumes and isinstance(agent_response, str):
                extracted_resumes = extract_resumes_from_response(agent_response)
                if extracted_resumes:
                    found_resumes = extracted_resumes
            
            # Display response and resumes
            st.chat_message("assistant").write(agent_response)
            
            if found_resumes:
                st.session_state.displaying_resumes = True
                st.session_state.latest_resumes = found_resumes
                st.subheader(f"Found {len(found_resumes)} matching resumes:")
                display_resume_grid(found_resumes)
            
        except Exception as e:
            st.error(f"Error: {str(e)}")
            if debug_mode:
                st.exception(e)
else:
    # Display chat history
    for msg in st.session_state.memory.chat_memory.messages:
        if msg.type == "human":
            st.chat_message("user").write(msg.content)
        else:
            st.chat_message("assistant").write(msg.content)
    
    # If we were previously displaying resumes, show them again
    if st.session_state.displaying_resumes and st.session_state.latest_resumes:
        st.subheader(f"Found {len(st.session_state.latest_resumes)} matching resumes:")
        display_resume_grid(st.session_state.latest_resumes)
