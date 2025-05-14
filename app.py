"""
ZappBot: Resume filtering chatbot with enhanced UI and reliable behavior
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

# ── PARSE AND PROCESS RESPONSE ────────────────────────────────────────
def process_response(text):
    """
    Process the response text to:
    1. Extract any introductory text
    2. Extract resume data
    3. Determine conclusion text
    """
    # First, check if this is a resume-listing response
    if "Here are some" in text and ("Experience:" in text or "experience:" in text) and ("Skills:" in text or "skills:" in text):
        # Find the introductory text (everything before the first name)
        # Look for pattern of a blank line followed by a name (text with no indentation)
        intro_pattern = r'^(.*?)\n\n([A-Z][a-z]+.*?)\n\nEmail:'
        intro_match = re.search(intro_pattern, text, re.DOTALL)
        
        intro_text = ""
        if intro_match:
            intro_text = intro_match.group(1).strip()
        
        # Extract the resumes - accommodate both formats (numbered and unnumbered)
        # First try standard format with blank lines
        resume_pattern = r'([A-Z][a-z]+ (?:[A-Z][a-z]+ )?(?:[A-Z][a-z]+)?)\s*\n\s*Email:\s*([^\n]+)\s*\nContact No:\s*([^\n]+)\s*\nLocation:\s*([^\n]+)\s*\nExperience:\s*([^\n]+)\s*\nSkills:\s*([^\n]+)'
        matches = re.findall(resume_pattern, text, re.MULTILINE | re.IGNORECASE)
        
        # If that didn't work, try the numbered format
        if not matches:
            resume_pattern = r'\d+\.\s+\*\*([^*]+)\*\*\s*\n\s*-\s+\*\*Email:\*\*\s+([^\n]+)\s*\n\s*-\s+\*\*Contact No:\*\*\s+([^\n]+)\s*\n\s*-\s+\*\*Location:\*\*\s+([^\n]+)\s*\n\s*-\s+\*\*Experience:\*\*\s+([^\n]+)\s*\n\s*-\s+\*\*Skills:\*\*\s+([^\n]+)'
            matches = re.findall(resume_pattern, text, re.MULTILINE)
        
        # Extract the conclusion (after all resumes)
        # Look for lines that contain phrases like "These candidates" or similar conclusion statements
        conclusion_pattern = r'(These candidates.*?)\s*$'
        conclusion_match = re.search(conclusion_pattern, text, re.DOTALL)
        
        conclusion_text = ""
        if conclusion_match:
            conclusion_text = conclusion_match.group(1).strip()
        
        # Convert resume matches to structured data
        resumes = []
        for match in matches:
            name, email, contact, location, experience, skills = match
            
            # Split skills and experience
            skill_list = [s.strip() for s in skills.split(',')]
            exp_list = [e.strip() for e in experience.split(',')]
            
            resumes.append({
                "name": name.strip(),
                "email": email.strip(),
                "contactNo": contact.strip(),
                "location": location.strip(),
                "experience": exp_list,
                "skills": skill_list
            })
        
        return {
            "is_resume_response": True,
            "intro_text": intro_text,
            "resumes": resumes,
            "conclusion_text": conclusion_text,
            "full_text": text  # Keep this for debugging
        }
    else:
        # Not a resume listing response
        return {
            "is_resume_response": False,
            "full_text": text
        }

# ── DISPLAY RESUME GRID ───────────────────────────────────────────────
def display_resume_grid(resumes, container=None):
    """Display resumes in a 3x3 grid layout with styled cards."""
    target = container if container else st
    
    if not resumes:
        target.warning("No resumes found matching the criteria.")
        return
    
    # Custom CSS for the resume cards
    target.markdown("""
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
    
    # Create a 3-column grid
    num_resumes = len(resumes)
    rows = (num_resumes + 2) // 3  # Ceiling division for number of rows
    
    for row in range(rows):
        cols = target.columns(3)
        for col in range(3):
            idx = row * 3 + col
            if idx < num_resumes:
                resume = resumes[idx]
                
                # Extract resume data
                name = resume.get("name", "Unknown")
                email = resume.get("email", "")
                phone = resume.get("contactNo", "")
                location = resume.get("location", "")
                
                # Get experience and skills
                experience = resume.get("experience", [])
                skills = resume.get("skills", [])
                
                with cols[col]:
                    html = f"""
                    <div class="resume-card">
                        <div class="resume-name">{name}</div>
                        <div class="resume-location">📍 {location}</div>
                        <div class="resume-contact">📧 {email}</div>
                        <div class="resume-contact">📱 {phone}</div>
                    """
                    
                    # Add experience section
                    if experience:
                        html += f'<div class="resume-section-title">Experience</div>'
                        for exp in experience[:3]:  # Limit to 3 experiences
                            html += f'<div class="resume-experience">• {exp}</div>'
                    
                    # Add skills section
                    if skills:
                        html += f'<div class="resume-section-title">Skills</div><div>'
                        for skill in skills[:7]:  # Limit to 7 skills
                            html += f'<span class="skill-tag">{skill}</span>'
                        html += '</div>'
                    
                    html += '</div>'
                    target.markdown(html, unsafe_allow_html=True)

# ── AGENT + MEMORY ─────────────────────────────────────────────────────
llm = ChatOpenAI(model=MODEL_NAME, api_key=OPENAI_API_KEY, temperature=0)

# Use the original, simple prompt that worked reliably
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful HR assistant named ZappBot. Use the `query_db` tool whenever the "
            "user asks for candidates or filtering. Otherwise, answer normally.",
        ),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
)

if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(
        memory_key="chat_history", return_messages=True
    )

if "agent_executor" not in st.session_state:
    agent = create_openai_tools_agent(llm, [query_db], prompt)
    st.session_state.agent_executor = AgentExecutor(
        agent=agent, tools=[query_db], memory=st.session_state.memory, verbose=True
    )

# Store resume responses for UI display
if "resume_responses" not in st.session_state:
    st.session_state.resume_responses = []

# ── STREAMLIT UI ───────────────────────────────────────────────────────
st.set_page_config(page_title="ZappBot", layout="wide")

# Apply custom CSS
st.markdown("""
<style>
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
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
    .resume-section {
        margin-top: 20px;
        padding: 15px;
        border-radius: 8px;
        background-color: #f8f9fa;
        border-left: 4px solid #0366d6;
    }
    .resume-query {
        font-weight: 600;
        margin-bottom: 10px;
        color: #0366d6;
    }
    .st-expander {
        border: none !important;
        box-shadow: none !important;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<div class="header-container"><div class="header-emoji">⚡</div><div class="header-text">ZappBot</div></div>', unsafe_allow_html=True)

# Sidebar with settings
with st.sidebar:
    st.header("Settings")
    debug_mode = st.checkbox("Debug Mode", value=False)
    if st.button("Clear Chat History"):
        st.session_state.memory.clear()
        st.session_state.resume_responses = []
        st.rerun()

# Main chat container
chat_container = st.container()

# Handle user input
user_input = st.chat_input("Ask me to find resumes...")
if user_input:
    # Show user message
    st.chat_message("user").write(user_input)
    
    # Process with agent - use the original method that works reliably
    with st.spinner("Thinking..."):
        try:
            # This is the key part - invoke the agent with the simple approach that works well
            response = st.session_state.agent_executor.invoke({"input": user_input})
            assistant_response = response["output"]
            
            # Display the assistant's message
            st.chat_message("assistant").write(assistant_response)
            
            # Process the response to extract resume data
            processed = process_response(assistant_response)
            
            # If this is a resume response, store it for the UI display
            if processed["is_resume_response"]:
                # Store the user query and processed response
                st.session_state.resume_responses.append({
                    "query": user_input,
                    "intro": processed["intro_text"],
                    "resumes": processed["resumes"],
                    "conclusion": processed.get("conclusion_text", ""),
                    "timestamp": datetime.now().isoformat()
                })
            
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
            # First, display the complete message text as is
            st.chat_message("assistant").write(msg.content)
            
            # Check if this message contains resume data, and if we don't already have it stored
            processed = process_response(msg.content)
            if processed["is_resume_response"]:
                # Find the corresponding user query if possible
                user_query = "Resume Search"
                idx = st.session_state.memory.chat_memory.messages.index(msg)
                if idx > 0 and st.session_state.memory.chat_memory.messages[idx-1].type == "human":
                    user_query = st.session_state.memory.chat_memory.messages[idx-1].content
                
                # Check if we already have this response stored
                found = False
                for resp in st.session_state.resume_responses:
                    if resp["query"] == user_query:
                        found = True
                        break
                
                # If not found, add it
                if not found:
                    st.session_state.resume_responses.append({
                        "query": user_query,
                        "intro": processed["intro_text"],
                        "resumes": processed["resumes"],
                        "conclusion": processed.get("conclusion_text", ""),
                        "timestamp": datetime.now().isoformat()
                    })

# Display all resume search results in the separate section
if st.session_state.resume_responses:
    st.markdown("---")
    st.subheader("Resume Search Results")
    
    # Create an expander for each resume search
    for i, resp in enumerate(st.session_state.resume_responses):
        with st.expander(f"Search {i+1}: {resp['query']}", expanded=(i == len(st.session_state.resume_responses)-1)):
            st.markdown(f"<div class='resume-query'>{resp['intro']}</div>", unsafe_allow_html=True)
            display_resume_grid(resp['resumes'])
            if resp['conclusion']:
                st.write(resp['conclusion'])

# Show debug info if enabled
if debug_mode:
    with st.expander("Debug Information"):
        st.subheader("Memory Contents")
        st.json({i: msg.content for i, msg in enumerate(st.session_state.memory.chat_memory.messages)})
        
        st.subheader("Resume Responses")
        display_data = []
        for i, resp in enumerate(st.session_state.resume_responses):
            display_data.append({
                "index": i,
                "query": resp["query"],
                "timestamp": resp["timestamp"],
                "resume_count": len(resp["resumes"])
            })
        st.json(display_data)
