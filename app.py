"""
Resume‑filtering chatbot with conversation memory and debugging
LangChain 0.3.25 • OpenAI 1.78.1 • Streamlit 1.34+
"""

import os, json
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

# ── SIMPLE RESUME DISPLAY FUNCTION ───────────────────────────────────────
def display_resume_grid(resumes):
    """Display resumes in a simple grid layout."""
    if not resumes:
        st.warning("No resumes found matching the criteria.")
        return
    
    # Apply some basic CSS for resume cards
    st.markdown("""
    <style>
    .resume-card {
        border: 1px solid #ddd;
        border-radius: 8px;
        padding: 15px;
        margin-bottom: 15px;
        background-color: white;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    }
    .resume-name {
        font-weight: bold;
        font-size: 16px;
        margin-bottom: 10px;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Create a 3-column grid
    cols = st.columns(3)
    
    # Distribute resumes across columns
    for i, resume in enumerate(resumes):
        col_idx = i % 3
        
        with cols[col_idx]:
            name = resume.get("name", "Unknown")
            email = resume.get("email", "")
            st.markdown(f"""
            <div class="resume-card">
                <div class="resume-name">{name}</div>
                <p>Email: {email}</p>
            </div>
            """, unsafe_allow_html=True)

# ── AGENT + MEMORY ─────────────────────────────────────────────────────
llm = ChatOpenAI(model=MODEL_NAME, api_key=OPENAI_API_KEY, temperature=0)
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful HR assistant. Use the `query_db` tool whenever the "
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

# ── STREAMLIT UI ───────────────────────────────────────────────────────
st.title("🧠 Resume Filtering Chatbot")

# Add debugging tools
with st.sidebar:
    st.header("Debug Tools")
    debug_mode = st.checkbox("Show Debug Info", value=True)
    if st.button("Clear Chat History"):
        st.session_state.memory.clear()
        st.rerun()

# Handle user input
user_input = st.chat_input("Ask me to find resumes...")
if user_input:
    # Display user message
    st.chat_message("user").write(user_input)
    
    # Process with agent
    with st.spinner("Thinking..."):
        try:
            # Invoke the agent
            response = st.session_state.agent_executor.invoke({"input": user_input})
            
            # Display the assistant's response
            st.chat_message("assistant").write(response["output"])
            
            # Debug information
            if debug_mode:
                st.subheader("🔍 Debug Information")
                
                # Show the raw response
                st.markdown("### Raw Response")
                st.json(response)
                
                # Try to extract the tool outputs 
                st.markdown("### Tool Outputs")
                try:
                    # Check if there are intermediate steps with tool outputs
                    if "intermediate_steps" in response:
                        for i, step in enumerate(response["intermediate_steps"]):
                            st.markdown(f"#### Step {i+1}")
                            # Tool call
                            action = step[0]
                            st.markdown(f"**Tool:** {action.tool}")
                            st.markdown(f"**Input:** {action.tool_input}")
                            
                            # Tool output
                            output = step[1]
                            st.markdown("**Output:**")
                            st.json(output)
                            
                            # If this is query_db and it has results, display them
                            if action.tool == "query_db" and isinstance(output, dict) and "results" in output:
                                st.markdown("#### Resume Results")
                                st.write(f"Found {len(output['results'])} resumes")
                                
                                # Display resumes in a simple grid
                                display_resume_grid(output["results"])
                    else:
                        st.write("No tool outputs found in the response")
                except Exception as e:
                    st.error(f"Error extracting tool outputs: {str(e)}")
                    
        except Exception as e:
            st.error(f"Error: {str(e)}")
            st.exception(e)
else:
    # Display chat history
    for msg in st.session_state.memory.chat_memory.messages:
        if msg.type == "human":
            st.chat_message("user").write(msg.content)
        else:
            st.chat_message("assistant").write(msg.content)
