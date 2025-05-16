"""
ZappBot: Resume‑filtering chatbot with optimized display + email sender + job match counts
LangChain 0.3.25 • OpenAI 1.78.1 • Streamlit 1.34+
"""

# Email imports
import smtplib, ssl
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

import os, json, re, hashlib
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

# SMTP constants
SMTP_HOST, SMTP_PORT = "smtp.gmail.com", 465
SMTP_USER, SMTP_PASS = st.secrets["SMTP_USER"], st.secrets["SMTP_PASS"]

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

# ── UNIVERSAL EMAIL FORMATTER ──────────────────────────────────────────
def reformat_email_body(llm_output, intro="", conclusion=""):
    """
    Formats LLM output (list of dicts, dict, or string) as neat plain text for emails.
    - llm_output: string, dict, or list of dicts (parsed if possible)
    - intro, conclusion: optional strings to prepend/append
    """
    import json

    lines = []
    # Try parsing JSON if LLM gave a JSON string
    if isinstance(llm_output, str):
        llm_output = llm_output.strip()
        # Try to parse if JSON-like
        try:
            parsed = json.loads(llm_output)
            llm_output = parsed
        except Exception:
            pass

    if intro:
        lines.append(intro.strip() + "\n")

    # Handle list of dicts (resumes, counts, etc)
    if isinstance(llm_output, list) and llm_output and isinstance(llm_output[0], dict):
        for i, item in enumerate(llm_output, 1):
            lines.append(f"Candidate {i}")
            lines.append("-" * 40)
            # Format name with emphasis
            name = item.get("name", "Unknown")
            lines.append(f"{name}")
            lines.append("")
            
            # Basic information
            email = item.get("email", "N/A")
            phone = item.get("contactNo", "N/A")
            location = item.get("location", "N/A")
            lines.append(f"Email:       {email}")
            lines.append(f"Contact No:  {phone}")
            lines.append(f"Location:    {location}")
            
            # Experience
            experience = item.get("experience", [])
            if experience:
                lines.append("\nExperience:")
                for exp in experience:
                    lines.append(f"- {exp}")
            
            # Skills
            skills = item.get("skills", [])
            if skills:
                lines.append("\nSkills:")
                skill_text = ", ".join(str(s) for s in skills)
                lines.append(skill_text)
            
            # Keywords
            keywords = item.get("keywords", [])
            if keywords:
                lines.append("\nKeywords:")
                keyword_text = ", ".join(str(k) for k in keywords)
                lines.append(keyword_text)
            
            # Jobs matched
            job_matches = item.get("jobsMatched")
            if job_matches is not None:
                lines.append(f"\nMatched to {job_matches} jobs")
                
            # Resume ID in debug mode
            if debug_mode and item.get("resumeId"):
                lines.append(f"\nID: {item.get('resumeId')}")
                
            lines.append("")
    # Handle dict (summary data)
    elif isinstance(llm_output, dict):
        for k, v in llm_output.items():
            lines.append(f"{k.capitalize():<20}: {v}")
        lines.append("")
    # Handle string (or anything else)
    else:
        lines.append(str(llm_output).strip())
        lines.append("")

    if conclusion:
        lines.append(conclusion.strip())
    lines.append("\nSent by ZappBot")

    return "\n".join(lines)

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
    "javascript": ["javascript", "js", "java script"],
    "c#": ["c#", "c sharp", "csharp"],
    "html": ["html", "hypertext markup language"],
}
TITLE_VARIANTS = {
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
            {"role": "user", "content": f"Query: {query}\n\nResumes: {json.dumps(resumes)}"},
        ],
    )
    content = json.loads(chat.choices[0].message.content)
    return content.get("top_resume_ids", [])

# ── TOOLS ─────────────────────────────────────────────────────────────
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
        and_clauses = []
        
        # Country filter
        if country:
            mongo_q["country"] = {"$in": COUNTRY_EQUIV.get(country.strip().lower(), [country])}
        
        # Skills filter - use $all to ensure candidates have ALL requested skills
        if skills:
            expanded = expand(skills, SKILL_VARIANTS)
            
            # Modified to use $all instead of $in for strict matching
            skills_clause = {
                "$or": [
                    {"skills.skillName": {"$all": expanded}},
                    {"keywords": {"$all": expanded}}
                ]
            }
            and_clauses.append(skills_clause)
        
        # Job titles filter with $elemMatch for proper array element matching
        if job_titles:
            expanded_titles = expand(job_titles, TITLE_VARIANTS)
            and_clauses.append({"jobExperiences": {"$elemMatch": {"title": {"$in": expanded_titles}}}})
        
        # Experience years filter - modified to handle decimal values
        if isinstance(min_experience_years, int) and min_experience_years > 0:
            and_clauses.append(
                {
                    "$expr": {
                        "$gte": [
                            {
                                "$toDouble": {
                                    "$ifNull": [
                                        {"$first": "$jobExperiences.duration"}, 
                                        "0"
                                    ]
                                }
                            },
                            min_experience_years,
                        ]
                    }
                }
            )
        
        # Combine all AND clauses
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

@tool
def send_email(to: str, subject: str, body: str) -> str:
    """Send a plain text email using SMTP_USER / SMTP_PASS from secrets.toml."""
    try:
        msg = MIMEMultipart("alternative")
        msg["Subject"], msg["From"], msg["To"] = subject, SMTP_USER, to
        # Plain text email only
        msg.attach(MIMEText(body, "plain"))
        ctx = ssl.create_default_context()
        with smtplib.SMTP_SSL(SMTP_HOST, SMTP_PORT, context=ctx) as srv:
            srv.login(SMTP_USER, SMTP_PASS)
            srv.send_message(msg)
        return "Email sent!"
    except Exception as e:
        return f"Email failed: {e}"

@tool
def get_job_match_counts(resume_ids: List[str]) -> Dict[str, Any]:
    """
    Given a list of resumeIds, return how many unique jobIds each resume is
    matched to in the resume_matches collection.
    """
    try:
        if not isinstance(resume_ids, list):
            return {"error": "resume_ids must be a list of strings"}
        results = []
        with get_mongo_client() as client:
            coll = client[DB_NAME]["resume_matches"]
            for rid in resume_ids:
                doc = coll.find_one({"resumeId": rid}, {"_id": 0, "matches.jobId": 1})
                jobs = doc.get("matches", []) if doc else []
                results.append({"resumeId": rid, "jobsMatched": len(jobs)})
        return {
            "message": f"Counts fetched for {len(results)} resumeIds.",
            "results_count": len(results),
            "results": results,
            "completed_at": datetime.utcnow().isoformat(),
        }
    except PyMongoError as err:
        return {"error": f"DB error: {str(err)}"}
    except Exception as exc:
        return {"error": str(exc)}

@tool
def get_resume_id_by_name(name: str) -> Dict[str, Any]:
    """
    Given a candidate name, return their resumeId if it exists in our records.
    """
    try:
        if "resume_ids" not in st.session_state:
            return {"error": "No resume IDs are stored in the current session."}
        
        # Normalize name by lowercasing and removing extra spaces
        name_norm = ' '.join(name.lower().split())
        
        # Try exact match first
        if name_norm in [k.lower() for k in st.session_state.resume_ids.keys()]:
            for k, v in st.session_state.resume_ids.items():
                if k.lower() == name_norm:
                    return {
                        "found": True, 
                        "name": k, 
                        "resumeId": v
                    }
        
        # Try partial match
        for k, v in st.session_state.resume_ids.items():
            if name_norm in k.lower():
                return {
                    "found": True, 
                    "name": k, 
                    "resumeId": v
                }
        
        # If no match found in session state, try database lookup
        with get_mongo_client() as client:
            coll = client[DB_NAME][COLL_NAME]
            # Try to find by name
            query = {"$or": [
                {"name": {"$regex": name, "$options": "i"}},
                {"fullName": {"$regex": name, "$options": "i"}}
            ]}
            doc = coll.find_one(query, {"_id": 0, "resumeId": 1, "name": 1, "fullName": 1})
            if doc and doc.get("resumeId"):
                display_name = doc.get("name") or doc.get("fullName") or name
                return {
                    "found": True,
                    "name": display_name,
                    "resumeId": doc["resumeId"]
                }
        
        return {"found": False, "message": f"No resumeId found for '{name}'"}
    except Exception as e:
        return {"error": str(e)}

# ── EXTRACT SEARCH SKILLS FROM QUERY ───────────────────────────────────
def extract_search_skills_from_query(query):
    """Extract skills mentioned in the search query."""
    # Patterns that might indicate skills in a query
    skill_patterns = [
        r'with skills? in ([\w\s,]+)',
        r'who knows? ([\w\s,]+)',
        r'experience with ([\w\s,]+)',
        r'proficient in ([\w\s,]+)',
        r'expertise in ([\w\s,]+)',
        r'familiar with ([\w\s,]+)',
        r'skilled in ([\w\s,]+)',
    ]
    
    for pattern in skill_patterns:
        match = re.search(pattern, query, re.IGNORECASE)
        if match:
            # Extract and split the skills
            skills_text = match.group(1)
            skills = [s.strip() for s in skills_text.split(',')]
            # Check for "and" in the last item
            if ' and ' in skills[-1]:
                last_skills = skills[-1].split(' and ')
                skills = skills[:-1] + [s.strip() for s in last_skills]
            return skills
    
    # If no pattern match, look for known skills in the query
    all_known_skills = set()
    for variants in SKILL_VARIANTS.values():
        all_known_skills.update(variants)
    
    found_skills = []
    words = query.lower().split()
    for word in words:
        if word in all_known_skills:
            found_skills.append(word)
    
    return found_skills

# ── PARSE AND PROCESS RESPONSE ────────────────────────────────────────
def extract_resume_ids_from_response(response_text):
    """Extract resumeIds from the HTML comment in the response."""
    meta_pattern = r'<!--RESUME_META:(.*?)-->'
    meta_match = re.search(meta_pattern, response_text)
    if meta_match:
        try:
            meta_data = json.loads(meta_match.group(1))
            return {item.get("name"): item.get("resumeId") for item in meta_data if item.get("resumeId")}
        except:
            return {}
    return {}

def process_response(text):
    """
    Process the response text to:
    1. Extract any introductory text
    2. Extract resume data
    3. Remove the resume list from the text to avoid redundancy
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
        resume_pattern = r'([A-Z][a-z]+ (?:[A-Z][a-z]+ )?(?:[A-Z][a-z]+)?)\s*\n\s*Email:\s*([^\n]+)\s*\nContact No:\s*([^\n]+)\s*\nLocation:\s*([^\n]+)\s*\nExperience:\s*([^\n]+)\s*\nSkills:\s*([^\n]+)(?:\s*\nKeywords:\s*([^\n]+))?'
        matches = re.findall(resume_pattern, text, re.MULTILINE | re.IGNORECASE)
        
        # If that didn't work, try the numbered format
        if not matches:
            resume_pattern = r'\d+\.\s+\*\*([^*]+)\*\*\s*\n\s*-\s+\*\*Email:\*\*\s+([^\n]+)\s*\n\s*-\s+\*\*Contact No:\*\*\s+([^\n]+)\s*\n\s*-\s+\*\*Location:\*\*\s+([^\n]+)\s*\n\s*-\s+\*\*Experience:\*\*\s+([^\n]+)\s*\n\s*-\s+\*\*Skills:\*\*\s+([^\n]+)(?:\s*\n\s*-\s+\*\*Keywords:\*\*\s+([^\n]+))?'
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
            if len(match) >= 7:  # If keywords are included
                name, email, contact, location, experience, skills, keywords = match
                keyword_list = [k.strip() for k in keywords.split(',')] if keywords else []
            else:  # If no keywords are included
                name, email, contact, location, experience, skills = match
                keyword_list = []
            
            # Split skills and experience
            skill_list = [s.strip() for s in skills.split(',')]
            exp_list = [e.strip() for e in experience.split(',')]
            
            resumes.append({
                "name": name.strip(),
                "email": email.strip(),
                "contactNo": contact.strip(),
                "location": location.strip(),
                "experience": exp_list,
                "skills": skill_list,
                "keywords": keyword_list
            })
        
        return {
            "is_resume_response": True,
            "intro_text": intro_text,
            "resumes": resumes,
            "conclusion_text": conclusion_text,
            "full_text": text  # Keep this for debug purposes
        }
    else:
        # Not a resume listing response
        return {
            "is_resume_response": False,
            "full_text": text
        }

# ── HELPER: attach missing resumeIds without showing them ──────────────
def attach_hidden_resume_ids(resume_list: List[Dict[str, Any]]) -> None:
    """
    For every resume in resume_list that lacks a 'resumeId', look it up by (email, contactNo)
    and add it. Also retrieves and attaches keywords from MongoDB if not present.
    """
    if not resume_list:
        return
    
    with get_mongo_client() as client:
        coll = client[DB_NAME][COLL_NAME]
        for res in resume_list:
            email = res.get("email")
            phone = res.get("contactNo")
            
            # Skip if we already have both resumeId and keywords
            if "resumeId" in res and res["resumeId"] and "keywords" in res and res["keywords"]:
                continue
                
            query = {}
            if email and phone:
                query = {"email": email, "contactNo": phone}
            elif "resumeId" in res and res["resumeId"]:
                query = {"resumeId": res["resumeId"]}
            else:
                continue
                
            doc = coll.find_one(
                query,
                {"_id": 0, "resumeId": 1, "keywords": 1},
            )
            
            if doc:
                # Add resumeId if it's missing
                if doc.get("resumeId") and "resumeId" not in res:
                    res["resumeId"] = doc["resumeId"]
                
                # Add keywords if they're missing or empty
                if doc.get("keywords") and (not res.get("keywords") or len(res["keywords"]) == 0):
                    res["keywords"] = doc["keywords"]

# ── DISPLAY RESUME GRID ───────────────────────────────────────────────
def display_resume_grid(resumes, container=None, search_skills=None):
    """
    Display resumes in a 3x3 grid layout with styled cards.
    
    Args:
        resumes: List of resume dictionaries to display
        container: Optional Streamlit container to render into
        search_skills: List of skills from the original search query to highlight
    """
    target = container if container else st
    
    if not resumes:
        target.warning("No resumes found matching the criteria.")
        return
    
    # Normalize search skills for case-insensitive comparison
    search_skills_lower = []
    if search_skills:
        # Flatten and normalize search skills
        search_skills_lower = [s.lower() for s in search_skills]
        # Add expansions from variants
        for skill in list(search_skills_lower):  # Create a copy to avoid modifying during iteration
            expanded = [s.lower() for s in SKILL_VARIANTS.get(skill.lower(), [])]
            search_skills_lower.extend(expanded)
        search_skills_lower = list(set(search_skills_lower))  # Remove duplicates
    
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
    .keyword-tag {
        display: inline-block;
        background-color: #fff8e1;
        color: #f57c00;
        border-radius: 12px;
        padding: 3px 10px;
        margin: 3px;
        font-size: 12px;
        font-weight: 500;
    }
    .matching-skill {
        background-color: #e3f2fd;
        color: #0d47a1;
        border: 1px solid #bbdefb;
    }
    .matching-keyword {
        background-color: #fff3e0;
        color: #e65100;
        border: 1px solid #ffe0b2;
    }
    .job-matches {
        margin-top: 8px;
        padding: 4px 10px;
        background-color: #E3F2FD;
        border-radius: 4px;
        display: inline-block;
        font-size: 14px;
        color: #0D47A1;
    }
    .resume-id {
        font-size: 10px;
        color: #6a737d;
        margin-top: 8px;
        word-break: break-all;
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
                resume_id = resume.get("resumeId", "")  # Extract resumeId for job matching
                
                # Get experience and skills
                experience = resume.get("experience", [])
                skills = resume.get("skills", [])
                keywords = resume.get("keywords", [])
                
                # Get job matches if available
                job_matches = resume.get("jobsMatched")
                
                with cols[col]:
                    html = f"""
                    <div class="resume-card">
                        <div class="resume-name">{name}</div>
                        <div class="resume-location">📍 {location}</div>
                        <div class="resume-contact">📧 {email}</div>
                        <div class="resume-contact">📱 {phone}</div>
                    """
                    
                    # Add resumeId as data attribute (hidden but accessible)
                    if resume_id:
                        html = html.replace('<div class="resume-card">', f'<div class="resume-card" data-resume-id="{resume_id}">')
                    
                    # Add job matches if available
                    if job_matches is not None:
                        html += f'<div class="job-matches">🔗 Matched to {job_matches} jobs</div>'
                    
                    # Add experience section
                    if experience:
                        html += f'<div class="resume-section-title">Experience</div>'
                        for exp in experience[:3]:  # Limit to 3 experiences
                            html += f'<div class="resume-experience">• {exp}</div>'
                    
                    # Add skills section
                    if skills:
                        html += f'<div class="resume-section-title">Skills</div><div>'
                        for skill in skills[:7]:  # Limit to 7 skills
                            skill_str = str(skill)
                            if isinstance(skill, dict) and "skillName" in skill:
                                # Handle skills objects with skillName field
                                skill_str = skill["skillName"]
                            
                            is_matching = search_skills_lower and skill_str.lower() in search_skills_lower
                            matching_class = " matching-skill" if is_matching else ""
                            html += f'<span class="skill-tag{matching_class}">{skill_str}</span>'
                        html += '</div>'
                    
                    # Add keywords section
                    if keywords:
                        html += f'<div class="resume-section-title">Keywords</div><div>'
                        for keyword in keywords[:7]:  # Limit to 7 keywords
                            keyword_str = str(keyword)
                            is_matching = search_skills_lower and keyword_str.lower() in search_skills_lower
                            matching_class = " matching-keyword" if is_matching else ""
                            html += f'<span class="keyword-tag{matching_class}">{keyword_str}</span>'
                        html += '</div>'
                    
                    # Show resume ID in debug mode
                    if debug_mode and resume_id:
                        html += f'<div class="resume-id">ID: {resume_id}</div>'
                    
                    html += '</div>'
                    st.markdown(html, unsafe_allow_html=True)

# ── AGENT + MEMORY ─────────────────────────────────────────────────────
llm = ChatOpenAI(model=MODEL_NAME, api_key=OPENAI_API_KEY, temperature=0)

# Updated prompt that instructs the agent to use resumeIds properly
agent_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are a helpful HR assistant named ZappBot.

# Resume Filtering
When searching for resumes, always use the `query_db` tool with all relevant parameters:
- `query`: The full user's search query
- `country`: The country filter if specified
- `min_experience_years`: Years of experience (minimum) if mentioned
- `max_experience_years`: Years of experience (maximum) if mentioned
- `job_titles`: A list of job titles to search for
- `skills`: A list of all skills mentioned in the query
- `top_k`: Default is 50

Always extract both skills and keywords from the database. Make sure when showing the search results, you include both the 'skills' and 'keywords' fields in your response.

# Resume Formatting
When displaying resume results, always format them consistently as follows:

First, provide a brief introduction line like:
"Here are some developers in [location] with [criteria]:"

Then, list each candidate in this exact format:

[Full Name]

Email: [email]
Contact No: [phone]
Location: [location]
Experience: [experience1], [experience2], [experience3]
Skills: [skill1], [skill2], [skill3], [skill4]
Keywords: [keyword1], [keyword2], [keyword3], [keyword4]

Maintain this precise format with consistent spacing and no bullet points or numbering, as it allows our UI to extract and display the resumes in a grid layout.

After listing all candidates, include a brief concluding sentence like:
"These candidates have diverse experiences and skills that may suit your needs."

- **Never join multiple candidates or items on one line, and never use commas or paragraphs to join candidates.**
- **Always keep each candidate in the exact block and field order above, with a blank line between candidates.**

# ResumeIDs and Tools

When a user asks about a specific candidate by name, use the `get_resume_id_by_name` tool to look up their resumeId. Then use this resumeId with the `get_job_match_counts` tool to find how many jobs they are matched to.

If the user asks to email or send these results, call the `send_email` tool.

If the user wants to check how many jobs a resume is matched to, use the `get_job_match_counts` tool with the appropriate resumeIds.
            """,
        ),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
)

# Initialize session state variables
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(
        memory_key="chat_history", return_messages=True
    )

if "resume_ids" not in st.session_state:
    st.session_state.resume_ids = {}

if "processed_responses" not in st.session_state:
    st.session_state.processed_responses = {}

if "job_match_data" not in st.session_state:
    st.session_state.job_match_data = {}

# Initialize or upgrade the agent
tools = [query_db, send_email, get_job_match_counts, get_resume_id_by_name]
if "agent_executor" not in st.session_state:
    agent = create_openai_tools_agent(llm, tools, agent_prompt)
    st.session_state.agent_executor = AgentExecutor(
        agent=agent, 
        tools=tools,
        memory=st.session_state.memory, 
        verbose=True
    )
    st.session_state.agent_upgraded = True
elif not st.session_state.get("agent_upgraded", False):
    upgraded_agent = create_openai_tools_agent(llm, tools, agent_prompt)
    st.session_state.agent_executor = AgentExecutor(
        agent=upgraded_agent,
        tools=tools,
        memory=st.session_state.memory,
        verbose=True,
    )
    st.session_state.agent_upgraded = True

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
    .tool-section {
        background-color: #f8f9fa;
        padding: 15px;
        border-radius: 8px;
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<div class="header-container"><div class="header-emoji">⚡</div><div class="header-text">ZappBot</div></div>', unsafe_allow_html=True)

# Sidebar with settings
with st.sidebar:
    st.header("Settings")
    debug_mode = st.checkbox("Debug Mode", value=False)
    
    # Email settings section
    st.subheader("Email Settings")
    default_recipient = st.text_input("Default Email Recipient", 
                                     placeholder="recipient@example.com",
                                     help="Default email to use when sending resume results")
    
    # Job matching tool section
    st.subheader("Job Matching")
    st.markdown("""
    To check job matches, ask about a specific candidate:
    ```
    How many jobs is [Candidate Name] matched to?
    ```
    """)
    
    if st.button("Clear Chat History"):
        st.session_state.memory.clear()
        st.session_state.processed_responses = {}
        st.session_state.job_match_data = {}
        st.session_state.resume_ids = {}
        st.rerun()

# Main chat container
chat_container = st.container()

# Handle user input
user_input = st.chat_input("Ask me to find resumes...")
if user_input:
    # Process with agent
    with st.spinner("Thinking..."):
        try:
            # Invoke the agent
            response = st.session_state.agent_executor.invoke({"input": user_input})
            response_text = response["output"]
            
            # Extract and store resumeIds from the response
            resume_ids = extract_resume_ids_from_response(response_text)
            if resume_ids:
                st.session_state.resume_ids.update(resume_ids)
            
            # Process the response
            processed = process_response(response_text)
            
            # Check if this contains job match data
            if "jobsMatched" in response_text:
                try:
                    # Try to extract job match data
                    matches_pattern = r'"results":\s*(\[.*?\])'
                    matches_match = re.search(matches_pattern, response_text)
                    if matches_match:
                        match_data = json.loads(matches_match.group(1))
                        # Store job match data
                        for item in match_data:
                            resume_id = item.get("resumeId")
                            if resume_id:
                                st.session_state.job_match_data[resume_id] = item.get("jobsMatched", 0)
                except:
                    pass  # Silently fail if we can't parse the job match data
            
            # Generate a unique key for this message
            timestamp = datetime.now().isoformat()
            message_key = f"user_{timestamp}"
            st.session_state.processed_responses[message_key] = processed
            
            # Force a refresh to show the new messages
            st.rerun()
            
        except Exception as e:
            st.error(f"Error: {str(e)}")
            if debug_mode:
                st.exception(e)


# Display the complete chat history
with chat_container:
    # Create a list to store all resume responses for display in the order they appear
    resume_responses = []
    
    # Display all messages
    for i, msg in enumerate(st.session_state.memory.chat_memory.messages):
        if msg.type == "human":
            st.chat_message("user").write(msg.content)
            
            # Store the user query for context if the next message is a resume response
            if i+1 < len(st.session_state.memory.chat_memory.messages):
                next_msg = st.session_state.memory.chat_memory.messages[i+1]
                if next_msg.type == "ai":
                    # Generate a key for the AI message
                    ai_msg_key = f"ai_{i+1}"
                    
                    # Ensure the message is processed
                    if ai_msg_key not in st.session_state.processed_responses:
                        st.session_state.processed_responses[ai_msg_key] = process_response(next_msg.content)
                    
                    # Get the processed message
                    processed_ai = st.session_state.processed_responses[ai_msg_key]
                    
                    # If this is a resume response, store it for later display
                    if processed_ai["is_resume_response"]:
                        resume_responses.append({
                            "query": msg.content,
                            "processed": processed_ai,
                            "index": i+1
                        })
                        
        else:  # AI message
            # Get or process the AI message
            msg_key = f"ai_{i}"
            if msg_key not in st.session_state.processed_responses:
                st.session_state.processed_responses[msg_key] = process_response(msg.content)
            
            processed = st.session_state.processed_responses[msg_key]
            
            # Display the message
            ai_message = st.chat_message("assistant")
            if processed["is_resume_response"]:
                # Extract and store resumeIds if they are in the message
                resume_ids = extract_resume_ids_from_response(processed["full_text"])
                if resume_ids:
                    st.session_state.resume_ids.update(resume_ids)
                
                hidden_meta = json.dumps([{"name": r.get("name"), "resumeId": r.get("resumeId", "")}for r in processed["resumes"]])
                # Just show the intro text in the chat message
                for item in json.loads(hidden_meta):
                    if item.get("name") and item.get("resumeId"):
                        st.session_state.resume_ids[item["name"]] = item["resumeId"]
                        
                ai_message.write(processed["intro_text"])
                
                # If there's a conclusion, add it 
                if processed.get("conclusion_text"):
                    ai_message.write(processed["conclusion_text"])
            else:
                # For non-resume responses, show the full text
                ai_message.write(processed["full_text"])
    
    # Display all resume grids after the chat
    if resume_responses:
        st.markdown("---")
        st.subheader("Resume Search Results")
        
        # Create an expander for each resume search
        for i, resp in enumerate(resume_responses):
            with st.expander(f"Search {i+1}: {resp['query']}", expanded=(i == len(resume_responses)-1)):
                st.markdown(f"<div class='resume-query'>{resp['processed']['intro_text']}</div>", unsafe_allow_html=True)
                
                # Make sure resumes have resumeIds
                attach_hidden_resume_ids(resp['processed']['resumes'])
                
                # Store resumeIds in session state
                for resume in resp['processed']['resumes']:
                    if resume.get("resumeId") and resume.get("name"):
                        st.session_state.resume_ids[resume["name"]] = resume["resumeId"]
                
                # Add job match data to resumes if available
                if st.session_state.job_match_data:
                    for resume in resp['processed']['resumes']:
                        resume_id = resume.get("resumeId")
                        if resume_id and resume_id in st.session_state.job_match_data:
                            resume["jobsMatched"] = st.session_state.job_match_data[resume_id]
                
                # Extract any skills from the original query to highlight matching skills
                search_skills = extract_search_skills_from_query(resp['query'])
                
                # Display the resume grid with highlighted search skills
                display_resume_grid(resp['processed']['resumes'], search_skills=search_skills)
                
                # Add a row with email button and job match button
                cols = st.columns([2, 1, 1])
                
                # Email button
                with cols[1]:
                    if resp['processed']['resumes']:
                        if st.button(f"📧 Email Results", key=f"email_btn_{i}"):
                            try:
                                # Universal formatted plain text for email (uses LLM/chat output)
                                plain_text_body = reformat_email_body(
                                    llm_output=resp['processed']['resumes'],
                                    intro=resp['processed']['intro_text'],
                                    conclusion=resp['processed'].get('conclusion_text', '')
                                )
                                
                                # Get recipient email
                                recipient = default_recipient
                                if not recipient:
                                    st.error("Please set a default email recipient in the sidebar.")
                                else:
                                    # Send the email
                                    result = send_email(
                                        to=recipient,
                                        subject=f"ZappBot Results: {resp['query']}",
                                        body=plain_text_body
                                    )
                                    st.success(f"Email sent to {recipient}")
                            except Exception as e:
                                st.error(f"Failed to send email: {str(e)}")
                
                # Job Match button
                with cols[2]:
                    if resp['processed']['resumes']:
                        if st.button("🔍 Match Jobs", key=f"job_btn_{i}"):
                            try:
                                # Extract resume IDs
                                resume_ids = []
                                for resume in resp['processed']['resumes']:
                                    resume_id = resume.get("resumeId")
                                    if resume_id:
                                        resume_ids.append(resume_id)
                                
                                if resume_ids:
                                    # Call get_job_match_counts
                                    result = get_job_match_counts(resume_ids)
                                    if "results" in result:
                                        # Store job match data
                                        for item in result["results"]:
                                            resume_id = item.get("resumeId")
                                            if resume_id:
                                                st.session_state.job_match_data[resume_id] = item.get("jobsMatched", 0)
                                        st.success(f"Job match data updated for {len(resume_ids)} resumes")
                                        st.rerun()
                                    else:
                                        st.error("Failed to get job match data")
                                else:
                                    st.warning("No resume IDs found")
                            except Exception as e:
                                st.error(f"Failed to get job matches: {str(e)}")
                
                # Display conclusion if available
                if resp['processed'].get('conclusion_text'):
                    st.write(resp['processed']['conclusion_text'])
    
    # Show debug info if enabled
    if debug_mode:
        with st.expander("Debug Information"):
            st.subheader("Memory Contents")
            st.json({i: msg.content for i, msg in enumerate(st.session_state.memory.chat_memory.messages)})
            
            st.subheader("Stored Resume IDs")
            st.json(st.session_state.resume_ids)
            
            st.subheader("Processed Responses")
            for key, value in st.session_state.processed_responses.items():
                if "full_text" in value:
                    # Create a shorter version for display
                    shorter_value = {k: v for k, v in value.items() if k != "full_text"}
                    shorter_value["full_text_length"] = len(value["full_text"])
                    st.json({key: shorter_value})
                else:
                    st.json({key: value})
            
            st.subheader("Job Match Data")
            st.json(st.session_state.job_match_data)
