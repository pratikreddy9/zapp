"""
ZappBot: Resume-filtering chatbot with optimized display + email sender + job match counts
LangChain 0.3.25 • OpenAI 1.78.1 • Streamlit 1.34+
"""

# ────────────────────────── IMPORTS ────────────────────────────────────
import smtplib, ssl
from email.mime.text import MIMEText, MIMEMultipart
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

# ────────────────────────── CONFIG ─────────────────────────────────────
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]

SMTP_HOST, SMTP_PORT = "smtp.gmail.com", 465
SMTP_USER, SMTP_PASS = st.secrets["SMTP_USER"], st.secrets["SMTP_PASS"]

MONGO_CFG = {
    "host": "notify.pesuacademy.com",
    "port": 27017,
    "username": "admin",
    "password": st.secrets["MONGO_PASS"],
    "authSource": "admin",
}

MODEL_NAME        = "gpt-4o"
EVAL_MODEL_NAME   = "gpt-4o"
TOP_K_DEFAULT     = 50
DB_NAME, COLL_NAME = "resumes_database", "resumes"

# ─────────────────── EMAIL BODY FORMATTER ─────────────────────────────
def reformat_email_body(llm_output, intro: str = "", conclusion: str = "") -> str:
    """
    Turn any LLM output (string | dict | list[dict]) into a neat plain-text email.
    """
    lines = []

    if isinstance(llm_output, str):
        llm_output = llm_output.strip()
        try:
            llm_output = json.loads(llm_output)
        except Exception:
            pass  # not JSON

    if intro:
        lines.append(intro.strip() + "\n")

    if isinstance(llm_output, list) and llm_output and isinstance(llm_output[0], dict):
        for i, item in enumerate(llm_output, 1):
            lines.append(f"Item {i}")
            lines.append("-" * 30)
            for k, v in item.items():
                lines.append(f"{k.capitalize():<15}: {v}")
            lines.append("")
    elif isinstance(llm_output, dict):
        for k, v in llm_output.items():
            lines.append(f"{k.capitalize():<20}: {v}")
        lines.append("")
    else:
        lines.append(str(llm_output).strip())
        lines.append("")

    if conclusion:
        lines.append(conclusion.strip())
    lines.append("\nSent by ZappBot")
    return "\n".join(lines)

# ─────────────────────── MONGO HELPER ─────────────────────────────────
def get_mongo_client() -> MongoClient:
    return MongoClient(**MONGO_CFG)

# ────────────────────── NORMALISATION MAPS ────────────────────────────
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
        "software developer", "software dev", "softwaredeveloper", "software engineer"
    ],
    "backend developer": [
        "backend developer", "backend dev", "back-end developer", "server-side developer"
    ],
    "frontend developer": [
        "frontend developer", "frontend dev", "front-end developer"
    ],
}

def expand(values: List[str], table: Dict[str, List[str]]) -> List[str]:
    out = set()
    for v in values:
        v_low = v.strip().lower()
        out.update(table.get(v_low, []))
        out.add(v)
    return list(out)

# ───────────── LLM-BASED RESUME SCORER (TOP-10) ───────────────────────
_openai_client = openai.OpenAI(api_key=OPENAI_API_KEY)
EVALUATOR_PROMPT = """
You are a resume scoring assistant. Return only the 10 best resumeIds.

JSON format:
{ "top_resume_ids": [...], "completed_at": "ISO" }
"""

def score_resumes(query: str, resumes: List[Dict[str, Any]]) -> List[str]:
    chat = _openai_client.chat.completions.create(
        model           = EVAL_MODEL_NAME,
        response_format = {"type": "json_object"},
        messages = [
            {"role": "system", "content": EVALUATOR_PROMPT},
            {"role": "user", "content": f"Query: {query}\n\nResumes:\n{json.dumps(resumes)}"},
        ],
    )
    content = json.loads(chat.choices[0].message.content)
    return content.get("top_resume_ids", [])

# ─────────────────────────── TOOL: query_db ───────────────────────────
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
            # ANY requested skill / keyword passes
            mongo_q["$or"] = [
                {"skills.skillName": {"$in": expanded}},
                {"keywords":        {"$in": expanded}},
            ]

        and_clauses = []
        if job_titles:
            and_clauses.append({"jobExperiences.title": {"$in": expand(job_titles, TITLE_VARIANTS)}})

        if isinstance(min_experience_years, int) and min_experience_years > 0:
            and_clauses.append({
                "$expr": {
                    "$gte": [
                        {"$toInt": {"$ifNull": [{"$first": "$jobExperiences.duration"}, "0"]}},
                        min_experience_years,
                    ]
                }
            })

        if and_clauses:
            mongo_q["$and"] = and_clauses

        # 🔍 DEBUG – show final Mongo filter
        if st.session_state.get("debug_mode"):
            st.write("Mongo filter →", mongo_q)

        with get_mongo_client() as client:
            coll = client[DB_NAME][COLL_NAME]
            candidates = list(coll.find(mongo_q, {"_id": 0, "embedding": 0}).limit(top_k))

        best_ids     = score_resumes(query, candidates)
        best_resumes = [r for r in candidates if r["resumeId"] in best_ids]

        # 🔍 DEBUG – how many survivors?
        if st.session_state.get("debug_mode"):
            st.write(f"Fetched {len(candidates)} docs, scorer kept {len(best_resumes)}")

        return {
            "message"      : f"{len(best_resumes)} resumes after scoring.",
            "results_count": len(best_resumes),
            "results"      : best_resumes,
            "completed_at" : datetime.utcnow().isoformat(),
        }

    except PyMongoError as err:
        return {"error": f"DB error: {str(err)}"}
    except Exception as exc:
        return {"error": str(exc)}

# ───────────────────── TOOL: send_email ───────────────────────────────
@tool
def send_email(to: str, subject: str, body: str) -> str:
    """Send a plain-text email."""
    try:
        msg = MIMEMultipart("alternative")
        msg["Subject"], msg["From"], msg["To"] = subject, SMTP_USER, to
        msg.attach(MIMEText(body, "plain"))
        ctx = ssl.create_default_context()
        with smtplib.SMTP_SSL(SMTP_HOST, SMTP_PORT, context=ctx) as srv:
            srv.login(SMTP_USER, SMTP_PASS)
            srv.send_message(msg)
        return "Email sent!"
    except Exception as e:
        return f"Email failed: {e}"

# ───────────── TOOL: get_job_match_counts ─────────────────────────────
@tool
def get_job_match_counts(resume_ids: List[str]) -> Dict[str, Any]:
    """Return how many unique jobIds each resumeId is matched to."""
    try:
        if not isinstance(resume_ids, list):
            return {"error": "resume_ids must be a list of strings"}

        results = []
        with get_mongo_client() as client:
            coll = client[DB_NAME]["resume_matches"]
            for rid in resume_ids:
                doc  = coll.find_one({"resumeId": rid}, {"_id": 0, "matches.jobId": 1})
                jobs = doc.get("matches", []) if doc else []
                results.append({"resumeId": rid, "jobsMatched": len(jobs)})

        return {
            "message"      : f"Counts fetched for {len(results)} resumeIds.",
            "results_count": len(results),
            "results"      : results,
            "completed_at" : datetime.utcnow().isoformat(),
        }
    except PyMongoError as err:
        return {"error": f"DB error: {str(err)}"}
    except Exception as exc:
        return {"error": str(exc)}

# ───────────── TOOL: get_resume_id_by_name ────────────────────────────
@tool
def get_resume_id_by_name(name: str) -> Dict[str, Any]:
    """Lookup a resumeId by candidate name."""
    try:
        if "resume_ids" not in st.session_state:
            return {"error": "No resume IDs stored."}

        name_norm = " ".join(name.lower().split())

        # exact / partial match in cached dict
        for cached_name, rid in st.session_state.resume_ids.items():
            if name_norm == cached_name.lower() or name_norm in cached_name.lower():
                return {"found": True, "name": cached_name, "resumeId": rid}

        # fallback DB search
        with get_mongo_client() as client:
            coll = client[DB_NAME][COLL_NAME]
            q    = {"$or": [{"name": {"$regex": name, "$options": "i"}},
                            {"fullName": {"$regex": name, "$options": "i"}}]}
            doc  = coll.find_one(q, {"_id": 0, "resumeId": 1, "name": 1, "fullName": 1})
            if doc and doc.get("resumeId"):
                return {"found": True,
                        "name" : doc.get("name") or doc.get("fullName") or name,
                        "resumeId": doc["resumeId"]}
        return {"found": False, "message": f"No resumeId for '{name}'"}
    except Exception as e:
        return {"error": str(e)}

# ──────────────── PARSE & PROCESS CHAT RESPONSE ───────────────────────
def extract_resume_ids_from_response(text: str) -> Dict[str, str]:
    match = re.search(r'<!--RESUME_META:(.*?)-->', text)
    if match:
        try:
            meta = json.loads(match.group(1))
            return {item["name"]: item["resumeId"] for item in meta if item.get("resumeId")}
        except Exception:
            return {}
    return {}

def process_response(text: str) -> Dict[str, Any]:
    if "Here are some" in text and re.search(r'\bSkills?:', text, re.I):
        intro_match = re.search(r'^(.*?)\n\n([A-Z][^\n]+)\n\nEmail:', text, re.S)
        intro_text  = intro_match.group(1).strip() if intro_match else ""

        resume_pattern = (
            r'([A-Z][A-Za-z ]+?)\s*\n\s*Email:\s*([^\n]+)\s*\nContact No:\s*([^\n]+)\s*'
            r'\nLocation:\s*([^\n]+)\s*\nExperience:\s*([^\n]+)\s*\nSkills:\s*([^\n]+)'
        )
        matches = re.findall(resume_pattern, text, re.I)

        concl_match   = re.search(r'(These candidates.*?)\s*$', text, re.S)
        conclusion    = concl_match.group(1).strip() if concl_match else ""

        resumes = []
        for m in matches:
            name,email,phone,loc,exp,skills = m
            resumes.append({
                "name"      : name.strip(),
                "email"     : email.strip(),
                "contactNo" : phone.strip(),
                "location"  : loc.strip(),
                "experience": [e.strip() for e in exp.split(',')],
                "skills"    : [s.strip() for s in skills.split(',')],
            })

        return {
            "is_resume_response": True,
            "intro_text"  : intro_text,
            "resumes"     : resumes,
            "conclusion_text": conclusion,
            "full_text"   : text,
        }
    return {"is_resume_response": False, "full_text": text}

# ──────────────── PATCH MISSING resumeIds / keywords ──────────────────
def attach_hidden_resume_ids(resume_list: List[Dict[str, Any]]) -> None:
    if not resume_list:
        return
    patched = []
    with get_mongo_client() as client:
        coll = client[DB_NAME][COLL_NAME]
        for res in resume_list:
            if res.get("resumeId"):
                if not res.get("keywords"):
                    kw_doc = coll.find_one({"resumeId": res["resumeId"]}, {"_id": 0, "keywords": 1})
                    if kw_doc and kw_doc.get("keywords"):
                        res["keywords"] = kw_doc["keywords"]
                continue
            doc = coll.find_one(
                {"email": res.get("email"), "contactNo": res.get("contactNo")},
                {"_id": 0, "resumeId": 1, "keywords": 1},
            )
            if doc and doc.get("resumeId"):
                res["resumeId"] = doc["resumeId"]
                if doc.get("keywords"):
                    res["keywords"] = doc["keywords"]
                patched.append(res)
    if st.session_state.get("debug_mode") and patched:
        st.write("Patched resumeIds for:", [r["email"] for r in patched])

# ──────────────────── DISPLAY RESUME GRID ─────────────────────────────
def display_resume_grid(resumes: List[Dict[str, Any]], query_text: str = "", container=None):
    target = container or st
    if not resumes:
        target.warning("No resumes found.")
        return

    # tokens → include synonyms
    base_tokens = set(re.findall(r"[A-Za-z0-9#+\-\.]+", query_text.lower()))
    query_tokens = set(base_tokens)
    for tok in base_tokens:
        query_tokens.update(SKILL_VARIANTS.get(tok, []))

    if st.session_state.get("debug_mode"):
        st.write("Highlight tokens:", sorted(query_tokens))

    # ---------- CSS ----------
    target.markdown("""
    <style>
    .resume-card{border:1px solid #e1e4e8;border-radius:10px;padding:16px;margin-bottom:15px;background:#fff;
                 box-shadow:0 3px 8px rgba(0,0,0,0.05);transition:.2s}
    .resume-card:hover{transform:translateY(-3px);box-shadow:0 5px 15px rgba(0,0,0,0.1)}
    .resume-name{font-weight:bold;font-size:18px;margin-bottom:8px;color:#24292e}
    .resume-location{color:#586069;font-size:14px;margin-bottom:10px}
    .resume-contact{font-size:14px;color:#444d56;margin-bottom:8px}
    .resume-section-title{font-weight:600;margin:12px 0 6px;font-size:15px;color:#24292e}
    .resume-experience{font-size:14px;color:#444d56;margin-bottom:4px}
    .skill-tag{display:inline-block;background:#f1f8ff;color:#0366d6;border-radius:12px;padding:3px 10px;margin:3px;
               font-size:12px;font-weight:500}
    .skill-match{background:#FFE0B2!important;color:#D84315!important}
    .job-matches{margin-top:8px;padding:4px 10px;background:#E3F2FD;border-radius:4px;display:inline-block;
                 font-size:14px;color:#0D47A1}
    .resume-id{font-size:10px;color:#6a737d;margin-top:8px;word-break:break-all}
    </style>
    """, unsafe_allow_html=True)

    rows = (len(resumes) + 2) // 3
    for r in range(rows):
        cols = target.columns(3)
        for c in range(3):
            idx = r*3 + c
            if idx >= len(resumes): continue
            res = resumes[idx]

            name      = res.get("name","Unknown")
            email     = res.get("email","")
            phone     = res.get("contactNo","")
            loc       = res.get("location","")
            resume_id = res.get("resumeId","")
            exp       = res.get("experience",[])
            skills    = res.get("skills",[])
            keywords  = res.get("keywords",[])
            combined  = []
            seen      = set()
            for lst in (skills, keywords):
                for s in lst:
                    sl=s.lower()
                    if sl not in seen:
                        combined.append(s.strip())
                        seen.add(sl)
            job_cnt = res.get("jobsMatched")

            html = f"<div class='resume-card' {('data-resume-id='+resume_id) if resume_id else ''}>"
            html+= f"<div class='resume-name'>{name}</div>"
            html+= f"<div class='resume-location'>📍 {loc}</div>"
            html+= f"<div class='resume-contact'>📧 {email}</div>"
            html+= f"<div class='resume-contact'>📱 {phone}</div>"

            if job_cnt is not None:
                html+= f"<div class='job-matches'>🔗 Matched to {job_cnt} jobs</div>"

            if exp:
                html+= "<div class='resume-section-title'>Experience</div>"
                for e in exp[:3]:
                    html+= f"<div class='resume-experience'>• {e}</div>"

            if combined:
                html+= "<div class='resume-section-title'>Skills & Keywords</div><div>"
                for sk in combined[:12]:
                    cls = " skill-match" if sk.lower() in query_tokens else ""
                    html+= f"<span class='skill-tag{cls}'>{sk}</span>"
                html+= "</div>"

            if st.session_state.get("debug_mode") and resume_id:
                html+= f"<div class='resume-id'>ID: {resume_id}</div>"
            html+= "</div>"
            st.markdown(html, unsafe_allow_html=True)

# ──────────────────────── AGENT & MEMORY ──────────────────────────────
llm = ChatOpenAI(model=MODEL_NAME, api_key=OPENAI_API_KEY, temperature=0)

agent_prompt = ChatPromptTemplate.from_messages([
    ("system", """
You are a helpful HR assistant named ZappBot.

When listing resumes, follow *exactly* this block format:
[Full Name]

Email: …
Contact No: …
Location: …
Experience: …
Skills: …

(One blank line between candidates.)

Use get_resume_id_by_name → get_job_match_counts when asked,
and send_email when the user wants results emailed.
"""),
    MessagesPlaceholder("chat_history"),
    ("user","{input}"),
    MessagesPlaceholder("agent_scratchpad"),
])

# ──────────────── STREAMLIT SESSION STATE INIT ───────────────────────
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(
        memory_key="chat_history", return_messages=True)

for key, default in [
    ("resume_ids",        {}),
    ("processed_responses", {}),
    ("job_match_data",    {}),
]:
    st.session_state.setdefault(key, default)

tools = [query_db, send_email, get_job_match_counts, get_resume_id_by_name]
if "agent_executor" not in st.session_state:
    st.session_state.agent_executor = AgentExecutor(
        agent=create_openai_tools_agent(llm, tools, agent_prompt),
        tools=tools,
        memory=st.session_state.memory,
        verbose=True,
    )

# ─────────────────────── STREAMLIT UI TOP ────────────────────────────
st.set_page_config(page_title="ZappBot", layout="wide")

# global debug flag (used by other functions)
with st.sidebar:
    st.header("Settings")
    st.session_state["debug_mode"] = st.checkbox("Debug Mode", value=False)
    default_recipient = st.text_input("Default Email Recipient", "")
    st.markdown("**Job-match tip:** _How many jobs is [Name] matched to?_")

    if st.button("Clear Chat History"):
        for k in ("memory","processed_responses","job_match_data","resume_ids"):
            if k in st.session_state: del st.session_state[k]
        st.experimental_rerun()

st.markdown("<h2>⚡ ZappBot</h2>", unsafe_allow_html=True)
chat_container = st.container()

# ─────────────────────── HANDLE USER INPUT ───────────────────────────
user_input = st.chat_input("Ask me to find resumes...")
if user_input:
    with st.spinner("Thinking..."):
        try:
            out = st.session_state.agent_executor.invoke({"input": user_input})
            response_text = out["output"]

            st.session_state.resume_ids.update(
                extract_resume_ids_from_response(response_text))

            processed = process_response(response_text)
            st.session_state.processed_responses[
                f"user_{datetime.now().isoformat()}"] = processed
            st.experimental_rerun()
        except Exception as e:
            st.error(str(e))
            if st.session_state["debug_mode"]:
                st.exception(e)

# ───────────────────────── RENDER CHAT + GRIDS ───────────────────────
resume_responses = []
for i, msg in enumerate(st.session_state.memory.chat_memory.messages):
    if msg.type == "human":
        st.chat_message("user").write(msg.content)
        if (i+1 < len(st.session_state.memory.chat_memory.messages) and
            st.session_state.memory.chat_memory.messages[i+1].type=="ai"):
            key = f"ai_{i+1}"
            if key not in st.session_state.processed_responses:
                st.session_state.processed_responses[key] = process_response(
                    st.session_state.memory.chat_memory.messages[i+1].content)
            if st.session_state.processed_responses[key]["is_resume_response"]:
                resume_responses.append((msg.content,
                    st.session_state.processed_responses[key]))
    else:
        key = f"ai_{i}"
        if key not in st.session_state.processed_responses:
            st.session_state.processed_responses[key] = process_response(msg.content)
        proc = st.session_state.processed_responses[key]
        ai_block = st.chat_message("assistant")
        if proc["is_resume_response"]:
            st.session_state.resume_ids.update(
                extract_resume_ids_from_response(proc["full_text"]))
            ai_block.write(proc["intro_text"])
            if proc.get("conclusion_text"):
                ai_block.write(proc["conclusion_text"])
        else:
            ai_block.write(proc["full_text"])

# ───────────────────── RENDER RESUME RESULT GRIDS ────────────────────
if resume_responses:
    st.markdown("---")
    st.subheader("Resume Search Results")
    for idx, (query_txt, proc) in enumerate(resume_responses, 1):
        with st.expander(f"Search {idx}: {query_txt}",
                         expanded=(idx==len(resume_responses))):
            st.markdown(f"<div class='resume-query'>{proc['intro_text']}</div>",
                        unsafe_allow_html=True)

            attach_hidden_resume_ids(proc["resumes"])  # IDs + keywords

            for r in proc["resumes"]:
                if r.get("resumeId") and r.get("name"):
                    st.session_state.resume_ids[r["name"]] = r["resumeId"]

            # inject job-match counts if cached
            for r in proc["resumes"]:
                rid = r.get("resumeId")
                if rid in st.session_state.job_match_data:
                    r["jobsMatched"] = st.session_state.job_match_data[rid]

            display_resume_grid(proc["resumes"], query_text=query_txt)

            # buttons
            cols = st.columns([2,1,1])
            with cols[1]:
                if st.button("📧 Email Results", key=f"email_{idx}"):
                    if not default_recipient:
                        st.error("Set default recipient in sidebar.")
                    else:
                        body = reformat_email_body(proc["resumes"],
                                                   intro=proc["intro_text"],
                                                   conclusion=proc.get("conclusion_text",""))
                        st.write(send_email(default_recipient,
                                            f"ZappBot Results – {query_txt}", body))
            with cols[2]:
                if st.button("🔍 Match Jobs", key=f"match_{idx}"):
                    ids = [r["resumeId"] for r in proc["resumes"] if r.get("resumeId")]
                    if ids:
                        result = get_job_match_counts(ids)
                        if "results" in result:
                            for item in result["results"]:
                                st.session_state.job_match_data[item["resumeId"]] = item["jobsMatched"]
                            st.success("Job-match data updated."); st.experimental_rerun()
                        else:
                            st.error("Failed to fetch job-match data.")

            if proc.get("conclusion_text"):
                st.write(proc["conclusion_text"])

# ───────────────── DEBUG PANE ─────────────────────────────────────────
if st.session_state["debug_mode"]:
    with st.expander("🔧 Debug Information"):
        st.json({"Memory": [m.content for m in st.session_state.memory.chat_memory.messages]})
        st.json({"Stored Resume IDs": st.session_state.resume_ids})
        st.json({"Job Match Data": st.session_state.job_match_data})
        st.json({"Processed Keys": list(st.session_state.processed_responses.keys())})
