import streamlit as st
import os, json, re
from pymongo import MongoClient
from typing import List, Dict, Any, Optional

# Configuration
st.set_page_config(page_title="Resume Search Tester", layout="wide")

# MongoDB connection settings
# You can replace these with your actual credentials or use secrets
MONGO_CFG = {
    "host": "notify.pesuacademy.com",
    "port": 27017,
    "username": "admin",
    "password": st.secrets["MONGO_PASS"] if "MONGO_PASS" in st.secrets else "",
    "authSource": "admin",
}
DB_NAME = "resumes_database"
COLL_NAME = "resumes"
TOP_K_DEFAULT = 50

# --- Variant dictionaries (from your original code) ---
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

# --- Helper Functions ---
def get_mongo_client() -> MongoClient:
    """Create and return a MongoDB client."""
    return MongoClient(**MONGO_CFG)

def expand(values: List[str], table: Dict[str, List[str]]) -> List[str]:
    """Expand terms using the variant dictionaries."""
    out = set()
    for v in values:
        if not v or not isinstance(v, str):
            continue
        v_low = v.strip().lower()
        out.update(table.get(v_low, []))
        out.add(v)
    return list(out)

def search_resumes(
    query_text: str,
    country: Optional[str] = None,
    min_experience_years: Optional[int] = None,
    job_titles: Optional[List[str]] = None,
    skills: Optional[List[str]] = None,
    search_method: str = "basic",
    top_k: int = TOP_K_DEFAULT,
) -> Dict[str, Any]:
    """
    Search for resumes in MongoDB using different search methods.
    
    Args:
        query_text: Raw query text (for reference only)
        country: Country filter
        min_experience_years: Minimum years of experience
        job_titles: List of job titles to search for
        skills: List of skills to search for
        search_method: Which search method to use
        top_k: Maximum number of results to return
        
    Returns:
        Dictionary with search results and metadata
    """
    try:
        # Connect to MongoDB
        mongo_q = {}
        
        # Choose search method
        if search_method == "basic":
            # Basic search with OR logic
            or_conditions = []
            
            # Job title matching
            if job_titles and len(job_titles) > 0:
                expanded_titles = expand(job_titles, TITLE_VARIANTS)
                or_conditions.append(
                    {"jobExperiences.title": {"$in": expanded_titles}}
                )
            
            # Skills matching
            if skills and len(skills) > 0:
                expanded_skills = expand(skills, SKILL_VARIANTS)
                skill_condition = {
                    "$or": [
                        {"skills.skillName": {"$in": expanded_skills}},
                        {"keywords": {"$in": expanded_skills}}
                    ]
                }
                or_conditions.append(skill_condition)
            
            # Country matching
            if country:
                country_values = COUNTRY_EQUIV.get(country.strip().lower(), [country])
                or_conditions.append(
                    {"country": {"$in": country_values}}
                )
            
            # Combine with OR logic
            if or_conditions:
                mongo_q["$or"] = or_conditions
                
        elif search_method == "regex":
            # Regex-based search with OR logic
            or_conditions = []
            
            # Job title matching with regex
            if job_titles and len(job_titles) > 0:
                title_conditions = []
                for title in job_titles:
                    if title and isinstance(title, str):
                        title_low = title.strip().lower()
                        # Add direct title variants
                        expanded = TITLE_VARIANTS.get(title_low, [title])
                        title_conditions.append({
                            "jobExperiences.title": {"$in": expanded}
                        })
                        # Add regex pattern
                        title_conditions.append({
                            "jobExperiences.title": {
                                "$regex": f"\\b{re.escape(title_low)}\\b",
                                "$options": "i"
                            }
                        })
                if title_conditions:
                    or_conditions.append({"$or": title_conditions})
            
            # Skills matching with regex
            if skills and len(skills) > 0:
                skill_conditions = []
                for skill in skills:
                    if skill and isinstance(skill, str):
                        skill_low = skill.strip().lower()
                        # Add direct skill variants
                        expanded = SKILL_VARIANTS.get(skill_low, [skill])
                        skill_conditions.append({
                            "skills.skillName": {"$in": expanded}
                        })
                        skill_conditions.append({
                            "keywords": {"$in": expanded}
                        })
                        # Add regex patterns
                        skill_conditions.append({
                            "skills.skillName": {
                                "$regex": f"\\b{re.escape(skill_low)}\\b",
                                "$options": "i"
                            }
                        })
                        skill_conditions.append({
                            "keywords": {
                                "$regex": f"\\b{re.escape(skill_low)}\\b",
                                "$options": "i"
                            }
                        })
                if skill_conditions:
                    or_conditions.append({"$or": skill_conditions})
            
            # Country matching with regex
            if country:
                country_values = COUNTRY_EQUIV.get(country.strip().lower(), [country])
                country_conditions = []
                country_conditions.append({
                    "country": {"$in": country_values}
                })
                country_conditions.append({
                    "country": {
                        "$regex": f"^{re.escape(country)}$",
                        "$options": "i"
                    }
                })
                or_conditions.append({"$or": country_conditions})
            
            # Combine with OR logic
            if or_conditions:
                mongo_q["$or"] = or_conditions
                
        elif search_method == "strict":
            # Strict search with AND logic
            and_conditions = []
            
            # Country filter
            if country:
                country_values = COUNTRY_EQUIV.get(country.strip().lower(), [country])
                and_conditions.append({
                    "country": {"$in": country_values}
                })
            
            # Job title filter
            if job_titles and len(job_titles) > 0:
                expanded_titles = expand(job_titles, TITLE_VARIANTS)
                and_conditions.append({
                    "jobExperiences.title": {"$in": expanded_titles}
                })
            
            # Skills filter - require ALL skills
            if skills and len(skills) > 0:
                expanded_skills = expand(skills, SKILL_VARIANTS)
                for skill in expanded_skills:
                    and_conditions.append({
                        "$or": [
                            {"skills.skillName": skill},
                            {"keywords": skill}
                        ]
                    })
            
            # Experience filter
            if isinstance(min_experience_years, int) and min_experience_years > 0:
                and_conditions.append({
                    "$expr": {
                        "$gte": [
                            {"$sum": {
                                "$map": {
                                    "input": {"$ifNull": ["$jobExperiences", []]},
                                    "as": "job",
                                    "in": {
                                        "$convert": {
                                            "input": {"$ifNull": ["$$job.duration", "0"]},
                                            "to": "double",
                                            "onError": 0,
                                            "onNull": 0
                                        }
                                    }
                                }
                            }},
                            min_experience_years
                        ]
                    }
                })
            
            # Combine with AND logic
            if and_conditions:
                mongo_q["$and"] = and_conditions
        
        # Execute the query
        with get_mongo_client() as client:
            coll = client[DB_NAME][COLL_NAME]
            
            # Find candidates matching the criteria
            candidates = list(coll.find(mongo_q, {"_id": 0}).limit(top_k))
            
            # Get a sample document structure if available
            sample_structure = None
            if candidates:
                sample = candidates[0]
                sample_structure = {
                    "fields": list(sample.keys()),
                    "jobExperiences": sample.get("jobExperiences", [])[:1] if "jobExperiences" in sample else None,
                    "skills": sample.get("skills", [])[:3] if "skills" in sample else None,
                    "keywords": sample.get("keywords", [])[:3] if "keywords" in sample else None,
                    "country": sample.get("country", None)
                }
        
        # Return the results
        return {
            "query": mongo_q,
            "count": len(candidates),
            "candidates": candidates,
            "sample_structure": sample_structure
        }
    
    except Exception as e:
        import traceback
        return {
            "error": str(e),
            "traceback": traceback.format_exc(),
            "query": mongo_q if 'mongo_q' in locals() else None
        }

# --- Streamlit UI ---
st.title("Resume Search Tester")
st.write("This app lets you test the resume search functionality and see the raw MongoDB results.")

# Search form
with st.form("search_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        query_text = st.text_input("Raw Query Text (for reference only)", 
                                   "Find software developer in Indonesia with SQL and Python skills")
        country = st.text_input("Country", "Indonesia")
        min_exp = st.number_input("Minimum Experience (years)", min_value=0, value=0, step=1)
    
    with col2:
        job_titles_input = st.text_input("Job Titles (comma-separated)", "software developer")
        job_titles = [t.strip() for t in job_titles_input.split(",")] if job_titles_input else []
        
        skills_input = st.text_input("Skills (comma-separated)", "SQL, Python")
        skills = [s.strip() for s in skills_input.split(",")] if skills_input else []
        
        search_method = st.selectbox(
            "Search Method", 
            ["basic", "regex", "strict"],
            format_func=lambda x: {
                "basic": "Basic (OR logic, exact matches)",
                "regex": "Regex (OR logic, flexible matching)",
                "strict": "Strict (AND logic, exact matches)"
            }.get(x, x)
        )
    
    submit_button = st.form_submit_button("Search Resumes")

# Execute search when form is submitted
if submit_button:
    with st.spinner("Searching..."):
        results = search_resumes(
            query_text=query_text,
            country=country,
            min_experience_years=min_exp,
            job_titles=job_titles,
            skills=skills,
            search_method=search_method
        )
    
    # Display results
    if "error" in results:
        st.error(f"Error: {results['error']}")
        st.code(results["traceback"])
    else:
        st.success(f"Found {results['count']} matching resumes")
        
        # Show the MongoDB query
        with st.expander("MongoDB Query", expanded=True):
            st.code(json.dumps(results["query"], indent=2))
        
        # Show sample document structure
        if results["sample_structure"]:
            with st.expander("Sample Document Structure", expanded=True):
                st.json(results["sample_structure"])
        
        # Show the raw results
        with st.expander(f"Raw Results ({results['count']} records)", expanded=False):
            st.json(results["candidates"])
        
        # Display candidates in a more readable format
        st.subheader("Matching Candidates")
        
        for i, candidate in enumerate(results["candidates"]):
            with st.expander(f"Candidate {i+1}: {candidate.get('name', 'Unknown')}"):
                # Basic information
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"**Name:** {candidate.get('name', 'N/A')}")
                    st.write(f"**Email:** {candidate.get('email', 'N/A')}")
                    st.write(f"**Contact:** {candidate.get('contactNo', 'N/A')}")
                    st.write(f"**Country:** {candidate.get('country', 'N/A')}")
                    st.write(f"**ResumeId:** {candidate.get('resumeId', 'N/A')}")
                
                # Job experience
                st.write("**Job Experience:**")
                job_exp = candidate.get("jobExperiences", [])
                if job_exp:
                    for job in job_exp:
                        st.write(f"- {job.get('title', 'N/A')} ({job.get('duration', 'N/A')} years)")
                else:
                    st.write("- No job experience found")
                
                # Skills
                st.write("**Skills:**")
                skills_list = candidate.get("skills", [])
                if skills_list:
                    skill_text = ""
                    for skill in skills_list:
                        if isinstance(skill, dict):
                            skill_text += f"- {skill.get('skillName', 'N/A')}\n"
                        else:
                            skill_text += f"- {skill}\n"
                    st.text(skill_text)
                else:
                    st.write("- No skills found")
                
                # Keywords
                st.write("**Keywords:**")
                keywords = candidate.get("keywords", [])
                if keywords:
                    st.write(", ".join(keywords))
                else:
                    st.write("- No keywords found")
