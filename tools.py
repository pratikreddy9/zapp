import os
from typing import List, Optional, Dict, Any
import logging
import sys

import streamlit as st
import smtplib, ssl
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime
from langchain_core.tools import tool
from pymongo.errors import PyMongoError
import json

from utils import get_mongo_client, reformat_email_body
from variants import expand, COUNTRY_EQUIV, SKILL_VARIANTS, TITLE_VARIANTS

# Configure logging to output to both console and Streamlit
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Constants
SMTP_HOST, SMTP_PORT = "smtp.gmail.com", 465
SMTP_USER, SMTP_PASS = st.secrets["SMTP_USER"], st.secrets["SMTP_PASS"]
TOP_K_DEFAULT = 30
DB_NAME = "resumes_database"
COLL_NAME = "resumes"

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
    """
    Filter MongoDB resumes based on specified criteria.
    
    Args:
        query: The original user query string
        country: Optional country to filter resumes by
        min_experience_years: Optional minimum years of experience required
        max_experience_years: Optional maximum years of experience
        job_titles: Optional list of job titles to match
        skills: Optional list of skills to match
        top_k: Maximum number of results to return (default: 100)
        
    Returns:
        Dictionary containing matching resumes and metadata
    """
    try:
        # Build the MongoDB query with AND/OR logic as specified
        mongo_q: Dict[str, Any] = {}
        and_conditions = []
        
        # Country filter (if provided)
        if country:
            # Case-insensitive country search
            country_name = country.strip()
            country_variants = COUNTRY_EQUIV.get(country_name.lower(), [country_name])
            
            # Add both original case and lowercase versions of country names
            country_matches = []
            for variant in country_variants:
                # Add original variant
                country_matches.append(variant)
                # Add capitalized variant
                country_matches.append(variant.capitalize())
                # Add uppercase variant
                country_matches.append(variant.upper())
                # Add title case variant
                country_matches.append(variant.title())
            
            # Remove duplicates
            country_matches = list(set(country_matches))
            
            # Use $regex with 'i' option for case-insensitive matching
            mongo_q["country"] = {"$regex": f"^({'|'.join(country_matches)})$", "$options": "i"}
        
        # Skills filter - using AND logic between different skills, but OR between variants
        if skills and len(skills) > 0:
            skill_conditions = []
            for skill in skills:
                # Expand variants for this specific skill
                expanded = expand([skill], SKILL_VARIANTS)
                # Create OR condition between skill name and keywords for this skill and its variants
                skill_conditions.append({
                    "$or": [
                        {"skills.skillName": {"$in": expanded}},
                        {"keywords": {"$in": expanded}}
                    ]
                })
            # Add all skill conditions with AND logic
            and_conditions.extend(skill_conditions)
        
        # Job titles filter - using AND logic between different titles, but OR between variants
        if job_titles and len(job_titles) > 0:
            title_conditions = []
            for title in job_titles:
                # Expand variants for this specific title
                expanded = expand([title], TITLE_VARIANTS)
                # Create OR condition for this title and its variants
                title_conditions.append({
                    "jobExperiences.title": {"$in": expanded}
                })
            # Add all title conditions with AND logic
            and_conditions.extend(title_conditions)
        
        # Experience filter - using totalExperience field if available, otherwise sum job durations
        if min_experience_years is not None:
            # Convert to float to handle decimal values like 0.5 years of experience
            min_exp = float(min_experience_years)
            experience_condition = {
                "$or": [
                    # First try using the totalExperience field if it exists and is not null
                    {"totalExperience": {"$gte": min_exp}},
                    # Fallback: use $expr to calculate total from job durations 
                    {"$expr": {
                        "$gte": [
                            {"$sum": {
                                "$map": {
                                    "input": "$jobExperiences",
                                    "as": "job",
                                    "in": {
                                        "$convert": {
                                            "input": "$$job.duration",
                                            "to": "double",
                                            "onError": 0,
                                            "onNull": 0
                                        }
                                    }
                                }
                            }},
                            min_exp
                        ]
                    }}
                ]
            }
            and_conditions.append(experience_condition)
        
        if max_experience_years is not None:
            # Convert to float to handle decimal values
            max_exp = float(max_experience_years)
            experience_condition = {
                "$or": [
                    # First try using the totalExperience field if it exists and is not null
                    {"totalExperience": {"$lte": max_exp}},
                    # Fallback: use $expr to calculate total from job durations
                    {"$expr": {
                        "$lte": [
                            {"$sum": {
                                "$map": {
                                    "input": "$jobExperiences",
                                    "as": "job",
                                    "in": {
                                        "$convert": {
                                            "input": "$$job.duration",
                                            "to": "double",
                                            "onError": 0,
                                            "onNull": 0
                                        }
                                    }
                                }
                            }},
                            max_exp
                        ]
                    }}
                ]
            }
            and_conditions.append(experience_condition)
        
        # Add the AND conditions to the query if there are any
        if and_conditions:
            mongo_q["$and"] = and_conditions
        
        # Execute the MongoDB query
        with get_mongo_client() as client:
            coll = client[DB_NAME][COLL_NAME]
            
            # Use logging for console output
            query_str = json.dumps(mongo_q, indent=2)
            logger.info(f"MongoDB Query: {query_str}")
            
            # Store query in session state for persistent display
            if "mongo_queries" not in st.session_state:
                st.session_state.mongo_queries = []
            
            # Add timestamp to the query for reference
            timestamped_query = {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "query": query_str,
                "parameters": {
                    "country": country,
                    "min_experience_years": min_experience_years,
                    "max_experience_years": max_experience_years,
                    "job_titles": job_titles,
                    "skills": skills,
                    "top_k": top_k
                }
            }
            
            # Add to the beginning of the list (most recent first)
            st.session_state.mongo_queries.insert(0, timestamped_query)
            
            # Keep only the last 10 queries to avoid clutter
            if len(st.session_state.mongo_queries) > 10:
                st.session_state.mongo_queries = st.session_state.mongo_queries[:10]
            
            # Get the candidate pool from MongoDB using top_k to limit results
            candidates = list(coll.find(mongo_q, {"_id": 0, "embedding": 0}).limit(top_k))
        
        # Return the results
        return {
            "message": f"Found {len(candidates)} resumes matching the criteria.",
            "results_count": len(candidates),
            "results": candidates,
            "completed_at": datetime.utcnow().isoformat(),
        }
    except PyMongoError as err:
        return {"error": f"Database error: {str(err)}"}
    except Exception as exc:
        return {"error": f"Unexpected error: {str(exc)}"}


@tool
def send_email(to: str, subject: str, body: str) -> str:
    """
    Send a plain text email using SMTP credentials from secrets.
    
    Args:
        to: Recipient email address
        subject: Email subject line
        body: Plain text email body
        
    Returns:
        Success message or error description
    """
    try:
        msg = MIMEMultipart("alternative")
        msg["Subject"], msg["From"], msg["To"] = subject, SMTP_USER, to
        # Plain text email only
        msg.attach(MIMEText(body, "plain"))
        ctx = ssl.create_default_context()
        with smtplib.SMTP_SSL(SMTP_HOST, SMTP_PORT, context=ctx) as srv:
            srv.login(SMTP_USER, SMTP_PASS)
            srv.send_message(msg)
        return "Email sent successfully!"
    except Exception as e:
        return f"Email sending failed: {e}"


@tool
def get_job_match_counts(resume_ids: List[str]) -> Dict[str, Any]:
    """
    Given a list of resumeIds, return how many unique jobIds each resume is
    matched to in the resume_matches collection.
    
    Args:
        resume_ids: List of resume IDs to check
        
    Returns:
        Dictionary with job match counts for each resume
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
            "message": f"Job match counts fetched for {len(results)} resumeIds.",
            "results_count": len(results),
            "results": results,
            "completed_at": datetime.utcnow().isoformat(),
        }
    except PyMongoError as err:
        return {"error": f"Database error: {str(err)}"}
    except Exception as exc:
        return {"error": f"Unexpected error: {str(exc)}"}


@tool
def get_resume_id_by_name(name: str) -> Dict[str, Any]:
    """
    Given a candidate name, return their resumeId if it exists in our records.
    
    Args:
        name: The candidate's name (full or partial)
        
    Returns:
        Dictionary with resumeId if found
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
        return {"error": f"Error looking up resume ID: {str(e)}"}
