import streamlit as st
import json
from typing import List, Dict, Any, Optional
from pymongo import MongoClient

# Configuration
st.set_page_config(page_title="MongoDB Resume Viewer", layout="wide")

# MongoDB connection settings
MONGO_CFG = {
    "host": "notify.pesuacademy.com",
    "port": 27017,
    "username": "admin",
    "password": st.secrets["MONGO_PASS"],
    "authSource": "admin",
}
DB_NAME = "resumes_database"
COLL_NAME = "resumes"
TOP_K_DEFAULT = 20

# Connect to MongoDB
def get_mongo_client() -> MongoClient:
    return MongoClient(**MONGO_CFG)

# Direct query to MongoDB with basic filtering options
def get_candidate_profiles(filters: Dict[str, Any] = None, top_k: int = TOP_K_DEFAULT) -> List[Dict[str, Any]]:
    """
    Get candidate profiles from MongoDB with basic filtering.
    """
    try:
        with get_mongo_client() as client:
            coll = client[DB_NAME][COLL_NAME]
            
            # Build the MongoDB query
            mongo_query = {}
            
            # Apply filters if provided
            if filters:
                # Country filter
                if filters.get("country"):
                    mongo_query["country"] = {"$regex": filters["country"], "$options": "i"}
                
                # Job title filter
                if filters.get("job_title"):
                    mongo_query["jobExperiences.title"] = {"$regex": filters["job_title"], "$options": "i"}
                
                # Skill filter
                if filters.get("skill"):
                    mongo_query["$or"] = [
                        {"skills.skillName": {"$regex": filters["skill"], "$options": "i"}},
                        {"keywords": {"$regex": filters["skill"], "$options": "i"}}
                    ]
                
                # Min experience filter (if available as a field)
                if filters.get("min_experience") and filters.get("min_experience") > 0:
                    mongo_query["totalExperience"] = {"$gte": filters["min_experience"]}
            
            # Execute the query
            candidates = list(coll.find(mongo_query, {
                "_id": 0, 
                "embedding": 0  # Exclude embedding field
            }).limit(top_k))
            
            return candidates
    except Exception as e:
        st.error(f"Database error: {str(e)}")
        return []

# Display candidates in a neat format
def display_candidate_profiles(candidates: List[Dict[str, Any]]):
    """
    Display candidate profiles in a clean format.
    """
    if not candidates:
        st.warning("No matching candidates found.")
        return
    
    st.success(f"Found {len(candidates)} matching candidates")
    
    # Custom CSS for better display
    st.markdown("""
    <style>
    .resume-card {
        border: 1px solid #e1e4e8;
        border-radius: 10px;
        padding: 16px;
        margin-bottom: 15px;
        background-color: white;
        box-shadow: 0 3px 8px rgba(0,0,0,0.05);
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
    
    # Display each candidate in a card
    for candidate in candidates:
        with st.expander(f"{candidate.get('name', 'Unknown')} - {candidate.get('country', 'Unknown')}"):
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.markdown(f"**ResumeID:** {candidate.get('resumeId', 'N/A')}")
                st.markdown(f"**Email:** {candidate.get('email', 'N/A')}")
                st.markdown(f"**Phone:** {candidate.get('contactNo', 'N/A')}")
                st.markdown(f"**Location:** {candidate.get('country', 'N/A')}")
                
                # Calculate total experience
                experiences = candidate.get("jobExperiences", [])
                total_exp = 0
                for exp in experiences:
                    duration = exp.get("duration", "0")
                    if duration is not None:
                        # Convert duration to float if possible
                        try:
                            if isinstance(duration, str) and (duration.isdigit() or duration.replace(".", "").isdigit()):
                                total_exp += float(duration)
                            elif isinstance(duration, (int, float)):
                                total_exp += float(duration)
                        except (ValueError, TypeError):
                            # Skip invalid durations
                            pass
                st.markdown(f"**Total Experience:** {total_exp} years")
            
            with col2:
                # Job experiences
                st.markdown("### Job Experiences")
                for job in candidate.get("jobExperiences", []):
                    if job.get("title"):
                        company = job.get("companyName", "N/A")
                        duration = job.get("duration", "N/A")
                        st.markdown(f"- **{job.get('title')}** at {company} ({duration} years)")
                
                # Skills
                st.markdown("### Skills")
                skills_html = ""
                skills = candidate.get("skills", [])
                for skill in skills:
                    if isinstance(skill, dict) and "skillName" in skill:
                        skills_html += f'<span class="skill-tag">{skill["skillName"]}</span>'
                
                st.markdown(skills_html, unsafe_allow_html=True)
                
                # Keywords
                keywords = candidate.get("keywords", [])
                if keywords:
                    st.markdown("### Keywords")
                    keywords_html = ""
                    for keyword in keywords:
                        keywords_html += f'<span class="skill-tag">{keyword}</span>'
                    st.markdown(keywords_html, unsafe_allow_html=True)
                    
            # Add raw data tab
            with st.expander("Raw Data"):
                # Remove embedding to save space
                display_data = {k: v for k, v in candidate.items() if k != 'embedding'}
                st.json(display_data)

# Main application
def main():
    st.title("MongoDB Resume Viewer")
    st.write("View raw resume data directly from MongoDB")
    
    # Sidebar filters
    st.sidebar.header("Search Filters")
    country = st.sidebar.text_input("Country", "Indonesia")
    job_title = st.sidebar.text_input("Job Title", "software developer")
    skill = st.sidebar.text_input("Skill", "SQL")
    min_experience = st.sidebar.number_input("Minimum Experience (years)", min_value=0, value=3)
    top_k = st.sidebar.number_input("Number of results", min_value=1, max_value=50, value=20)
    
    # Create filter dict
    filters = {
        "country": country,
        "job_title": job_title,
        "skill": skill,
        "min_experience": min_experience
    }
    
    # Search button
    if st.sidebar.button("Search"):
        with st.spinner("Searching for candidates..."):
            # Get candidates from MongoDB with filters
            candidates = get_candidate_profiles(filters, top_k=top_k)
            
            # Display the raw query being sent for debugging
            st.sidebar.subheader("MongoDB Query")
            mongo_query = {}
            if country:
                mongo_query["country"] = {"$regex": country, "$options": "i"}
            if job_title:
                mongo_query["jobExperiences.title"] = {"$regex": job_title, "$options": "i"}
            if skill:
                mongo_query["$or"] = [
                    {"skills.skillName": {"$regex": skill, "$options": "i"}},
                    {"keywords": {"$regex": skill, "$options": "i"}}
                ]
            if min_experience > 0:
                mongo_query["totalExperience"] = {"$gte": min_experience}
            
            st.sidebar.code(json.dumps(mongo_query, indent=2))
            
            # Display results
            display_candidate_profiles(candidates)
    
    # Initial load without filters
    else:
        st.info("Use the sidebar filters to search for candidates, then click 'Search'.")

if __name__ == "__main__":
    main()
