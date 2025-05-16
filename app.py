import streamlit as st
import json, re
from typing import List, Dict, Any, Optional
from pymongo import MongoClient
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import ResponseSchema, StructuredOutputParser

# Configuration
st.set_page_config(page_title="ZappBot Resume Search", layout="wide")

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
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
MODEL_NAME = "gpt-4o"
TOP_K_DEFAULT = 100  # Increased to get more candidates for LLM to consider

# Connect to MongoDB
def get_mongo_client() -> MongoClient:
    return MongoClient(**MONGO_CFG)

# Get candidate profiles with minimal filtering
def get_candidate_profiles(query_text: str, country: Optional[str] = None, skills: Optional[List[str]] = None, top_k: int = TOP_K_DEFAULT) -> List[Dict[str, Any]]:
    """
    Get candidate profiles from MongoDB with very minimal filtering.
    """
    try:
        with get_mongo_client() as client:
            coll = client[DB_NAME][COLL_NAME]
            
            # Extract key search terms from the query using regex
            country_match = re.search(r'in\s+(\w+)', query_text)
            if country_match and not country:
                country = country_match.group(1)
            
            skills_match = re.findall(r'(?:with|and)\s+(\w+)(?:\s+and\s+(\w+))?', query_text)
            extracted_skills = []
            for match in skills_match:
                extracted_skills.extend([s for s in match if s])
            
            if not skills and extracted_skills:
                skills = extracted_skills
            
            # Build a simple query based on country or skills if available
            mongo_q = {}
            
            if country:
                # Case insensitive country search
                mongo_q["country"] = {"$regex": country, "$options": "i"}
            
            # If query is empty, return a sample of profiles rather than nothing
            candidates = list(coll.find(mongo_q, {
                "_id": 0, 
                "embedding": 0  # Exclude embedding field
            }).limit(top_k))
            
            return candidates
    except Exception as e:
        st.error(f"Database error: {str(e)}")
        return []

# Use LangChain to score candidates based on the query
def score_candidates_with_langchain(query: str, candidates: List[Dict[str, Any]], top_k: int = 10) -> List[Dict[str, Any]]:
    """
    Score candidates using LangChain and return the top matches, with awareness of data quality issues.
    """
    try:
        if not candidates:
            return []
        
        # Initialize LangChain components
        llm = ChatOpenAI(model=MODEL_NAME, api_key=OPENAI_API_KEY, temperature=0)
        
        # Define output schema for structured parsing
        response_schemas = [
            ResponseSchema(name="top_candidates", 
                          description=f"List of resumeIds for the best matches to the query, up to {top_k}"),
            ResponseSchema(name="reasoning", 
                          description="Brief explanation of how you evaluated the candidates")
        ]
        output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
        format_instructions = output_parser.get_format_instructions()
        
        # Create prompt template with awareness of data quality issues
        template = """
        You are a sophisticated HR recruiter with expertise in matching candidates to job requirements.
        
        QUERY: {query}
        
        CANDIDATES:
        {candidates}
        
        IMPORTANT DATA QUALITY ISSUES:
        1. Job titles are inconsistent (e.g., "software developer", "software engineer", "developer" can all mean the same role)
        2. Skills may be in either the "skills" or "keywords" arrays or both
        3. Experience data may be missing or inconsistent
        4. Some fields may be null or empty
        
        Given these data issues, use your judgment to find the best matches. Consider:
        
        1. Similar job titles: Match "software developer" with related titles like "software engineer", "web developer", "full stack developer", etc.
        2. Relevant skills: Look for required skills or closely related alternatives in both skills and keywords
        3. Experience: Consider total years of experience AND relevance of experience
        4. Location: Match location if specified
        
        SELECT THE BEST CANDIDATES:
        - Choose up to {top_k} candidates that best match the query requirements
        - It's better to include good partial matches than to return very few or no results
        - If experience years are specified, treat it as "at least X years" in relevant roles
        - For skills, consider functional equivalents (e.g., "MySQL" ~ "SQL" ~ "PostgreSQL" for database skills)
        
        {format_instructions}
        """
        
        prompt = ChatPromptTemplate.from_template(template)
        
        # Prepare candidate data with enough context for the LLM
        candidate_data = []
        for c in candidates:
            # Get all job experiences with duration to help the LLM understand experience level
            job_experiences = []
            for job in c.get("jobExperiences", []):
                job_title = job.get("title", "")
                duration = job.get("duration", "")
                company = job.get("companyName", "")
                if job_title or duration or company:
                    job_experiences.append({
                        "title": job_title,
                        "duration": duration,
                        "company": company
                    })
            
            # Get all skills and keywords
            skills = []
            if "skills" in c:
                for skill in c.get("skills", []):
                    if isinstance(skill, dict) and "skillName" in skill:
                        skills.append(skill.get("skillName"))
                    elif isinstance(skill, str):
                        skills.append(skill)
            
            # Create a candidate entry with enhanced context
            candidate_entry = {
                "resumeId": c.get("resumeId", ""),
                "name": c.get("name", "Unknown"),
                "country": c.get("country", ""),
                "totalExperience": c.get("totalExperience", sum(float(job.get("duration", 0)) for job in c.get("jobExperiences", []) 
                                                             if job.get("duration") and str(job.get("duration", "")).replace(".", "").isdigit())),
                "jobExperiences": job_experiences,
                "skills": skills,
                "keywords": c.get("keywords", [])
            }
            candidate_data.append(candidate_entry)
        
        # Invoke LLM
        chain = prompt | llm
        response = chain.invoke({
            "query": query,
            "candidates": json.dumps(candidate_data, indent=2),
            "top_k": top_k,
            "format_instructions": format_instructions
        })
        
        # Parse response
        parsed_output = output_parser.parse(response.content)
        top_ids = parsed_output.get("top_candidates", [])
        reasoning = parsed_output.get("reasoning", "")
        
        # Extract top candidates
        top_candidates = [c for c in candidates if c.get("resumeId") in top_ids]
        
        # Add reasoning to the first candidate as metadata
        if top_candidates:
            top_candidates[0]["_matching_explanation"] = reasoning
        
        return top_candidates
        
    except Exception as e:
        st.error(f"LangChain scoring error: {str(e)}")
        import traceback
        st.error(traceback.format_exc())
        return []

# Display candidates in a neat format
def display_candidate_profiles(candidates: List[Dict[str, Any]]):
    """
    Display candidate profiles in a clean format.
    """
    if not candidates:
        st.warning("No matching candidates found.")
        return
    
    # Extract and display matching explanation if available
    if candidates and "_matching_explanation" in candidates[0]:
        with st.expander("Matching Logic Explanation", expanded=False):
            st.write(candidates[0].get("_matching_explanation", ""))
    
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
    .highlight {
        background-color: #fff8c5;
        padding: 2px 4px;
        border-radius: 3px;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Extract query terms for highlighting
    query_terms = set()
    if "query_text" in st.session_state:
        query_terms = set(re.findall(r'\b\w+\b', st.session_state.query_text.lower()))
    
    # Display each candidate in an expander
    for i, candidate in enumerate(candidates):
        with st.expander(f"{i+1}. {candidate.get('name', 'Unknown')} - {candidate.get('country', 'Unknown')}", expanded=i==0):
            col1, col2 = st.columns([1, 2])
            
            with col1:
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
                
                # Education
                if candidate.get("educationalQualifications"):
                    st.markdown("**Education:**")
                    for edu in candidate.get("educationalQualifications", []):
                        degree = edu.get("degree", "")
                        field = edu.get("field", "")
                        year = edu.get("graduationYear", "")
                        institution = edu.get("institution", "")
                        if any([degree, field, year, institution]):
                            st.markdown(f"- {degree or ''} {field or ''} ({year or 'N/A'}) - {institution or 'N/A'}")
            
            with col2:
                # Job experiences
                st.markdown("**Job Experiences:**")
                for job in candidate.get("jobExperiences", []):
                    title = job.get("title", "")
                    company = job.get("companyName", "")
                    duration = job.get("duration", "")
                    
                    # Simple highlight for matching job titles
                    if title and any(term in title.lower() for term in query_terms if len(term) > 3):
                        title = f"<span class='highlight'>{title}</span>"
                    
                    if title or company:
                        st.markdown(f"- {title} at {company or 'Unknown'} ({duration or 'N/A'} years)", unsafe_allow_html=True)
                
                # Skills
                st.markdown("**Skills:**")
                skills_html = ""
                
                # Process skills from the skills array
                skills = []
                for skill in candidate.get("skills", []):
                    if isinstance(skill, dict) and "skillName" in skill:
                        skills.append(skill["skillName"])
                    elif isinstance(skill, str):
                        skills.append(skill)
                
                # Highlight skills that match query terms
                for skill in skills:
                    skill_class = "skill-tag"
                    if any(term in skill.lower() for term in query_terms if len(term) > 3):
                        skill_class = "skill-tag highlight"
                    skills_html += f'<span class="{skill_class}">{skill}</span>'
                
                st.markdown(skills_html, unsafe_allow_html=True)
                
                # Keywords
                if candidate.get("keywords"):
                    st.markdown("**Keywords:**")
                    keywords_html = ""
                    for keyword in candidate.get("keywords", []):
                        keyword_class = "skill-tag"
                        if any(term in keyword.lower() for term in query_terms if len(term) > 3):
                            keyword_class = "skill-tag highlight"
                        keywords_html += f'<span class="{keyword_class}">{keyword}</span>'
                    st.markdown(keywords_html, unsafe_allow_html=True)

# Main application
def main():
    st.title("ZappBot Resume Search")
    st.write("Enhanced resume search powered by LangChain")
    
    # Debug mode toggle
    debug_mode = st.sidebar.checkbox("Debug Mode", value=False)
    
    # Search form
    with st.form("search_form"):
        query = st.text_area("Search Query", 
                             "Find software developer in Indonesia with 3 years experience and SQL and Python skills",
                             height=100)
        top_k = st.number_input("Number of results", min_value=1, max_value=20, value=10)
        submit_button = st.form_submit_button("Search")
    
    if submit_button:
        # Store query for highlighting
        st.session_state.query_text = query
        
        with st.spinner("Searching for candidates..."):
            # Step 1: Get candidates from MongoDB with minimal filtering
            # Increased from 20 to 100 to get more candidates
            candidates = get_candidate_profiles(query, top_k=100)
            
            # Debug: Show the raw candidates data if debug mode is enabled
            if debug_mode:
                with st.expander("Debug: Raw Candidates Count", expanded=True):
                    st.write(f"Retrieved {len(candidates)} candidates from database")
                    
                with st.expander("Debug: First 5 Candidates Sample", expanded=False):
                    # Create a sanitized version without large fields
                    sanitized_candidates = []
                    for c in candidates[:5]:  # Only show first 5
                        sanitized = {k: v for k, v in c.items() if k != 'embedding'}
                        # Further simplify the output
                        if 'skills' in sanitized:
                            if isinstance(sanitized['skills'][0], dict) and "skillName" in sanitized['skills'][0]:
                                sanitized['skills'] = [s.get('skillName') for s in sanitized['skills'][:5]]
                            else:
                                sanitized['skills'] = sanitized['skills'][:5]
                        if 'jobExperiences' in sanitized:
                            sanitized['jobExperiences'] = [
                                {'title': j.get('title', ''), 'duration': j.get('duration', '')}
                                for j in sanitized['jobExperiences'][:3]
                            ]
                        if 'keywords' in sanitized:
                            sanitized['keywords'] = sanitized['keywords'][:5]
                        sanitized_candidates.append(sanitized)
                    
                    st.json(sanitized_candidates)
            
            # Step 2: Use LangChain to score and rank candidates
            top_candidates = score_candidates_with_langchain(query, candidates, top_k=top_k)
            
            # Debug: Show LLM explanation if debug mode is enabled
            if debug_mode and top_candidates and "_matching_explanation" in top_candidates[0]:
                with st.expander("Debug: LLM Matching Logic", expanded=True):
                    st.write(top_candidates[0].get("_matching_explanation", ""))
            
            # Step 3: Display results
            display_candidate_profiles(top_candidates)

if __name__ == "__main__":
    main()
