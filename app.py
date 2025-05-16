import streamlit as st
import json
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
TOP_K_DEFAULT = 50

# Connect to MongoDB
def get_mongo_client() -> MongoClient:
    return MongoClient(**MONGO_CFG)

# Direct query to MongoDB with minimal filtering
def get_candidate_profiles(query_text: str, top_k: int = TOP_K_DEFAULT) -> List[Dict[str, Any]]:
    """
    Get candidate profiles from MongoDB with minimal filtering.
    """
    try:
        with get_mongo_client() as client:
            coll = client[DB_NAME][COLL_NAME]
            
            # Simply get all profiles without filtering
            # We'll use the LLM to score them instead
            candidates = list(coll.find({}, {
                "_id": 0, 
                "embedding": 0  # Exclude embedding field as it's large and not needed
            }).limit(top_k))
            
            return candidates
    except Exception as e:
        st.error(f"Database error: {str(e)}")
        return []

# Use LangChain to score candidates based on the query
def score_candidates_with_langchain(query: str, candidates: List[Dict[str, Any]], top_k: int = 10) -> List[Dict[str, Any]]:
    """
    Score candidates using LangChain and return the top matches.
    """
    try:
        if not candidates:
            return []
        
        # Initialize LangChain components
        llm = ChatOpenAI(model=MODEL_NAME, api_key=OPENAI_API_KEY, temperature=0)
        
        # Define output schema for structured parsing
        response_schemas = [
            ResponseSchema(name="top_candidates", 
                          description=f"List of IDs for the top {top_k} candidates that best match the query"),
            ResponseSchema(name="reasoning", 
                          description="Brief explanation of how you evaluated the candidates")
        ]
        output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
        format_instructions = output_parser.get_format_instructions()
        
        # Create prompt template
        template = """
        You are a sophisticated HR matching system. Your task is to identify the best candidates 
        for a job based on the given query.
        
        QUERY: {query}
        
        CANDIDATES:
        {candidates}
        
        Analyze each candidate's profile and evaluate how well they match the query. 
        Consider these factors:
        1. Job titles matching the requirements
        2. Relevant skills and keywords
        3. Experience level
        4. Location if specified
        
        Select the top {top_k} candidates that best match the query.
        
        {format_instructions}
        """
        
        prompt = ChatPromptTemplate.from_template(template)
        
        # Prepare candidate data (with resumeIds and names for reference)
        candidate_data = [
            {
                "resumeId": c.get("resumeId", ""),
                "name": c.get("name", "Unknown"),
                "skills": [s.get("skillName") for s in c.get("skills", []) if isinstance(s, dict) and "skillName" in s],
                "keywords": c.get("keywords", []),
                "jobExperiences": [
                    {
                        "title": j.get("title", ""),
                        "duration": j.get("duration", "")
                    } for j in c.get("jobExperiences", [])
                ],
                "country": c.get("country", "")
            } for c in candidates
        ]
        
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
        
        # Extract top candidates
        top_candidates = [c for c in candidates if c.get("resumeId") in top_ids]
        
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
                total_exp = sum(float(exp.get("duration", 0)) for exp in experiences if exp.get("duration", "").isdigit() or exp.get("duration", "").replace(".", "").isdigit())
                st.markdown(f"**Total Experience:** {total_exp} years")
            
            with col2:
                # Job experiences
                st.markdown("### Job Experiences")
                for job in candidate.get("jobExperiences", []):
                    if job.get("title") and job.get("companyName"):
                        duration = job.get("duration", "N/A")
                        st.markdown(f"- **{job.get('title')}** at {job.get('companyName')} ({duration} years)")
                
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

# Main application
def main():
    st.title("ZappBot Resume Search")
    st.write("Direct LangChain-based resume search without dictionary variants or regex")
    
    # Search form
    with st.form("search_form"):
        query = st.text_area("Search Query", 
                             "Find software developer in Indonesia with 3 years experience and SQL and Python skills",
                             height=100)
        top_k = st.number_input("Number of results", min_value=1, max_value=50, value=10)
        submit_button = st.form_submit_button("Search")
    
    if submit_button:
        with st.spinner("Searching for candidates..."):
            # Step 1: Get candidates from MongoDB (minimal filtering)
            candidates = get_candidate_profiles(query, top_k=50)  # Get more candidates for LLM to analyze
            
            # Step 2: Use LangChain to score and rank candidates
            top_candidates = score_candidates_with_langchain(query, candidates, top_k=top_k)
            
            # Step 3: Display results
            display_candidate_profiles(top_candidates)

if __name__ == "__main__":
    main()
