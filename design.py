import streamlit as st
from typing import List, Dict, Any

def display_resume_grid(resumes, container=None):
    """
    Display resumes in a 3x3 grid layout with styled cards.
    
    Args:
        resumes: List of resume dictionaries to display
        container: Optional Streamlit container to render into (defaults to st)
    """
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
    .keyword-tag {
        display: inline-block;
        background-color: #FFF8E1;
        color: #FF8F00;
        border-radius: 12px;
        padding: 3px 10px;
        margin: 3px;
        font-size: 12px;
        font-weight: 500;
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
                keywords = resume.get("keywords", [])  # Extract keywords
                
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
                            html += f'<span class="skill-tag">{skill}</span>'
                        html += '</div>'
                    
                    # Add keywords section (with different styling)
                    if keywords:
                        html += f'<div class="resume-section-title">Keywords</div><div>'
                        for keyword in keywords[:5]:  # Limit to 5 keywords
                            html += f'<span class="keyword-tag">{keyword}</span>'
                        html += '</div>'
                    
                    # Show resume ID in debug mode
                    debug_mode = getattr(st.session_state, 'debug_mode', False)
                    if debug_mode and resume_id:
                        html += f'<div class="resume-id">ID: {resume_id}</div>'
                    
                    html += '</div>'
                    st.markdown(html, unsafe_allow_html=True)
