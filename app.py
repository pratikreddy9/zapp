"""
ZappBot: Resume‑filtering chatbot with optimized display + email sender + job match counts
LangChain 0.3.25 • OpenAI 1.78.1 • Streamlit 1.34+
"""

import os, json, re
from datetime import datetime

import streamlit as st
from langchain_openai import ChatOpenAI
from langchain.agents import create_openai_tools_agent, AgentExecutor
from langchain.memory import ConversationBufferMemory

# Import modular components
from prompts import agent_prompt
from design import display_resume_grid
from variants import expand
from utils import (
    get_mongo_client, 
    extract_resume_ids_from_response, 
    process_response, 
    attach_hidden_resume_ids
)
from tools import query_db, send_email, get_job_match_counts, get_resume_id_by_name

# ── CONFIG ─────────────────────────────────────────────────────────────
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
MODEL_NAME = "gpt-4o"
DB_NAME = "resumes_database"
COLL_NAME = "resumes"

# ── AGENT + MEMORY ─────────────────────────────────────────────────────
llm = ChatOpenAI(model=MODEL_NAME, api_key=OPENAI_API_KEY, temperature=0)

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
                
                # Display the resume grid
                display_resume_grid(resp['processed']['resumes'])
                
                # Add a row with email button and job match button
                cols = st.columns([2, 1, 1])
                
                # Email button
                with cols[1]:
                    if resp['processed']['resumes']:
                        if st.button(f"📧 Email Results", key=f"email_btn_{i}"):
                            try:
                                # Format email body using reformat_email_body from utils.py
                                from utils import reformat_email_body
                                
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
    
    # Display MongoDB Queries in a separate expander
    if debug_mode and "mongo_queries" in st.session_state and st.session_state.mongo_queries:
        with st.expander("Recent MongoDB Queries"):
            for i, q in enumerate(st.session_state.mongo_queries):
                st.markdown(f"**Query {i+1} - {q['timestamp']}**")
                st.code(q["query"], language="json")
                st.write("Parameters:")
                st.json(q["parameters"])
                if i < len(st.session_state.mongo_queries) - 1:
                    st.markdown("---")  # Add separator between queries
