import sqlite3
import fitz  # PyMuPDF
import re
import ollama
import json
import streamlit as st
import pandas as pd
import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

# Database Setup
def init_db():
    with sqlite3.connect('recruiter.db', timeout=10) as conn:
        cursor = conn.cursor()
        cursor.execute('''CREATE TABLE IF NOT EXISTS Jobs 
                          (job_id TEXT PRIMARY KEY, job_title TEXT, job_description TEXT)''')
        cursor.execute('''CREATE TABLE IF NOT EXISTS Resumes 
                          (resume_id INTEGER PRIMARY KEY AUTOINCREMENT, job_id TEXT, 
                           name TEXT, score INTEGER, summary TEXT, status TEXT)''')
        conn.commit()

# PDF Text Extraction
def extract_text_from_pdf(file_bytes):
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    
    # Cleaning: Remove extra whitespaces and non-standard characters
    text = re.sub(r'\s+', ' ', text).strip()
    return text if text else "Error: No text found (Scan/Image?)"

def analyze_resume(resume_text, job_desc):
    # Constructing a clear system prompt for local LLMs
    system_prompt = "You are a recruitment assistant. You must analyze the resume against the job description and return ONLY valid JSON."
    
    user_prompt = f"""
    Job Description: {job_desc}
    Resume Text: {resume_text}

    Analyze the resume against the job description.
    Return a valid JSON object with the following fields:
    - "name": Candidate's full name (string)
    - "email": Candidate's email address (string)
    - "experience_years": Total years of relevant experience (integer)
    - "skills_match_score": Score from 0-100 based on skills match (integer)
    - "education_score": Score from 0-100 based on education match (integer)
    - "summary": A detailed professional summary (4-6 sentences) highlighting the candidate's key qualifications, experience, and specific fit for the role. Do not leave this empty. (string)

    Ensure the output is valid JSON.
    """

    # API Client Setup
    api_key = os.getenv("GROQ_API_KEY")
    try:
        if "GROQ_API_KEY" in st.secrets:
            api_key = st.secrets["GROQ_API_KEY"]
    except FileNotFoundError:
        pass # Not running in Streamlit Cloud or no secrets file

    if not api_key:
        st.error("⚠️ GROQ_API_KEY is missing! If running on Streamlit Cloud, add it to 'Advanced Settings' -> 'Secrets'.")
            
    client = None
    if api_key:
        client = Groq(api_key=api_key)
    
    # Analyze resume
    if client:
        try:
            response = client.chat.completions.create(
                messages=[
                    {'role': 'system', 'content': system_prompt},
                    {'role': 'user', 'content': user_prompt}
                ],
                model="llama3-70b-8192", # Switched to stable Llama 3 model
                temperature=0.2,
                max_tokens=1024,
                response_format={"type": "json_object"}
            )
            content = response.choices[0].message.content
        except Exception as e:
            st.error(f"API Error: {str(e)}")
            content = "{}" # Fallback to empty JSON string to allow defaults to be set
    else:
        # Fallback to local Ollama if no API key
        try:
            response = ollama.chat(
                model='gemma3:4b',
                messages=[ 
                    {'role': 'system', 'content': system_prompt},
                    {'role': 'user', 'content': user_prompt}
                ],
                options={
                    'temperature': 0.2,
                    'num_ctx': 8192,
                    'num_predict': 512
                }
            )
            content = response['message']['content']
        except Exception as e:
             st.error(f"Ollama Error (Is it running?): {str(e)}")
             content = "{}"
    
    # specific cleanup for markdown code blocks which some models include
    if "```json" in content:
        content = content.split("```json")[1].split("```")[0].strip()
    elif "```" in content:
        content = content.split("```")[0].strip()

    # Try to parse JSON
    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        # Fallback: Try to find JSON object using regex
        match = re.search(r'\{.*\}', content, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group(0))
            except json.JSONDecodeError:
                # st.error(f"Failed to parse JSON. Raw output:\n{content}") # Muted to reduce noise
                data = {}
        else:
            # st.error(f"No JSON found in response. Raw output:\n{content}")
            data = {}
    
    # Ensure all required keys exist with default values
    data.setdefault('name', 'Unknown')
    data.setdefault('email', 'N/A')
    data.setdefault('experience_years', 0)
    data.setdefault('skills_match_score', 50)
    data.setdefault('education_score', 50)
    data.setdefault('summary', 'Unable to generate summary')
    
    # PASS 2: Force detailed summary generation if missing or too short
    # This solves the issue where the single-pass JSON generation truncates the summary
    if data['summary'] == 'Unable to generate summary' or len(data['summary']) < 100:
        summary_prompt = f"""
        Job Description: {job_desc}
        Resume Text: {resume_text}
        
        Write a detailed professional summary (4-6 sentences) of the candidate's profile, highlighting their key skills, experience, and suitability for the role. Focus on actionable insights for a recruiter. Do not include any introductory text, just the summary.
        """
        try:
            if client:
                 summary_response = client.chat.completions.create(
                    messages=[{'role': 'user', 'content': summary_prompt}],
                    model="llama3-70b-8192",
                    temperature=0.4,
                    max_tokens=512
                )
                 new_summary = summary_response.choices[0].message.content.strip()
            else:
                summary_response = ollama.chat(
                    model='gemma3:4b',
                    messages=[{'role': 'user', 'content': summary_prompt}],
                    options={'temperature': 0.4, 'num_ctx': 8192, 'num_predict': 512}
                )
                new_summary = summary_response['message']['content'].strip()
            
            if new_summary:
                data['summary'] = new_summary
        except Exception:
            pass # Keep the default if 2nd pass fails
    
    return data

# Page Config must be the first Streamlit command
st.set_page_config(page_title="DevLabs AI Resume Screening", page_icon="https://media.licdn.com/dms/image/v2/D4D0BAQGgmm3eOlAjXg/company-logo_200_200/company-logo_200_200/0/1694777298283/devlabsalliance_logo?e=2147483647&v=beta&t=htaMe525FbPL787VFd54WUM1flrpsrEs2DMa3ym00qs", layout="wide", initial_sidebar_state="collapsed")

def load_css():
    st.markdown("""
        <style>
        /* Import Google Fonts - Inter for modern typography */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
        
        /* ===== GLOBAL STYLES ===== */
        html, body, [class*="css"] {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
        }
        
        /* Main App Background - Soft Blue Gradient */
        .stApp {
            background: linear-gradient(135deg, #dbeafe 0%, #f0f9ff 30%, #ffffff 100%) !important;
            min-height: 100vh;
        }
        
        /* Hide Streamlit's default sidebar completely */
        section[data-testid="stSidebar"] {
            display: none !important;
        }
        
        /* Main content area */
        .main .block-container {
            padding: 0 !important;
            max-width: 100% !important;
        }
        
        /* ===== HEADER STYLING ===== */
        .app-header {
            background: white;
            border-bottom: 1px solid #e5e7eb;
            padding: 12px 24px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            position: sticky;
            top: 0;
            z-index: 1000;
        }
        
        .logo {
            display: flex;
            align-items: center;
            gap: 10px;
            font-size: 1.25rem;
            font-weight: 700;
            color: #111827;
        }
        
        .logo-icon {
            font-size: 1.5rem;
        }
        
        /* ===== MAIN LAYOUT ===== */
        .main-container {
            display: flex;
            height: calc(100vh - 65px);
            overflow: hidden;
        }
        
        /* Left Sidebar - Job List */
        .job-sidebar {
            width: 200px;
            min-width: 200px;
            background: white;
            border-right: 1px solid #e5e7eb;
            display: flex;
            flex-direction: column;
            padding: 16px;
            gap: 12px;
            overflow-y: auto;
        }
        
        .new-job-btn {
            background: #3b82f6 !important;
            color: white !important;
            border: none !important;
            border-radius: 20px !important;
            padding: 10px 16px !important;
            font-weight: 600 !important;
            font-size: 0.9rem !important;
            cursor: pointer;
            display: flex;
            align-items: center;
            gap: 6px;
            transition: all 0.2s ease !important;
            width: 100%;
            justify-content: center;
        }
        
        .new-job-btn:hover {
            background: #2563eb !important;
            transform: translateY(-1px);
        }
        
        .job-item {
            padding: 12px 14px;
            border-radius: 10px;
            cursor: pointer;
            transition: all 0.15s ease;
            border: 1px solid transparent;
            font-size: 0.9rem;
            color: #374151;
        }
        
        .job-item:hover {
            background: #f3f4f6;
        }
        
        .job-item.active {
            background: #eff6ff;
            border-color: #3b82f6;
            color: #1d4ed8;
            font-weight: 600;
        }
        
        .job-title-text {
            font-weight: 500;
            margin-bottom: 2px;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }
        
        .job-id-text {
            font-size: 0.75rem;
            color: #f7fafa;
        }
        
        /* Middle Panel - Job Description */
        .description-panel {
            flex: 1;
            padding: 24px;
            display: flex;
            flex-direction: column;
            gap: 16px;
            overflow-y: auto;
            min-width: 300px;
        }
        
        .panel-title {
            font-size: 0.85rem;
            font-weight: 600;
            color: #6b7280;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-bottom: 4px;
        }
        
        /* White text for Your Jobs panel title */
        .panel-title-white {
            font-size: 0.85rem;
            font-weight: 600;
            color: #141414 !important;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-bottom: 4px;
        }
        
        /* Right Panel - Upload & Results */
        .upload-panel {
            width: 420px;
            min-width: 380px;
            padding: 24px;
            border-left: 1px solid #e5e7eb;
            background: rgba(255, 255, 255, 0.5);
            display: flex;
            flex-direction: column;
            gap: 16px;
            overflow-y: auto;
        }
        
        /* ===== TYPOGRAPHY ===== */
        h1, h2, h3, h4, h5, h6 {
            color: #111827 !important;
            font-weight: 700 !important;
            letter-spacing: -0.02em !important;
        }
        
        p, div, span, label {
            color: #374151 !important;
            line-height: 1.6 !important;
        }
        /* ===== INPUTS & TEXTAREAS ===== */
        div[data-baseweb="input"], 
        div[data-baseweb="base-input"], 
        div[data-baseweb="textarea"],
        div[data-baseweb="select"] {
            background-color: #FFFFFF !important;
            border: 1px solid #e5e7eb !important;
            border-radius: 12px !important;
            box-shadow: 0 1px 3px rgba(0,0,0,0.04) !important;
        }
        
        /* Focus states with blue accent */
        div[data-baseweb="input"]:focus-within,
        div[data-baseweb="textarea"]:focus-within {
            border-color: #3b82f6 !important;
            box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.15) !important;
        }
        
        input[type="text"], 
        textarea, 
        .stTextArea textarea, 
        .stTextInput input {
            color: #111827 !important;
            caret-color: #3b82f6 !important;
            background-color: #FFFFFF !important;
            font-size: 0.95rem !important;
            padding: 12px 16px !important;
            border: none !important;
        }
        
        /* Large textarea for job description */
        .stTextArea textarea {
            min-height: 300px !important;
            padding: 20px !important;
            font-size: 0.95rem !important;
            line-height: 1.7 !important;
            border-radius: 12px !important;
        }
        
        /* Input Labels */
        .stTextInput label, 
        .stSelectbox label, 
        .stTextArea label, 
        .stFileUploader label {
            color: #111827 !important;
            font-weight: 600 !important;
            font-size: 0.95rem !important;
            margin-bottom: 8px !important;
        }
        
        /* Placeholder text */
        ::placeholder {
            color: #9ca3af !important;
            opacity: 1 !important;
        }
        
        /* ===== BUTTONS ===== */
        div.stButton > button {
            background-color: #111827 !important;
            color: white !important;
            border-radius: 12px !important;
            padding: 12px 24px !important;
            border: none !important;
            font-weight: 600 !important;
            transition: all 0.2s ease !important;
            font-size: 0.95rem !important;
        }
        
        div.stButton > button:hover {
            background-color: #000000 !important;
            transform: translateY(-2px) !important;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15) !important;
        }
        
        /* Blue primary buttons */
        div.stButton > button[kind="primary"],
        div.stButton > button[kind="primary"] p,
        div.stButton > button[kind="primary"] span,
        div.stButton > button[kind="primary"] div,
        div[data-testid="stFormSubmitButton"] > button,
        div[data-testid="stFormSubmitButton"] > button p,
        div[data-testid="stFormSubmitButton"] > button span {
            background-color: #3b82f6 !important;
            color: #FFFFFF !important;
        }
        
        /* Secondary buttons (job history items) - blue like save button */
        div.stButton > button[kind="secondary"],
        div.stButton > button[kind="secondary"] p,
        div.stButton > button[kind="secondary"] span,
        div.stButton > button[kind="secondary"] div {
            background-color: #3b82f6 !important;
            color: #FFFFFF !important;
            text-align: left !important;
            justify-content: flex-start !important;
            padding-left: 12px !important;
        }
        
        div.stButton > button[kind="primary"]:hover,
        div[data-testid="stFormSubmitButton"] > button:hover {
            background-color: #2563eb !important;
        }
        
        /* ===== FILE UPLOADER ===== */
        section[data-testid="stFileUploaderDropzone"] {
            background-color: #FFFFFF !important;
            border: 2px dashed #cbd5e1 !important;
            border-radius: 16px !important;
            padding: 40px 30px !important;
            text-align: center !important;
            transition: all 0.2s ease !important;
            min-height: 180px !important;
            display: flex !important;
            flex-direction: column !important;
            align-items: center !important;
            justify-content: center !important;
        }
        
        section[data-testid="stFileUploaderDropzone"]:hover {
            border-color: #3b82f6 !important;
            background-color: #f8fafc !important;
        }
        
        section[data-testid="stFileUploaderDropzone"] div, 
        section[data-testid="stFileUploaderDropzone"] span,
        section[data-testid="stFileUploaderDropzone"] small {
            color: #6b7280 !important;
        }
        
        /* Browse button in file uploader - blue */
        section[data-testid="stFileUploaderDropzone"] button,
        section[data-testid="stFileUploaderDropzone"] button p,
        section[data-testid="stFileUploaderDropzone"] button span {
            background-color: #3b82f6 !important;
            color: #FFFFFF !important;
        }
        
        section[data-testid="stFileUploaderDropzone"] button:hover {
            background-color: #2563eb !important;
        }
        
        /* ===== RESULTS CARDS ===== */
        .result-card {
            background: white;
            border-radius: 12px;
            padding: 16px;
            margin-bottom: 12px;
            border: 1px solid #e5e7eb;
            transition: all 0.2s ease;
            box-shadow: 0 1px 3px rgba(0,0,0,0.04);
        }
        
        .result-card:hover {
            border-color: #3b82f6;
            box-shadow: 0 4px 12px rgba(59, 130, 246, 0.1);
        }
        
        .result-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 8px;
        }
        
        .candidate-name {
            font-weight: 600;
            color: #111827;
            font-size: 1rem;
        }
        
        .score-badge {
            background: linear-gradient(135deg, #3b82f6, #1d4ed8);
            color: white;
            padding: 6px 12px;
            border-radius: 20px;
            font-weight: 700;
            font-size: 0.9rem;
        }
        
        .score-badge.high {
            background: linear-gradient(135deg, #10b981, #059669);
        }
        
        .score-badge.medium {
            background: linear-gradient(135deg, #f59e0b, #d97706);
        }
        
        .score-badge.low {
            background: linear-gradient(135deg, #ef4444, #dc2626);
        }
        
        .result-summary {
            font-size: 0.85rem;
            color: #6b7280;
            line-height: 1.5;
        }
        
        .status-pill {
            display: inline-block;
            padding: 4px 10px;
            border-radius: 20px;
            font-size: 0.75rem;
            font-weight: 600;
            margin-top: 8px;
        }
        
        .status-shortlisted {
            background: #d1fae5;
            color: #065f46;
        }
        
        .status-rejected {
            background: #fee2e2;
            color: #991b1b;
        }
        
        /* ===== STATS BAR ===== */
        .stats-bar {
            display: flex;
            gap: 24px;
            padding: 12px 20px;
            background: white;
            border-radius: 12px;
            border: 1px solid #e5e7eb;
            margin-bottom: 16px;
        }
        
        .stat-item {
            text-align: center;
        }
        
        .stat-value {
            font-size: 1.5rem;
            font-weight: 700;
            color: #111827;
        }
        
        .stat-label {
            font-size: 0.75rem;
            color: #6b7280;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }
        
        /* ===== PROGRESS BAR ===== */
        .stProgress > div > div {
            background-color: #3b82f6 !important;
            border-radius: 10px !important;
        }
        
        .stProgress > div {
            background-color: #e5e7eb !important;
            border-radius: 10px !important;
        }
        
        /* ===== ALERTS ===== */
        div.stAlert {
            border-radius: 12px !important;
            border: none !important;
        }
        
        /* ===== EMPTY STATE ===== */
        .empty-state {
            text-align: center;
            padding: 40px 20px;
            color: #9ca3af;
        }
        
        .empty-state-icon {
            font-size: 3rem;
            margin-bottom: 16px;
        }
        
        .empty-state-title {
            font-size: 1.1rem;
            font-weight: 600;
            color: #6b7280;
            margin-bottom: 8px;
        }
        
        .empty-state-text {
            font-size: 0.9rem;
            color: #9ca3af;
        }
        
        /* Hide Streamlit branding */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
        
        /* Fix column gaps */
        /* Fix column gaps */
        [data-testid="column"] {
            padding: 0 8px !important;
        }

        /* Hide the Browse button in file uploader */
        [data-testid='stFileUploader'] section > button {
            display: none !important;
        }
        
        /* Adjust dropzone padding since button is gone */
        [data-testid='stFileUploaderDropzone'] {
            padding: 60px 20px !important;
        }

        /* Processing Button Animation */
        @keyframes pulse-animation {
            0% { opacity: 1; transform: scale(1); }
            50% { opacity: 0.8; transform: scale(0.98); }
            100% { opacity: 1; transform: scale(1); }
        }
        
        .processing-btn {
            animation: pulse-animation 1.5s infinite ease-in-out;
            width: 100%;
            border: none;
            background: #3b82f6;
            color: white;
            padding: 12px 24px;
            border-radius: 12px;
            font-weight: 600;
            cursor: not-allowed;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 8px;
            box-shadow: 0 4px 12px rgba(59, 130, 246, 0.3);
        }
        </style>
    """, unsafe_allow_html=True)

load_css()
init_db()

# Initialize session state
if 'selected_job_id' not in st.session_state:
    st.session_state.selected_job_id = None
if 'creating_new_job' not in st.session_state:
    st.session_state.creating_new_job = False
if 'screening_complete' not in st.session_state:
    st.session_state.screening_complete = False

# ===== HEADER =====
st.markdown("""
    <div class="app-header">
        <div class="logo">
            <span class="logo-icon"></span>
            <img src="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRxvcmVB85wmTZ1kLG2nK8KQXPnihQDzpEtXw&s" alt="DevLabs AI Logo" style="width: 50px; height: 50px; margin-right:10px">
            <span>DevLabs AI Resume Screening</span>
        </div>
        <div style="display: flex; gap: 16px; align-items: center;">
            <span style="font-size: 0.9rem; color: #6b7280;">Screen candidates intelligently with AI</span>
        </div>
    </div>
""", unsafe_allow_html=True)

# ===== MAIN LAYOUT: 3-COLUMN =====
col_sidebar, col_description, col_upload = st.columns([1, 2.5, 2])

# ===== LEFT SIDEBAR: JOB LIST =====
with col_sidebar:
    st.markdown('<p class="panel-title-white">Your Jobs</p>', unsafe_allow_html=True)
    
    # New Job Button
    if st.button("➕ New Job", key="new_job_btn", use_container_width=True):
        st.session_state.creating_new_job = True
        st.session_state.selected_job_id = None
    
    st.markdown("<hr style='margin: 12px 0; border-color: #e5e7eb;'>", unsafe_allow_html=True)
    
    # Fetch all jobs
    with sqlite3.connect('recruiter.db', timeout=10) as conn:
        jobs_df = pd.read_sql_query("SELECT job_id, job_title FROM Jobs ORDER BY job_id DESC", conn)
    
    if jobs_df.empty:
        st.markdown("""
            <div class="empty-state">
                <div class="empty-state-icon">📋</div>
                <div class="empty-state-title">No jobs yet</div>
                <div class="empty-state-text">Click "+ New Job" to create your first job posting</div>
            </div>
        """, unsafe_allow_html=True)
    else:
        # Scrollable container for job list
        with st.container(height=500, border=False):
            for _, job in jobs_df.iterrows():
                is_active = st.session_state.selected_job_id == job['job_id']
                button_type = "primary" if is_active else "secondary"
                
                if st.button(
                    f"📄 {job['job_title'][:20]}..." if len(job['job_title']) > 20 else f"📄 {job['job_title']}",
                    key=f"job_{job['job_id']}",
                    use_container_width=True,
                    type=button_type
                ):
                    st.session_state.selected_job_id = job['job_id']
                    st.session_state.creating_new_job = False
                    st.rerun()

# ===== CENTER PANEL: JOB DESCRIPTION =====
with col_description:
    st.markdown('<p class="panel-title">Job Description</p>', unsafe_allow_html=True)
    
    if st.session_state.creating_new_job:
        # CREATE NEW JOB FORM
        st.markdown("### Create New Job")
        
        # Create New Job - No form to allow real-time validation
        
        new_job_id = st.text_input("Job ID", placeholder="e.g., SE-101", help="Unique identifier for this job")
        
        # Check for duplicate ID immediately
        is_duplicate = False
        if new_job_id:
            with sqlite3.connect('recruiter.db', timeout=10) as conn:
                existing = conn.execute("SELECT 1 FROM Jobs WHERE job_id = ?", (new_job_id,)).fetchone()
                if existing:
                    st.error(f"⚠️ Job ID '{new_job_id}' already exists!")
                    is_duplicate = True

        new_job_title = st.text_input("Job Title", placeholder="e.g., Senior Software Engineer")
        new_job_desc = st.text_area(
            "Job Description",
            height=300,
            placeholder="Enter the full job description here...\n\nInclude:\n• Required skills and qualifications\n• Years of experience needed\n• Key responsibilities\n• Nice-to-have skills"
        )
        
        # Form buttons
        col_save, col_cancel = st.columns(2)
        
        # Save button - disabled if duplicate ID
        if col_save.button("💾 Save Job", use_container_width=True, type="primary", disabled=is_duplicate):
            if new_job_id and new_job_title and new_job_desc:
                if not is_duplicate: # Double check
                    with sqlite3.connect('recruiter.db', timeout=10) as conn:
                        try:
                            conn.execute("INSERT INTO Jobs VALUES (?, ?, ?)", (new_job_id, new_job_title, new_job_desc))
                            conn.commit()
                            
                            # Success! Update state and rerun to show in sidebar
                            st.session_state.selected_job_id = new_job_id
                            st.session_state.creating_new_job = False
                            st.success(f"✅ Job created successfully!")
                            st.rerun()
                            
                        except sqlite3.IntegrityError:
                            st.error(f"❌ Job ID '{new_job_id}' already exists.")
            else:
                st.warning("⚠️ Please fill in all fields.")

        # Cancel button
        if col_cancel.button("Cancel", use_container_width=True):
            st.session_state.creating_new_job = False
            st.rerun()
    
    elif st.session_state.selected_job_id:
        # SHOW SELECTED JOB
        with sqlite3.connect('recruiter.db', timeout=10) as conn:
            job_data = conn.execute(
                "SELECT job_id, job_title, job_description FROM Jobs WHERE job_id = ?",
                (st.session_state.selected_job_id,)
            ).fetchone()
        
        if job_data:
            st.markdown(f"### {job_data[1]}")
            st.markdown(f"<span style='color: #6b7280; font-size: 0.85rem;'>ID: {job_data[0]}</span>", unsafe_allow_html=True)
            
            # Editable job description
            updated_desc = st.text_area(
                "Edit Description",
                value=job_data[2],
                height=350,
                label_visibility="collapsed"
            )
            
            if st.button("💾 Update Job", use_container_width=True):
                with sqlite3.connect('recruiter.db', timeout=10) as conn:
                    conn.execute("UPDATE Jobs SET job_description = ? WHERE job_id = ?", (updated_desc, job_data[0]))
                    conn.commit()
                st.success("✅ Job description updated!")
                st.rerun()
        else:
            st.error("Job not found!")
            st.session_state.selected_job_id = None
    
    else:
        # NO JOB SELECTED - SHOW INSTRUCTIONS
        st.markdown("""
            <div class="empty-state" style="padding: 60px 40px;">
                <div class="empty-state-icon">📝</div>
                <div class="empty-state-title">Select or create a job</div>
                <div class="empty-state-text">
                    Choose a job from the left sidebar to view and edit its description,<br>
                    or click "+ New Job" to create a new job posting.
                </div>
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown("### 📖 How it works")
        st.markdown("""
        1. **Create a job** — Click "+ New Job" and enter the job details
        2. **Upload resumes** — Drag and drop PDF resumes on the right
        3. **Get AI rankings** — See candidates ranked by match score instantly
        """)

# ===== RIGHT PANEL: UPLOAD & RESULTS =====
with col_upload:
    st.markdown('<p class="panel-title">Resume Screening</p>', unsafe_allow_html=True)
    
    if st.session_state.selected_job_id:
        # Get job description for screening
        with sqlite3.connect('recruiter.db', timeout=10) as conn:
            job_info = conn.execute(
                "SELECT job_title, job_description FROM Jobs WHERE job_id = ?",
                (st.session_state.selected_job_id,)
            ).fetchone()
            
            # Get existing results for this job
            results_df = pd.read_sql_query(
                "SELECT * FROM Resumes WHERE job_id = ? ORDER BY score DESC",
                conn,
                params=(st.session_state.selected_job_id,)
            )
        
        if job_info:
            job_title, job_desc = job_info
            
            # Stats bar if there are results
            if not results_df.empty:
                total = len(results_df)
                shortlisted = len(results_df[results_df['status'] == 'Shortlisted'])
                avg_score = results_df['score'].mean()
                
                st.markdown(f"""
                    <div class="stats-bar">
                        <div class="stat-item">
                            <div class="stat-value">{total}</div>
                            <div class="stat-label">Resumes</div>
                        </div>
                        <div class="stat-item">
                            <div class="stat-value">{shortlisted}</div>
                            <div class="stat-label">Shortlisted</div>
                        </div>
                        <div class="stat-item">
                            <div class="stat-value">{avg_score:.0f}%</div>
                            <div class="stat-label">Avg Score</div>
                        </div>
                    </div>
                """, unsafe_allow_html=True)
            
            # File uploader
            uploaded_files = st.file_uploader(
                "Upload Resumes",
                type="pdf",
                accept_multiple_files=True,
                help="Upload PDF resumes to screen against this job",
                label_visibility="collapsed"
            )
            
            
            if uploaded_files:
                screen_button_placeholder = st.empty()
                if screen_button_placeholder.button("🚀 Start Screening", use_container_width=True, type="primary"):
                    screen_button_placeholder.markdown("""
                        <button class="processing-btn">
                            ⏳ Processing Resumes...
                        </button>
                    """, unsafe_allow_html=True)
                    
                    st.markdown("---")
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    has_error = False
                    
                    with st.spinner("Processing resumes..."):
                        for i, file in enumerate(uploaded_files):
                            file_name = file.name
                            status_text.markdown(f"🔍 **Processing:** {file_name}")
                            
                            # Extract text
                            text = extract_text_from_pdf(file.read())
                            status_text.markdown(f"🔍 **Processing:** {file_name} ({len(text)} chars extracted)")
                            
                            if text.startswith("Error:"):
                                st.error(f"❌ {file_name}: {text} (Make sure the PDF contains selectable text, not just scanned images.)")
                                progress_bar.progress((i + 1) / len(uploaded_files))
                                continue

                            # Analyze
                            data = analyze_resume(text, job_desc)
                            
                            # Calculate score
                            try:
                                weighted_score = (data['skills_match_score'] * 0.7) + \
                                                 (min(data['experience_years'] * 10, 100) * 0.2) + \
                                                 (data['education_score'] * 0.1)
                            except KeyError:
                                weighted_score = 0
                            
                            status = "Shortlisted" if weighted_score >= 70 else "Rejected"
                            
                            # Save to database
                            try:
                                with sqlite3.connect('recruiter.db', timeout=10) as conn:
                                    conn.execute(
                                        "INSERT INTO Resumes (job_id, name, score, summary, status) VALUES (?, ?, ?, ?, ?)",
                                        (st.session_state.selected_job_id, data['name'], int(weighted_score), data['summary'], status)
                                    )
                                    conn.commit()
                            except Exception as e:
                                st.error(f"❌ Database Error for {file_name}: {str(e)}")
                                has_error = True
                            
                            progress_bar.progress((i + 1) / len(uploaded_files))
                    
                    status_text.empty()
                    
                    if not has_error:
                        st.success(f"✅ Screened {len(uploaded_files)} resumes!")
                        import time
                        time.sleep(1) # Visual pause
                        st.session_state.screening_complete = True
                        st.rerun()
            
            # Display results
            if not results_df.empty:
                st.markdown("### 📊 Ranked Candidates")
                
                for _, candidate in results_df.iterrows():
                    score = candidate['score']
                    score_class = "high" if score >= 80 else "medium" if score >= 60 else "low"
                    status_class = "status-shortlisted" if candidate['status'] == "Shortlisted" else "status-rejected"
                    
                    st.markdown(f"""
                        <div class="result-card">
                            <div class="result-header">
                                <span class="candidate-name">{candidate['name']}</span>
                                <span class="score-badge {score_class}">{score}%</span>
                            </div>
                            <div class="result-summary">{candidate['summary'][:150]}...</div>
                            <span class="status-pill {status_class}">{candidate['status']}</span>
                        </div>
                    """, unsafe_allow_html=True)
            else:
                st.markdown("""
                    <div class="empty-state" style="margin-top: 20px;">
                        <div class="empty-state-icon">📤</div>
                        <div class="empty-state-title">Upload resumes to screen</div>
                        <div class="empty-state-text">
                            Drag and drop PDF files above<br>
                            to analyze candidates for this job
                        </div>
                    </div>
                """, unsafe_allow_html=True)
    
    else:
        # No job selected
        st.markdown("""
            <div class="empty-state" style="padding: 80px 20px;">
                <div class="empty-state-icon">👈</div>
                <div class="empty-state-title">Select a job first</div>
                <div class="empty-state-text">
                    Choose a job from the left sidebar<br>
                    to start screening resumes
                </div>
            </div>
        """, unsafe_allow_html=True)
