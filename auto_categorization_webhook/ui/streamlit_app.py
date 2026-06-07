"""
ui/streamlit_app.py
-------------------
Modern SaaS-style Streamlit dashboard for the Auto Categorization Webhook.

This app communicates with the FastAPI backend to classify support tickets
using an LLM-powered webhook. It includes robust error handling, session
history management, and dynamic confidence visualization.
"""

from __future__ import annotations

import os
import time
from datetime import datetime
from typing import Any, Dict, Optional

import pandas as pd
import requests
import streamlit as st
from dotenv import load_dotenv

# ======================================================================= #
# Configuration & Setup                                                     #
# ======================================================================= #

# Load environment variables
load_dotenv()

BACKEND_URL: str = os.getenv("BACKEND_URL", "http://localhost:8000").rstrip("/")
API_KEY: str = os.getenv("API_KEY", "")

st.set_page_config(
    page_title="AI Auto Categorization",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ======================================================================= #
# Styling (CSS)                                                             #
# ======================================================================= #

def inject_custom_css() -> None:
    """Inject custom CSS for a modern, SaaS-style look."""
    st.markdown(
        """
        <style>
        /* Gradient Header */
        .gradient-header {
            background: linear-gradient(90deg, #1E3A8A 0%, #3B82F6 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-size: 3rem;
            font-weight: 800;
            margin-bottom: 0px;
            padding-bottom: 0px;
        }
        
        /* Subtitle */
        .subtitle {
            color: #6B7280;
            font-size: 1.2rem;
            margin-top: 0px;
            margin-bottom: 2rem;
        }

        /* Metric Cards */
        [data-testid="stMetric"] {
            background-color: var(--secondary-background-color);
            border-radius: 12px;
            padding: 15px;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
            border: 1px solid rgba(128, 128, 128, 0.2);
        }

        /* Custom Status Badges */
        .badge-success {
            background-color: #DEF7EC;
            color: #03543F;
            padding: 4px 8px;
            border-radius: 6px;
            font-weight: 600;
            font-size: 0.9em;
        }
        .badge-error {
            background-color: #FDE8E8;
            color: #9B1C1C;
            padding: 4px 8px;
            border-radius: 6px;
            font-weight: 600;
            font-size: 0.9em;
        }
        .badge-warning {
            background-color: #FEF3C7;
            color: #92400E;
            padding: 4px 8px;
            border-radius: 6px;
            font-weight: 600;
            font-size: 0.9em;
        }

        /* Progress Bar Wrapper */
        .progress-wrapper {
            width: 100%;
            background-color: #E5E7EB;
            border-radius: 9999px;
            height: 12px;
            margin-top: 8px;
            margin-bottom: 8px;
            overflow: hidden;
        }
        .progress-fill {
            height: 100%;
            border-radius: 9999px;
            transition: width 0.5s ease-in-out;
        }
        .progress-green { background-color: #10B981; }
        .progress-orange { background-color: #F59E0B; }
        .progress-red { background-color: #EF4444; }

        </style>
        """,
        unsafe_allow_html=True,
    )


# ======================================================================= #
# State Management                                                          #
# ======================================================================= #

def init_session_state() -> None:
    """Initialize Streamlit session state variables."""
    if "history" not in st.session_state:
        st.session_state.history = []
    if "last_health" not in st.session_state:
        st.session_state.last_health = None


# ======================================================================= #
# API Interactions                                                          #
# ======================================================================= #

def check_backend_health() -> Optional[Dict[str, Any]]:
    """Fetch health status from the backend."""
    try:
        response = requests.get(f"{BACKEND_URL}/health", timeout=5.0)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException:
        return None


def classify_ticket(ticket_id: str, title: str, description: str) -> tuple[Optional[Dict[str, Any]], Optional[str]]:
    """
    Send a classification request to the backend.
    Returns (data, error_message).
    """
    headers = {
        "Content-Type": "application/json",
        "X-API-Key": API_KEY,
    }
    payload = {
        "ticket_id": ticket_id,
        "title": title,
        "description": description,
    }

    try:
        response = requests.post(
            f"{BACKEND_URL}/classify",
            json=payload,
            headers=headers,
            timeout=40.0
        )
        
        if response.status_code == 200:
            return response.json(), None
        elif response.status_code == 401:
            return None, "401 Unauthorized: Invalid or missing API Key."
        elif response.status_code == 422:
            return None, f"422 Validation Error: {response.json().get('detail', 'Invalid input')}"
        elif response.status_code == 503:
            return None, "503 Service Unavailable: Rate limit exceeded or backend unreachable."
        elif response.status_code == 504:
            return None, "504 Gateway Timeout: Classification timed out."
        else:
            return None, f"Unexpected Error {response.status_code}: {response.text}"

    except requests.exceptions.ConnectionError:
        return None, "Connection Error: Cannot reach the backend service."
    except requests.exceptions.Timeout:
        return None, "Timeout: Request took too long to complete."
    except Exception as e:
        return None, f"An unexpected error occurred: {str(e)}"


# ======================================================================= #
# UI Components                                                             #
# ======================================================================= #

def render_sidebar() -> None:
    """Render the sidebar with system information and configuration."""
    with st.sidebar:
        st.image("https://cdn-icons-png.flaticon.com/512/4712/4712109.png", width=60)
        st.title("System Config")
        
        st.markdown("---")
        st.markdown("### 🔌 Connection")
        st.code(BACKEND_URL, language="text")
        
        api_status = "✅ Configured" if API_KEY else "❌ Missing"
        st.markdown(f"**API Key Status:** {api_status}")
        
        st.markdown("---")
        st.markdown("### ⚙️ Quick Actions")
        if st.button("🔄 Refresh Health Status", use_container_width=True):
            st.session_state.last_health = check_backend_health()
        
        if st.button("🗑️ Clear History", use_container_width=True):
            st.session_state.history = []
            st.rerun()


def render_header_and_health() -> None:
    """Render the main header and backend status panel."""
    # Header
    col1, col2 = st.columns([1, 4])
    with col1:
        st.image("https://cdn-icons-png.flaticon.com/512/2040/2040946.png", width=100)
    with col2:
        st.markdown('<p class="gradient-header">AI Auto Categorization</p>', unsafe_allow_html=True)
        st.markdown('<p class="subtitle">LLM-Powered Support Ticket Classification Platform</p>', unsafe_allow_html=True)

    st.markdown(f"*{datetime.now().strftime('%B %d, %Y - %H:%M:%S')}*")
    st.markdown("---")

    # Backend Status Panel
    st.markdown("### 📊 System Status")
    
    # Auto-fetch health on load if not present
    if st.session_state.last_health is None:
        st.session_state.last_health = check_backend_health()
        
    health_data = st.session_state.last_health
    
    if health_data:
        h_col1, h_col2, h_col3, h_col4 = st.columns(4)
        with h_col1:
            st.metric("Status", "Healthy", delta="Online")
        with h_col2:
            st.metric("Model", health_data.get("model", "N/A"))
        with h_col3:
            st.metric("Examples Loaded", health_data.get("examples_loaded", 0))
        with h_col4:
            st.metric("Version", health_data.get("version", "1.0.0"))
    else:
        st.error("Backend is currently offline or unreachable. Please verify `BACKEND_URL` and ensure FastAPI is running.")


def render_confidence_bar(confidence: float) -> None:
    """Render a custom progress bar mapping confidence to color."""
    percent = int(confidence * 100)
    
    if percent >= 90:
        color_class = "progress-green"
    elif percent >= 70:
        color_class = "progress-orange"
    else:
        color_class = "progress-red"
        
    html = f"""
    <div style="font-weight: 600; margin-bottom: 4px;">Confidence Score: {percent}%</div>
    <div class="progress-wrapper">
        <div class="progress-fill {color_class}" style="width: {percent}%;"></div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


def render_results_dashboard(result: Dict[str, Any]) -> None:
    """Render the classification results in a dashboard layout."""
    st.markdown("---")
    st.markdown("### 🎯 Classification Results")
    
    # Manual Review Alert
    if result.get("low_confidence", False):
        st.warning("⚠ **Manual Review Required:** The model is not highly confident in this classification.")
    else:
        st.success("✅ **Classification Successful:** High confidence achieved.")
        
    render_confidence_bar(result.get("confidence", 0.0))
    st.markdown("<br>", unsafe_allow_html=True)
    
    r_col1, r_col2, r_col3 = st.columns(3)
    with r_col1:
        st.metric("Category", result.get("category", "N/A"))
    with r_col2:
        st.metric("Subcategory", result.get("subcategory", "N/A"))
    with r_col3:
        st.metric("Priority", result.get("priority", "N/A"))
        
    st.markdown("<br>", unsafe_allow_html=True)
    m_col1, m_col2, m_col3 = st.columns(3)
    with m_col1:
        st.metric("Model Used", result.get("model", "N/A"))
    with m_col2:
        st.metric("Latency (ms)", result.get("latency_ms", 0))
    with m_col3:
        # Displaying request ID using caption to avoid overflow in metric card
        st.markdown("**Request ID**")
        st.caption(f"`{result.get('request_id', 'N/A')}`")

    # Reasoning / Expanded view
    with st.expander("🔍 View AI Reasoning & Raw JSON Response"):
        if "reasoning" in result:
            st.markdown(f"**Reasoning:** {result['reasoning']}")
        st.json(result)


def render_history_table() -> None:
    """Render the session classification history as a dataframe."""
    st.markdown("---")
    st.markdown("### 📋 Classification History")
    
    if not st.session_state.history:
        st.info("No tickets classified in this session yet.")
        return
        
    df = pd.DataFrame(st.session_state.history)
    st.dataframe(
        df[["timestamp", "ticket_id", "category", "priority", "confidence"]],
        use_container_width=True,
        hide_index=True
    )
    
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📥 Download History (CSV)",
        data=csv,
        file_name="classification_history.csv",
        mime="text/csv",
    )


# ======================================================================= #
# Main Application Flow                                                     #
# ======================================================================= #

def main() -> None:
    """Main entry point for the Streamlit application."""
    init_session_state()
    inject_custom_css()
    
    render_sidebar()
    render_header_and_health()
    
    st.markdown("### 📝 Ticket Submission Form")
    
    with st.form("ticket_form", clear_on_submit=False):
        ticket_id = st.text_input("Ticket ID*", placeholder="e.g. T-1001")
        title = st.text_input("Title*", placeholder="Brief summary of the issue")
        description = st.text_area(
            "Description*", 
            placeholder="Full details of the support request...",
            height=150,
            max_chars=2000
        )
        
        submitted = st.form_submit_button("Classify Ticket", type="primary")
        
    if submitted:
        if not ticket_id or not title or not description:
            st.error("All fields (Ticket ID, Title, Description) are required.")
        elif not API_KEY:
            st.error("API Key is missing. Please configure it in your `.env` file.")
        else:
            with st.spinner("🤖 Analyzing ticket using few-shot classification..."):
                time.sleep(0.5) # Slight UX delay for animation visibility
                result, error = classify_ticket(ticket_id, title, description)
                
            if error:
                st.error(error)
            elif result:
                # Add to history
                history_entry = {
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "ticket_id": ticket_id,
                    "category": result.get("category"),
                    "priority": result.get("priority"),
                    "confidence": f"{int(result.get('confidence', 0)*100)}%",
                }
                # Insert at top of history
                st.session_state.history.insert(0, history_entry)
                
                # Render results
                render_results_dashboard(result)

    render_history_table()


if __name__ == "__main__":
    main()
