"""Streamlit frontend for Eve Online RAG Bot."""

import os
import requests
import streamlit as st

# Page configuration
st.set_page_config(
    page_title="Eve Online RAG Bot",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom styling
st.markdown("""
    <style>
    .main {
        max-width: 1200px;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        display: flex;
        flex-direction: column;
    }
    .user-message {
        background-color: #e3f2fd;
        margin-left: 20%;
        margin-right: 0;
    }
    .bot-message {
        background-color: #f5f5f5;
        margin-left: 0;
        margin-right: 20%;
    }
    </style>
""", unsafe_allow_html=True)

# Configuration
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")
HEALTH_CHECK_ENDPOINT = f"{BACKEND_URL}/api/v1/health"
CHAT_ENDPOINT = f"{BACKEND_URL}/api/v1/chat"

# Session state initialization
if "messages" not in st.session_state:
    st.session_state.messages = []
if "backend_available" not in st.session_state:
    st.session_state.backend_available = False


def check_backend_health():
    """Check if backend service is available."""
    try:
        response = requests.get(HEALTH_CHECK_ENDPOINT, timeout=3)
        return response.status_code == 200
    except Exception:
        return False


def send_message(message: str) -> str | None:
    """
    Send message to backend and get response.
    
    Args:
        message: User message to send
        
    Returns:
        Bot response or None if request fails
    """
    try:
        response = requests.post(
            CHAT_ENDPOINT,
            json={"message": message},
            timeout=30
        )
        if response.status_code == 200:
            data = response.json()
            return data.get("response", "No response received")
        else:
            return f"Error: Server returned status {response.status_code}"
    except requests.exceptions.Timeout:
        return "Error: Request timeout. Please try again."
    except requests.exceptions.ConnectionError:
        return f"Error: Cannot connect to backend at {BACKEND_URL}"
    except Exception as e:
        return f"Error: {str(e)}"


# Main interface
st.title("🚀 Eve Online RAG Bot")
st.markdown("Chat with an AI about Eve Online!")

# Sidebar
with st.sidebar:
    st.header("Settings")
    
    # Backend status
    if check_backend_health():
        st.success("✅ Backend connected")
        st.session_state.backend_available = True
    else:
        st.error("❌ Backend unavailable")
        st.session_state.backend_available = False
        st.info(f"Backend URL: {BACKEND_URL}")
    
    # Clear chat history button
    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.rerun()
    
    st.divider()
    st.caption("Eve Online RAG Bot uses retrieval-augmented generation to provide accurate information about Eve Online.")

# Display chat history
st.subheader("Chat History")

# Create a container for messages
messages_container = st.container()

with messages_container:
    for message in st.session_state.messages:
        if message["role"] == "user":
            with st.chat_message("user"):
                st.write(message["content"])
        else:
            with st.chat_message("assistant"):
                st.write(message["content"])

# Chat input
st.divider()

if not st.session_state.backend_available:
    st.warning(
        "⚠️ Backend service is not available. Please ensure the backend is running."
    )
    user_input = st.text_input(
        "Message:",
        disabled=True,
        placeholder="Waiting for backend to be available..."
    )
else:
    user_input = st.text_input(
        "Message:",
        placeholder="Ask me about Eve Online..."
    )

# Process user input
if user_input:
    # Add user message to history
    st.session_state.messages.append({
        "role": "user",
        "content": user_input
    })
    
    # Show user message
    with messages_container:
        with st.chat_message("user"):
            st.write(user_input)
    
    # Get bot response
    with st.spinner("Thinking..."):
        bot_response = send_message(user_input)
    
    # Add bot message to history
    if bot_response:
        st.session_state.messages.append({
            "role": "assistant",
            "content": bot_response
        })
    
    # Show bot response
    with messages_container:
        with st.chat_message("assistant"):
            st.write(bot_response or "No response received")
    
    # Rerun to clear input and update UI
    st.rerun()
